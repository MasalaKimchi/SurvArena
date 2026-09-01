from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from survarena.data.feature_roles import is_low_cardinality_numeric_categorical


@dataclass(slots=True)
class RobustnessTrack:
    track_id: str
    kind: str
    severity: float
    params: dict[str, Any]


def resolve_robustness_tracks(
    config: dict[str, Any] | None,
    *,
    dataset_id: str,
    feature_columns: list[str],
    seed_pool: list[int],
) -> list[RobustnessTrack]:
    if not config or not bool(config.get("enabled", False)):
        return [RobustnessTrack(track_id="baseline", kind="baseline", severity=0.0, params={})]
    tracks = list(config.get("tracks", ["missingness", "covariate_noise", "label_noise"]))
    severities = [float(level) for level in config.get("severity_levels", [0.05, 0.15])]
    resolved: list[RobustnessTrack] = [RobustnessTrack(track_id="baseline", kind="baseline", severity=0.0, params={})]
    for kind in tracks:
        if str(kind) == "baseline":
            continue
        for severity in severities:
            pct = int(round(severity * 100))
            track_id = f"{kind}_s{pct}"
            resolved.append(
                RobustnessTrack(
                    track_id=track_id,
                    kind=str(kind),
                    severity=float(severity),
                    params={
                        "dataset_id": dataset_id,
                        "feature_columns": list(feature_columns),
                        "seed_pool": list(seed_pool),
                    },
                )
            )
    return resolved


def _rng_for_track(track: RobustnessTrack, seed: int) -> np.random.Generator:
    payload = f"{track.track_id}:{seed}".encode("utf-8")
    digest = sha256(payload).digest()
    int_seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(int_seed)


def apply_robustness_track(X: pd.DataFrame, *, track: RobustnessTrack, split: Any, seed: int) -> pd.DataFrame:
    if track.kind == "baseline":
        return X
    X_out = X.copy(deep=True)
    rng = _rng_for_track(track, seed)
    test_idx = np.asarray(split.test_idx, dtype=int)
    # Only perturb genuinely CONTINUOUS features. Low-cardinality integer columns (e.g. a stage
    # code) are treated as CATEGORICAL by the preprocessor (see data/feature_roles.py /
    # data/preprocess.py); adding Gaussian noise or median-imputing them yields off-vocabulary
    # values that map to an all-zero one-hot row and are silently dropped. We reuse the project's
    # categorical-detection helper (and exclude bool columns, also treated as categorical) so those
    # columns are left unperturbed rather than corrupted.
    numeric_cols = [
        col
        for col in X_out.select_dtypes(include=[np.number]).columns
        if not pd.api.types.is_bool_dtype(X_out[col]) and not is_low_cardinality_numeric_categorical(X_out[col])
    ]
    if not numeric_cols:
        return X_out

    if track.kind == "missingness":
        n = len(test_idx)
        mask = rng.random((n, len(numeric_cols))) < float(track.severity)
        test_frame = X_out.iloc[test_idx].copy()
        test_values = test_frame[numeric_cols].to_numpy(dtype=float)
        test_values[mask] = np.nan
        medians = X_out.iloc[split.train_idx][numeric_cols].median(numeric_only=True).fillna(0.0).to_numpy(dtype=float)
        rows, cols = np.where(np.isnan(test_values))
        if rows.size:
            test_values[rows, cols] = medians[cols]
        X_out.loc[test_frame.index, numeric_cols] = test_values
        return X_out

    if track.kind == "covariate_noise":
        std = X_out.iloc[split.train_idx][numeric_cols].std(numeric_only=True).replace(0.0, 1.0).to_numpy(dtype=float)
        noise = rng.normal(loc=0.0, scale=std * float(track.severity), size=(len(test_idx), len(numeric_cols)))
        test_values = X_out.iloc[test_idx][numeric_cols].to_numpy(dtype=float)
        X_out.iloc[test_idx, [X_out.columns.get_loc(col) for col in numeric_cols]] = test_values + noise
        return X_out

    if track.kind == "label_noise":
        # Label noise is handled by the runner when this track is enabled.
        return X_out

    return X_out


def apply_label_noise(event: np.ndarray, *, track: RobustnessTrack, split: Any, seed: int) -> np.ndarray:
    """Flip a fraction of TEST-set event labels (test-time outcome corruption).

    This is a TEST-TIME label-corruption track: it measures how stable a metric is when the
    *evaluation* outcomes are noisy. It is NOT training-label robustness -- only labels at the
    test indices are flipped, so training labels are left untouched. The flip behavior itself is
    intentional and unchanged.
    """
    event_out = np.asarray(event, dtype=np.int32).copy()
    if track.kind != "label_noise" or track.severity <= 0.0:
        return event_out
    rng = _rng_for_track(track, seed)
    test_idx = np.asarray(split.test_idx, dtype=int)
    flips = rng.random(len(test_idx)) < float(track.severity)
    target_idx = test_idx[flips]
    event_out[target_idx] = 1 - event_out[target_idx]
    return event_out
