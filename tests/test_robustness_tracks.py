from __future__ import annotations

import numpy as np
import pandas as pd

from survarena.data.robustness import apply_label_noise, apply_robustness_track, resolve_robustness_tracks
from survarena.data.splitters import SplitDefinition


def _split() -> SplitDefinition:
    return SplitDefinition(
        split_id="s0",
        seed=11,
        repeat=0,
        fold=0,
        train_idx=np.asarray([0, 1, 2, 3], dtype=int),
        test_idx=np.asarray([4, 5], dtype=int),
    )


def test_resolve_robustness_tracks_returns_baseline_when_disabled() -> None:
    tracks = resolve_robustness_tracks(None, dataset_id="d", feature_columns=["x"], seed_pool=[11])
    assert len(tracks) == 1
    assert tracks[0].track_id == "baseline"


def test_covariate_noise_track_changes_test_rows_only() -> None:
    tracks = resolve_robustness_tracks(
        {"enabled": True, "tracks": ["covariate_noise"], "severity_levels": [0.2]},
        dataset_id="d",
        feature_columns=["x", "y"],
        seed_pool=[11],
    )
    track = [t for t in tracks if t.kind == "covariate_noise"][0]
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    out = apply_robustness_track(X, track=track, split=_split(), seed=11)
    assert np.allclose(out.iloc[:4].to_numpy(), X.iloc[:4].to_numpy())
    assert not np.allclose(out.iloc[4:].to_numpy(), X.iloc[4:].to_numpy())


def test_label_noise_flips_subset_of_test_labels() -> None:
    tracks = resolve_robustness_tracks(
        {"enabled": True, "tracks": ["label_noise"], "severity_levels": [1.0]},
        dataset_id="d",
        feature_columns=["x"],
        seed_pool=[11],
    )
    track = [t for t in tracks if t.kind == "label_noise"][0]
    event = np.asarray([0, 0, 1, 1, 0, 1], dtype=int)
    noisy = apply_label_noise(event, track=track, split=_split(), seed=11)
    assert np.array_equal(noisy[:4], event[:4])
    assert np.array_equal(noisy[4:], 1 - event[4:])
