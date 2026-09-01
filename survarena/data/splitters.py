from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split

try:  # StratifiedGroupKFold is not present in older pinned sklearn versions.
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - depends on the pinned sklearn version.
    StratifiedGroupKFold = None


_SPLIT_MANIFEST_FILENAME = "manifest.json"
_SPLIT_MANIFEST_VERSION = 1


@dataclass(slots=True)
class SplitDefinition:
    split_id: str
    seed: int
    repeat: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    val_idx: np.ndarray | None = None

    def to_json_dict(self) -> dict:
        payload = {
            "split_id": self.split_id,
            "seed": self.seed,
            "repeat": self.repeat,
            "fold": self.fold,
            "train_idx": self.train_idx.tolist(),
            "test_idx": self.test_idx.tolist(),
            "stratification": "event",
        }
        if self.val_idx is not None:
            payload["val_idx"] = self.val_idx.tolist()
        return payload


def _split_file_path(root: Path, task_id: str, split_id: str) -> Path:
    return root / "data" / "splits" / task_id / f"{split_id}.json"


def _split_manifest_path(root: Path, task_id: str) -> Path:
    return root / "data" / "splits" / task_id / _SPLIT_MANIFEST_FILENAME


def _event_fingerprint(event: np.ndarray) -> str:
    """Stable hash of event labels; requires strict 0/1 (or bool) to avoid int8 overflow collisions."""
    arr = np.asarray(event)
    if not np.isin(arr, [0, 1, False, True]).all():
        raise ValueError("Event labels must be binary 0/1 (or bool) for deterministic split fingerprinting.")
    encoded = arr.astype(np.uint8, copy=False).tobytes()
    return sha256(encoded).hexdigest()


def _content_fingerprint(values: object) -> str:
    """Deterministic, row-order-sensitive sha256 of array-like / DataFrame content.

    M5: splits are stored as *positional* indices, so the reuse fingerprint must pin the
    actual data (feature matrix and time), not just the event vector. Otherwise permuted or
    silently-mutated rows reuse stale indices. Hashing is row-order-sensitive (a permutation
    changes the digest) and NaN-safe (NaN has a stable IEEE-754 bit pattern; object columns
    map NaN/None to a sentinel token). Numeric columns are hashed from their contiguous bytes;
    non-numeric columns fall back to a canonical string representation.
    """
    hasher = sha256()

    def _update_numeric(array: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode("utf-8"))
        hasher.update(str(contiguous.shape).encode("utf-8"))
        hasher.update(contiguous.tobytes())

    def _update_object(series: pd.Series) -> None:
        # Canonical string form; NaN/None collapse to a fixed sentinel so hashing never crashes.
        def _canonical(value: object) -> str:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return "\x00__nan__"
            return str(value)

        text = "\x01".join(series.astype("object").map(_canonical))
        hasher.update(text.encode("utf-8"))

    if isinstance(values, pd.DataFrame):
        # Column order is part of the identity of the positional index layout.
        hasher.update("\x02".join(str(col) for col in values.columns).encode("utf-8"))
        for column in values.columns:
            series = values[column]
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                _update_numeric(series.to_numpy(dtype=np.float64))
            else:
                _update_object(series)
            hasher.update(b"\x03")
    else:
        array = np.asarray(values)
        if array.dtype.kind in ("f", "i", "u", "b"):
            _update_numeric(array)
        else:
            _update_object(pd.Series(array.reshape(-1)))
    return hasher.hexdigest()


def _expected_split_manifest_payload(
    *,
    split_strategy: str,
    n_samples: int,
    event: np.ndarray,
    seeds: list[int],
    outer_folds: int,
    outer_repeats: int,
    X: object | None = None,
    time: np.ndarray | None = None,
) -> dict:
    payload: dict[str, object] = {
        "version": _SPLIT_MANIFEST_VERSION,
        "split_strategy": split_strategy,
        "n_samples": int(n_samples),
        "event_fingerprint": _event_fingerprint(event),
        "event_rate": float(np.mean(event)),
        "seeds": [int(seed) for seed in seeds],
        # M5: pin the feature matrix and time vector so reusing positional indices against
        # permuted/mutated data triggers the reuse-mismatch path. None when the caller has not
        # yet wired X/time (older manifests also lack these keys -> dict inequality -> mismatch).
        "x_fingerprint": None if X is None else _content_fingerprint(X),
        "time_fingerprint": None if time is None else _content_fingerprint(np.asarray(time, dtype=float)),
    }
    if split_strategy == "repeated_nested_cv":
        payload.update(
            {
                "outer_folds": int(outer_folds),
                "outer_repeats": int(outer_repeats),
                "seed_policy": "one_seed_per_repeat",
            }
        )
    elif split_strategy == "fixed_split":
        payload.update(
            {
                "outer_folds": None,
                "outer_repeats": None,
                "seed_policy": "single_fixed_split",
            }
        )
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")
    return payload


def _split_manifest_payload_diff(observed: object, expected: Mapping[str, object]) -> list[str]:
    """Return a deterministic, bounded compatibility diff for a cached manifest payload."""

    def _bounded(value: object, *, limit: int = 160) -> str:
        rendered = repr(value)
        return rendered if len(rendered) <= limit else f"{rendered[: limit - 3]}..."

    if not isinstance(observed, Mapping):
        return [f"manifest_payload has type {type(observed).__name__}; expected a mapping"]

    observed_by_name = {str(key): value for key, value in observed.items()}
    expected_keys = set(expected)
    observed_keys = set(observed_by_name)
    differences = [
        f"missing field '{key}' (expected {_bounded(expected[key])})" for key in sorted(expected_keys - observed_keys)
    ]
    differences.extend(
        f"unexpected field '{key}' (cached {_bounded(observed_by_name[key])})"
        for key in sorted(observed_keys - expected_keys)
    )
    differences.extend(
        f"changed field '{key}': cached={_bounded(observed_by_name[key])}, expected={_bounded(expected[key])}"
        for key in sorted(expected_keys & observed_keys)
        if observed_by_name[key] != expected[key]
    )
    return differences or ["manifest payload differs but no field-level difference could be rendered"]


def write_split_manifest(root: Path, task_id: str, manifest_payload: dict, split_ids: list[str]) -> None:
    path = _split_manifest_path(root, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_payload": manifest_payload,
        "split_ids": list(split_ids),
        "split_count": int(len(split_ids)),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_split_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_split(root: Path, task_id: str, split: SplitDefinition) -> None:
    path = _split_file_path(root, task_id, split.split_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(split.to_json_dict(), handle, indent=2)


def read_split(path: Path) -> SplitDefinition:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SplitDefinition(
        split_id=payload["split_id"],
        seed=int(payload["seed"]),
        repeat=int(payload["repeat"]),
        fold=int(payload["fold"]),
        train_idx=np.asarray(payload["train_idx"], dtype=int),
        test_idx=np.asarray(payload["test_idx"], dtype=int),
        val_idx=np.asarray(payload["val_idx"], dtype=int) if "val_idx" in payload else None,
    )


def create_repeated_nested_outer_splits(
    *,
    n_samples: int,
    event: np.ndarray,
    seeds: list[int],
    outer_folds: int,
    repeats: int,
    groups: np.ndarray | None = None,
) -> list[SplitDefinition]:
    if outer_folds < 2:
        raise ValueError("outer_folds must be >= 2 for repeated nested CV.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1 for repeated nested CV.")
    if len(seeds) < repeats:
        raise ValueError(
            f"Need at least {repeats} seeds for repeated nested CV, but received {len(seeds)}."
        )

    splits: list[SplitDefinition] = []
    indices = np.arange(n_samples)
    groups_arr = None if groups is None else np.asarray(groups)

    for repeat, seed in enumerate(seeds[:repeats]):
        if groups_arr is not None:
            # H6: group-aware outer CV so no group_id (e.g. subject) spans multiple folds.
            if StratifiedGroupKFold is not None:
                # Preferred: keep event stratification AND group-disjointness.
                splitter = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
            else:
                # Fallback for older sklearn: group-disjoint folds without event stratification.
                splitter = GroupKFold(n_splits=outer_folds)
            fold_iter = splitter.split(indices, event, groups_arr)
        else:
            # No groups: preserve the exact prior StratifiedKFold behavior.
            splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
            fold_iter = splitter.split(indices, event)
        for fold, (train_idx, test_idx) in enumerate(fold_iter):
            split_id = f"repeat_{repeat}_fold_{fold}"
            splits.append(
                SplitDefinition(
                    split_id=split_id,
                    seed=int(seed),
                    repeat=repeat,
                    fold=fold,
                    train_idx=np.asarray(train_idx, dtype=int),
                    test_idx=np.asarray(test_idx, dtype=int),
                )
            )
    return splits


def create_fixed_split(
    *,
    n_samples: int,
    event: np.ndarray,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    groups: np.ndarray | None = None,
) -> list[SplitDefinition]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    indices = np.arange(n_samples)
    val_size_in_holdout = val_ratio / (1.0 - train_ratio)
    if groups is not None:
        # H6: group-disjoint holdout so a subject cannot land in both train and test/val.
        groups_arr = np.asarray(groups)
        outer = GroupShuffleSplit(n_splits=1, test_size=1.0 - train_ratio, random_state=seed)
        train_pos, holdout_pos = next(outer.split(indices, event, groups_arr))
        train_idx = indices[train_pos]
        holdout_idx = indices[holdout_pos]
        inner = GroupShuffleSplit(n_splits=1, test_size=1.0 - val_size_in_holdout, random_state=seed)
        val_pos, test_pos = next(inner.split(holdout_idx, event[holdout_idx], groups_arr[holdout_idx]))
        val_idx = holdout_idx[val_pos]
        test_idx = holdout_idx[test_pos]
    else:
        # No groups: preserve the exact prior event-stratified train_test_split behavior.
        train_idx, holdout_idx = train_test_split(
            indices,
            test_size=1.0 - train_ratio,
            stratify=event,
            random_state=seed,
        )
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=1.0 - val_size_in_holdout,
            stratify=event[holdout_idx],
            random_state=seed,
        )
    return [
        SplitDefinition(
            split_id="fixed_split_0",
            seed=seed,
            repeat=0,
            fold=0,
            train_idx=np.asarray(train_idx, dtype=int),
            test_idx=np.asarray(test_idx, dtype=int),
            val_idx=np.asarray(val_idx, dtype=int),
        )
    ]


def load_or_create_splits(
    *,
    root: Path,
    task_id: str,
    split_strategy: str,
    n_samples: int,
    event: np.ndarray,
    seeds: list[int],
    outer_folds: int = 5,
    outer_repeats: int = 3,
    regenerate_on_mismatch: bool = False,
    groups: np.ndarray | None = None,
    X: object | None = None,
    time: np.ndarray | None = None,
) -> list[SplitDefinition]:
    # H6/M5: `groups`, `X` and `time` are optional with backward-compatible defaults so existing
    # callers behave identically. The benchmark runner and compare API now pass groups (from
    # metadata.group_col) for group-disjoint splitting, and X/time for content-aware split-cache
    # invalidation.
    groups_arr = None if groups is None else np.asarray(groups)

    def _validate_split_integrity(splits_to_check: list[SplitDefinition], n_rows: int) -> None:
        seen_split_ids: set[str] = set()
        for split in splits_to_check:
            if split.split_id in seen_split_ids:
                raise ValueError(f"Duplicate split_id detected: {split.split_id}")
            seen_split_ids.add(split.split_id)

            train_idx = np.asarray(split.train_idx, dtype=int)
            test_idx = np.asarray(split.test_idx, dtype=int)
            val_idx = np.asarray(split.val_idx, dtype=int) if split.val_idx is not None else None

            for name, idx in (("train", train_idx), ("test", test_idx), ("val", val_idx)):
                if idx is None:
                    continue
                if idx.size == 0:
                    raise ValueError(f"{name} indices are empty for {split.split_id}")
                if np.any(idx < 0) or np.any(idx >= n_rows):
                    raise ValueError(f"{name} indices are out of bounds for {split.split_id}")
                if np.unique(idx).size != idx.size:
                    raise ValueError(f"{name} indices contain duplicates for {split.split_id}")

            if np.intersect1d(train_idx, test_idx).size > 0:
                raise ValueError(f"Train/test overlap detected for {split.split_id}")
            if val_idx is not None:
                if np.intersect1d(train_idx, val_idx).size > 0:
                    raise ValueError(f"Train/validation overlap detected for {split.split_id}")
                if np.intersect1d(test_idx, val_idx).size > 0:
                    raise ValueError(f"Test/validation overlap detected for {split.split_id}")

            # H6: when groups are available, no group_id may straddle train/test/val partitions.
            if groups_arr is not None:
                train_groups = set(np.unique(groups_arr[train_idx]).tolist())
                test_groups = set(np.unique(groups_arr[test_idx]).tolist())
                if train_groups & test_groups:
                    raise ValueError(f"Train/test group overlap detected for {split.split_id}")
                if val_idx is not None:
                    val_groups = set(np.unique(groups_arr[val_idx]).tolist())
                    if train_groups & val_groups:
                        raise ValueError(f"Train/validation group overlap detected for {split.split_id}")
                    if test_groups & val_groups:
                        raise ValueError(f"Test/validation group overlap detected for {split.split_id}")

    def _validate_event_stratification(
        splits_to_check: list[SplitDefinition],
        event_labels: np.ndarray,
        *,
        abs_floor: float = 0.03,
    ) -> None:
        overall_rate = float(np.mean(event_labels))

        def _allowed_deviation(n_fold: int) -> float:
            # L3: a fixed absolute tolerance (old 0.03) is too loose at low event rates and too
            # strict on tiny folds. Allow the largest of:
            #   - a modest absolute floor (`abs_floor`),
            #   - a relative tolerance (25% of the overall event rate), and
            #   - 4 binomial sampling standard errors for this fold size
            #     (4 * sqrt(p*(1-p)/n_fold)) so small folds get a wider, size-aware allowance.
            # This still fires on genuinely degenerate folds (e.g. ~0 events) while tolerating
            # statistically-normal stratification drift.
            relative = 0.25 * overall_rate
            standard_error = math.sqrt(max(overall_rate * (1.0 - overall_rate), 0.0) / max(n_fold, 1))
            return max(abs_floor, relative, 4.0 * standard_error)

        def _check(name: str, idx: np.ndarray, split_id: str) -> None:
            n_fold = int(np.asarray(idx).size)
            rate = float(np.mean(event_labels[idx]))
            allowed = _allowed_deviation(n_fold)
            if abs(rate - overall_rate) > allowed:
                raise ValueError(
                    f"{name} split is not event-stratified enough for {split_id}: "
                    f"rate={rate:.4f}, overall_rate={overall_rate:.4f}, allowed_deviation={allowed:.4f}"
                )

        for split in splits_to_check:
            _check("Train", split.train_idx, split.split_id)
            _check("Test", split.test_idx, split.split_id)
            if split.val_idx is not None:
                _check("Validation", split.val_idx, split.split_id)

    manifest_payload = _expected_split_manifest_payload(
        split_strategy=split_strategy,
        n_samples=n_samples,
        event=event,
        seeds=seeds,
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
        X=X,
        time=time,
    )
    manifest_path = _split_manifest_path(root, task_id)
    if manifest_path.exists():
        manifest = read_split_manifest(manifest_path)
        # M5: dict equality includes the new x_fingerprint/time_fingerprint keys, so older
        # manifests lacking them (or content-mismatched data) fall through to the reuse-mismatch
        # path below rather than silently reusing stale positional indices.
        if manifest.get("manifest_payload") == manifest_payload:
            split_ids = [str(split_id) for split_id in manifest.get("split_ids", [])]
            loaded_splits = [read_split(_split_file_path(root, task_id, split_id)) for split_id in split_ids]
            _validate_split_integrity(loaded_splits, n_samples)
            _validate_event_stratification(loaded_splits, event)
            return loaded_splits
        if not regenerate_on_mismatch:
            differences = _split_manifest_payload_diff(manifest.get("manifest_payload"), manifest_payload)
            raise ValueError(
                "Existing split manifest payload mismatch for "
                f"task_id='{task_id}' at '{manifest_path}'.\n"
                + "\n".join(f"- {difference}" for difference in differences)
                + "\nDeterministic contract violation: refusing to regenerate splits automatically. "
                "Re-run the benchmark with --regenerate-splits to deliberately replace these split artifacts."
            )

    if split_strategy == "repeated_nested_cv":
        splits = create_repeated_nested_outer_splits(
            n_samples=n_samples,
            event=event,
            seeds=seeds,
            outer_folds=outer_folds,
            repeats=outer_repeats,
            groups=groups_arr,
        )
    elif split_strategy == "fixed_split":
        splits = create_fixed_split(n_samples=n_samples, event=event, seed=seeds[0], groups=groups_arr)
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    _validate_split_integrity(splits, n_samples)
    _validate_event_stratification(splits, event)
    for split in splits:
        write_split(root, task_id, split)
    write_split_manifest(root, task_id, manifest_payload, [split.split_id for split in splits])
    return splits
