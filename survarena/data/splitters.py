from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


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
    encoded = np.asarray(event, dtype=np.int8).tobytes()
    return sha256(encoded).hexdigest()


def _expected_split_manifest_payload(
    *,
    split_strategy: str,
    n_samples: int,
    event: np.ndarray,
    seeds: list[int],
    outer_folds: int,
    outer_repeats: int,
) -> dict:
    payload: dict[str, object] = {
        "version": _SPLIT_MANIFEST_VERSION,
        "split_strategy": split_strategy,
        "n_samples": int(n_samples),
        "event_fingerprint": _event_fingerprint(event),
        "event_rate": float(np.mean(event)),
        "seeds": [int(seed) for seed in seeds],
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

    for repeat, seed in enumerate(seeds[:repeats]):
        skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(indices, event)):
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
) -> list[SplitDefinition]:
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    indices = np.arange(n_samples)
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=1.0 - train_ratio,
        stratify=event,
        random_state=seed,
    )
    val_size_in_holdout = val_ratio / (1.0 - train_ratio)
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
) -> list[SplitDefinition]:
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

    def _validate_event_stratification(
        splits_to_check: list[SplitDefinition],
        event_labels: np.ndarray,
        *,
        tolerance: float = 0.03,
    ) -> None:
        overall_rate = float(np.mean(event_labels))
        for split in splits_to_check:
            train_rate = float(np.mean(event_labels[split.train_idx]))
            test_rate = float(np.mean(event_labels[split.test_idx]))
            if abs(train_rate - overall_rate) > tolerance:
                raise ValueError(
                    f"Train split is not event-stratified enough for {split.split_id}: "
                    f"train_rate={train_rate:.4f}, overall_rate={overall_rate:.4f}, tolerance={tolerance:.4f}"
                )
            if abs(test_rate - overall_rate) > tolerance:
                raise ValueError(
                    f"Test split is not event-stratified enough for {split.split_id}: "
                    f"test_rate={test_rate:.4f}, overall_rate={overall_rate:.4f}, tolerance={tolerance:.4f}"
                )
            if split.val_idx is not None:
                val_rate = float(np.mean(event_labels[split.val_idx]))
                if abs(val_rate - overall_rate) > tolerance:
                    raise ValueError(
                        f"Validation split is not event-stratified enough for {split.split_id}: "
                        f"val_rate={val_rate:.4f}, overall_rate={overall_rate:.4f}, tolerance={tolerance:.4f}"
                    )

    manifest_payload = _expected_split_manifest_payload(
        split_strategy=split_strategy,
        n_samples=n_samples,
        event=event,
        seeds=seeds,
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
    )
    manifest_path = _split_manifest_path(root, task_id)
    if manifest_path.exists():
        manifest = read_split_manifest(manifest_path)
        if manifest.get("manifest_payload") == manifest_payload:
            split_ids = [str(split_id) for split_id in manifest.get("split_ids", [])]
            loaded_splits = [read_split(_split_file_path(root, task_id, split_id)) for split_id in split_ids]
            _validate_split_integrity(loaded_splits, n_samples)
            _validate_event_stratification(loaded_splits, event)
            return loaded_splits

    if split_strategy == "repeated_nested_cv":
        splits = create_repeated_nested_outer_splits(
            n_samples=n_samples,
            event=event,
            seeds=seeds,
            outer_folds=outer_folds,
            repeats=outer_repeats,
        )
    elif split_strategy == "fixed_split":
        splits = create_fixed_split(n_samples=n_samples, event=event, seed=seeds[0])
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    _validate_split_integrity(splits, n_samples)
    _validate_event_stratification(splits, event)
    for split in splits:
        write_split(root, task_id, split)
    write_split_manifest(root, task_id, manifest_payload, [split.split_id for split in splits])
    return splits
