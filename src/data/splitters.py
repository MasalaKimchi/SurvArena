from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


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
    splits: list[SplitDefinition] = []
    indices = np.arange(n_samples)

    for repeat in range(repeats):
        seed = seeds[repeat % len(seeds)]
        skf = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(indices, event)):
            split_id = f"repeat_{repeat}_fold_{fold}"
            splits.append(
                SplitDefinition(
                    split_id=split_id,
                    seed=seed,
                    repeat=repeat,
                    fold=fold,
                    train_idx=train_idx,
                    test_idx=test_idx,
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

    split_dir = root / "data" / "splits" / task_id
    if split_dir.exists():
        files = sorted(split_dir.glob("*.json"))
        if files:
            loaded_splits = [read_split(path) for path in files]
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

    _validate_event_stratification(splits, event)
    for split in splits:
        write_split(root, task_id, split)
    return splits
