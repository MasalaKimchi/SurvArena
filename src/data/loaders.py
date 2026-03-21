from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from src.data.profiling import build_dataset_diagnostics, infer_feature_metadata, summarize_feature_types
from src.data.schema import DatasetMetadata, SurvivalDataset


def _load_dataset_config(configs_dir: Path, dataset_id: str) -> dict:
    config_path = configs_dir / "datasets" / f"{dataset_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_from_sksurv(dataset_id: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from sksurv import datasets as sk_datasets

    mapping: dict[str, Callable[[], tuple[pd.DataFrame, np.ndarray]]] = {
        "gbsg2": sk_datasets.load_gbsg2,
        "flchain": sk_datasets.load_flchain,
        "whas500": sk_datasets.load_whas500,
    }
    if dataset_id not in mapping:
        raise ValueError(f"No scikit-survival loader mapped for dataset '{dataset_id}'.")
    X, y = mapping[dataset_id]()
    names = y.dtype.names or ()
    bool_fields = [name for name in names if y.dtype[name].kind == "b"]
    numeric_fields = [name for name in names if y.dtype[name].kind in {"i", "u", "f"}]
    if not bool_fields or not numeric_fields:
        raise ValueError(f"Could not infer event/time columns from structured target for '{dataset_id}'.")

    event_field = bool_fields[0]
    time_field = numeric_fields[0]
    event_raw = y[event_field].astype(bool)
    if "cens" in event_field.lower():
        event = (~event_raw).astype(np.int32)
    else:
        event = event_raw.astype(np.int32)
    time = y[time_field].astype(np.float64)
    return X, time, event


def _load_support_pycox() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from pycox.datasets import support

    frame = support.read_df()
    event = frame["event"].to_numpy(dtype=np.int32)
    time = frame["duration"].to_numpy(dtype=np.float64)
    X = frame.drop(columns=["event", "duration"])
    return X, time, event


def _load_metabric_pycox() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from pycox.datasets import metabric

    frame = metabric.read_df()
    event = frame["event"].to_numpy(dtype=np.int32)
    time = frame["duration"].to_numpy(dtype=np.float64)
    X = frame.drop(columns=["event", "duration"])
    return X, time, event


def _load_pbc_lifelines() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from lifelines.datasets import load_pbc

    frame = load_pbc()
    event = (frame["status"] == 2).astype(np.int32).to_numpy()
    time = frame["time"].astype(float).to_numpy()
    drop_cols = ["status", "time", "id"] if "id" in frame.columns else ["status", "time"]
    X = frame.drop(columns=drop_cols, errors="ignore")
    return X, time, event


def _load_kkbox_placeholder() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    raise NotImplementedError(
        "KKBox loader is a Large v1 placeholder. Add a custom loader that reads local data files."
    )


def load_dataset(dataset_id: str, repo_root: Path) -> SurvivalDataset:
    dataset_cfg = _load_dataset_config(repo_root / "configs", dataset_id)

    loaders: dict[str, Callable[[], tuple[pd.DataFrame, np.ndarray, np.ndarray]]] = {
        "support": _load_support_pycox,
        "metabric": _load_metabric_pycox,
        "gbsg2": lambda: _load_from_sksurv("gbsg2"),
        "flchain": lambda: _load_from_sksurv("flchain"),
        "whas500": lambda: _load_from_sksurv("whas500"),
        "pbc": _load_pbc_lifelines,
        "kkbox": _load_kkbox_placeholder,
    }
    if dataset_id not in loaders:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")

    X, time, event = loaders[dataset_id]()
    X = pd.DataFrame(X).reset_index(drop=True)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    feature_metadata = infer_feature_metadata(X)
    diagnostics = build_dataset_diagnostics(
        X,
        event=event,
        feature_metadata=feature_metadata,
    )
    metadata = DatasetMetadata(
        dataset_id=dataset_cfg["dataset_id"],
        name=dataset_cfg.get("name", dataset_cfg["dataset_id"]),
        source=dataset_cfg.get("source", "unknown"),
        task_type=dataset_cfg.get("task_type", "right_censored_survival"),
        event_col=dataset_cfg.get("event_col", "event"),
        time_col=dataset_cfg.get("time_col", "time"),
        group_col=dataset_cfg.get("group_col"),
        feature_types=dataset_cfg.get("feature_types", summarize_feature_types(feature_metadata)),
        feature_metadata=feature_metadata,
        diagnostics=diagnostics,
        split_strategy=dataset_cfg.get("split_strategy", "stratified_event"),
        primary_metric=dataset_cfg.get("primary_metric", "uno_c"),
        notes=dataset_cfg.get("notes", ""),
        raw={**dataset_cfg, "diagnostics": diagnostics.to_dict()},
    )

    dataset = SurvivalDataset(metadata=metadata, X=X, time=time, event=event)
    dataset.validate()
    return dataset
