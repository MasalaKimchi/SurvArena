from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from survarena.data.profiling import build_dataset_diagnostics, infer_feature_metadata, summarize_feature_types
from survarena.data.schema import DatasetMetadata, SurvivalDataset
from survarena.data.user_dataset import load_user_dataset


def _load_dataset_config(configs_dir: Path, dataset_id: str) -> dict:
    config_path = configs_dir / "datasets" / f"{dataset_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_from_sksurv(dataset_id: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from sksurv import datasets as sk_datasets

    mapping: dict[str, Callable[[], tuple[pd.DataFrame, np.ndarray]]] = {
        "aids": sk_datasets.load_aids,
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
    # scikit-survival's load_* helpers already return structured targets where
    # True means "event occurred", even when the raw field name is confusing
    # (for example, AIDS uses `censor` and GBSG2 uses `cens`).
    event = y[event_field].astype(np.int32)
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


def _load_local_file(dataset_cfg: dict, repo_root: Path) -> SurvivalDataset:
    local_path = dataset_cfg.get("local_path")
    if not local_path:
        raise ValueError(f"Dataset '{dataset_cfg['dataset_id']}' is configured as local_file but has no local_path.")

    path = repo_root / local_path
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared dataset file not found for '{dataset_cfg['dataset_id']}': {path}. "
            "Run the documented preparation command for this local dataset first."
        )

    dataset = load_user_dataset(
        path,
        time_col=dataset_cfg.get("time_col", "time"),
        event_col=dataset_cfg.get("event_col", "event"),
        dataset_id=dataset_cfg["dataset_id"],
        dataset_name=dataset_cfg.get("name"),
        id_col=dataset_cfg.get("id_col"),
        drop_columns=dataset_cfg.get("drop_columns", []),
    )
    dataset.metadata.source = dataset_cfg.get("source", "local_file")
    dataset.metadata.task_type = dataset_cfg.get("task_type", dataset.metadata.task_type)
    dataset.metadata.group_col = dataset_cfg.get("group_col")
    dataset.metadata.split_strategy = dataset_cfg.get("split_strategy", dataset.metadata.split_strategy)
    dataset.metadata.primary_metric = dataset_cfg.get("primary_metric", dataset.metadata.primary_metric)
    dataset.metadata.notes = dataset_cfg.get("notes", dataset.metadata.notes)
    dataset.metadata.raw = {**dataset_cfg, "diagnostics": dataset.metadata.diagnostics.to_dict()}
    return dataset


def _load_nwtco_pycox() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    from pycox.datasets import nwtco

    frame = nwtco.read_df(processed=False).copy()
    event = frame["rel"].to_numpy(dtype=np.int32)
    time = frame["edrel"].to_numpy(dtype=np.float64)
    X = (
        frame.assign(
            instit_2=frame["instit"] - 1,
            histol_2=frame["histol"] - 1,
            study_4=frame["study"] - 3,
            stage=frame["stage"].astype("category"),
        )
        .drop(columns=["rownames", "Unnamed: 0", "seqno", "instit", "histol", "study", "edrel", "rel"], errors="ignore")
        .reset_index(drop=True)
    )
    for col in X.columns.drop("stage"):
        X[col] = X[col].astype("float32")
    return X, time, event


def load_dataset(dataset_id: str, repo_root: Path) -> SurvivalDataset:
    dataset_cfg = _load_dataset_config(repo_root / "configs", dataset_id)
    if dataset_cfg.get("source") == "local_file" or dataset_cfg.get("local_path"):
        return _load_local_file(dataset_cfg, repo_root)

    loaders: dict[str, Callable[[], tuple[pd.DataFrame, np.ndarray, np.ndarray]]] = {
        "support": _load_support_pycox,
        "metabric": _load_metabric_pycox,
        "nwtco": _load_nwtco_pycox,
        "aids": lambda: _load_from_sksurv("aids"),
        "gbsg2": lambda: _load_from_sksurv("gbsg2"),
        "flchain": lambda: _load_from_sksurv("flchain"),
        "whas500": lambda: _load_from_sksurv("whas500"),
    }
    if dataset_id not in loaders:
        raise ValueError(f"Unknown dataset_id: {dataset_id}")

    X, time, event = loaders[dataset_id]()
    X = pd.DataFrame(X).reset_index(drop=True)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if np.any(time <= 0):
        positive = time[time > 0]
        reference = float(np.nanmin(positive)) if positive.size else 1.0
        eps = max(1e-8, reference * 1e-6)
        time = np.where(time <= 0, eps, time)
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
