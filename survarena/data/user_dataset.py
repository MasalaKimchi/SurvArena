from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from survarena.data.io import read_tabular_data
from survarena.data.profiling import build_dataset_diagnostics, infer_feature_metadata, summarize_feature_types
from survarena.data.schema import DatasetMetadata, SurvivalDataset


def _coerce_event_indicator(series: pd.Series, event_col: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int).to_numpy(dtype=np.int32)

    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any():
            raise ValueError(f"Column '{event_col}' must not contain missing event indicators.")
        if ((values != 0) & (values != 1)).any():
            raise ValueError(f"Column '{event_col}' must contain only binary event indicators.")
        return values.astype(int).to_numpy(dtype=np.int32)

    normalized = series.astype(str).str.strip().str.lower()
    truth_map = {
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "event": 1,
        "censored": 0,
        "dead": 1,
        "alive": 0,
    }
    mapped = normalized.map(truth_map)
    if mapped.isna().any():
        raise ValueError(f"Column '{event_col}' must be binary or coercible to binary values.")
    return mapped.astype(int).to_numpy(dtype=np.int32)


def load_user_dataset(
    data: pd.DataFrame | str | Path,
    *,
    time_col: str,
    event_col: str,
    dataset_id: str = "user_dataset",
    dataset_name: str | None = None,
    id_col: str | None = None,
    drop_columns: list[str] | None = None,
) -> SurvivalDataset:
    frame = read_tabular_data(data).reset_index(drop=True)
    missing = [col for col in (time_col, event_col) if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required survival label columns: {missing}")

    time = pd.to_numeric(frame[time_col], errors="coerce")
    if time.isna().any():
        raise ValueError(f"Column '{time_col}' must be numeric.")
    event = _coerce_event_indicator(frame[event_col], event_col)

    feature_drop = {time_col, event_col}
    if id_col is not None:
        feature_drop.add(id_col)
    if drop_columns:
        feature_drop.update(drop_columns)
    X = frame.drop(columns=[col for col in feature_drop if col in frame.columns], errors="ignore")
    if X.empty:
        raise ValueError("No feature columns remain after removing label columns.")

    feature_metadata = infer_feature_metadata(X)
    diagnostics = build_dataset_diagnostics(
        X,
        event=np.asarray(event, dtype=np.int32),
        feature_metadata=feature_metadata,
    )
    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        name=dataset_name or dataset_id,
        source="user_provided",
        task_type="right_censored_survival",
        event_col=event_col,
        time_col=time_col,
        feature_types=summarize_feature_types(feature_metadata),
        feature_metadata=feature_metadata,
        diagnostics=diagnostics,
        split_strategy="fixed_split",
        primary_metric="harrell_c",
        notes="User-provided dataset loaded through SurvivalPredictor.",
        raw={
            "dataset_id": dataset_id,
            "dataset_name": dataset_name or dataset_id,
            "id_col": id_col,
            "drop_columns": list(drop_columns or []),
            "diagnostics": diagnostics.to_dict(),
        },
    )
    dataset = SurvivalDataset(
        metadata=metadata,
        X=pd.DataFrame(X).reset_index(drop=True),
        time=time.to_numpy(dtype=np.float64),
        event=np.asarray(event, dtype=np.int32),
    )
    dataset.validate()
    return dataset
