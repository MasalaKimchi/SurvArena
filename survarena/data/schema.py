from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class FeatureMetadata:
    name: str
    inferred_type: str
    dtype: str
    n_unique: int
    missing_fraction: float
    is_constant: bool = False
    is_id_like: bool = False
    cardinality: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "inferred_type": self.inferred_type,
            "dtype": self.dtype,
            "n_unique": self.n_unique,
            "missing_fraction": self.missing_fraction,
            "is_constant": self.is_constant,
            "is_id_like": self.is_id_like,
            "cardinality": self.cardinality,
        }


@dataclass(slots=True)
class DatasetDiagnostics:
    n_rows: int
    n_features: int
    n_events: int
    event_rate: float
    censoring_rate: float
    missing_fraction: float
    feature_type_counts: dict[str, int] = field(default_factory=dict)
    high_cardinality_features: list[str] = field(default_factory=list)
    id_like_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "n_events": self.n_events,
            "event_rate": self.event_rate,
            "censoring_rate": self.censoring_rate,
            "missing_fraction": self.missing_fraction,
            "feature_type_counts": dict(self.feature_type_counts),
            "high_cardinality_features": list(self.high_cardinality_features),
            "id_like_features": list(self.id_like_features),
            "warnings": list(self.warnings),
        }


@dataclass(slots=True)
class DatasetMetadata:
    dataset_id: str
    name: str
    source: str
    task_type: str = "right_censored_survival"
    event_col: str = "event"
    time_col: str = "time"
    group_col: str | None = None
    feature_types: list[str] = field(default_factory=list)
    feature_metadata: list[FeatureMetadata] = field(default_factory=list)
    diagnostics: DatasetDiagnostics | None = None
    split_strategy: str = "stratified_event"
    primary_metric: str = "uno_c"
    notes: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SurvivalDataset:
    metadata: DatasetMetadata
    X: pd.DataFrame
    time: np.ndarray
    event: np.ndarray

    def validate(self) -> None:
        if len(self.X) != len(self.time) or len(self.X) != len(self.event):
            raise ValueError("X, time, and event must have the same row count.")
        if (self.time < 0).any():
            raise ValueError("Survival times must be nonnegative.")
        unique_events = set(np.unique(self.event).tolist())
        if not unique_events.issubset({0, 1, False, True}):
            raise ValueError("Event indicator must be binary.")
