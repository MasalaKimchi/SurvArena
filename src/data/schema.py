from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


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
