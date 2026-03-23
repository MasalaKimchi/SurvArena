from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np
import pandas as pd

from survarena.data.preprocess import TabularPreprocessor
from survarena.data.schema import SurvivalDataset


@dataclass(frozen=True, slots=True)
class ValidationPlan:
    source: str
    holdout_frac: float | None
    train_X: pd.DataFrame
    train_time: np.ndarray
    train_event: np.ndarray
    validation_X: pd.DataFrame
    validation_time: np.ndarray
    validation_event: np.ndarray


def default_holdout_frac(n_rows: int) -> float:
    if n_rows < 500:
        return 0.2
    if n_rows < 5_000:
        return 0.15
    if n_rows < 25_000:
        return 0.1
    return 0.05


def prepare_validation_fold_cache(*, method_id: str, plan: ValidationPlan) -> list[dict[str, Any]]:
    preprocessor = TabularPreprocessor(scale_numeric=(method_id != "rsf"))
    X_train = preprocessor.fit_transform(plan.train_X).to_numpy()
    X_validation = preprocessor.transform(plan.validation_X).to_numpy()
    return [
        {
            "X_train": X_train,
            "X_val": X_validation,
            "time_train": plan.train_time,
            "event_train": plan.train_event,
            "time_val": plan.validation_time,
            "event_val": plan.validation_event,
        }
    ]


def build_validation_plan(
    dataset: SurvivalDataset,
    *,
    tuning_dataset: SurvivalDataset | None = None,
    holdout_frac: float | None = None,
    seed: int,
) -> ValidationPlan:
    if tuning_dataset is not None:
        validation_X = _align_validation_frame(dataset.X, tuning_dataset.X)
        return ValidationPlan(
            source="tuning_data",
            holdout_frac=None,
            train_X=dataset.X.reset_index(drop=True),
            train_time=np.asarray(dataset.time, dtype=float),
            train_event=np.asarray(dataset.event, dtype=int),
            validation_X=validation_X,
            validation_time=np.asarray(tuning_dataset.time, dtype=float),
            validation_event=np.asarray(tuning_dataset.event, dtype=int),
        )

    resolved_holdout_frac = default_holdout_frac(len(dataset.X)) if holdout_frac is None else _validate_holdout_frac(holdout_frac)
    if len(dataset.X) < 4:
        raise ValueError("Need at least 4 rows to create an automatic validation holdout.")

    event = np.asarray(dataset.event, dtype=int)
    unique_events, counts = np.unique(event, return_counts=True)
    if unique_events.size < 2 or int(counts.min()) < 2:
        raise ValueError(
            "Automatic validation holdout requires at least two samples in each event class. "
            "Provide tuning_data for low-event datasets."
        )

    train_test_split = importlib.import_module("sklearn.model_selection").train_test_split
    indices = np.arange(len(dataset.X))
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=resolved_holdout_frac,
        stratify=event,
        random_state=seed,
    )
    return ValidationPlan(
        source="auto_holdout",
        holdout_frac=resolved_holdout_frac,
        train_X=dataset.X.iloc[train_idx].reset_index(drop=True),
        train_time=np.asarray(dataset.time[train_idx], dtype=float),
        train_event=np.asarray(dataset.event[train_idx], dtype=int),
        validation_X=dataset.X.iloc[validation_idx].reset_index(drop=True),
        validation_time=np.asarray(dataset.time[validation_idx], dtype=float),
        validation_event=np.asarray(dataset.event[validation_idx], dtype=int),
    )


def _align_validation_frame(train_X: pd.DataFrame, validation_X: pd.DataFrame) -> pd.DataFrame:
    train_columns = list(train_X.columns)
    validation_columns = list(validation_X.columns)
    if train_columns == validation_columns:
        return validation_X.reset_index(drop=True)

    train_column_set = set(train_columns)
    validation_column_set = set(validation_columns)
    if train_column_set != validation_column_set:
        missing = sorted(train_column_set - validation_column_set)
        extra = sorted(validation_column_set - train_column_set)
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {missing}")
        if extra:
            details.append(f"extra columns: {extra}")
        raise ValueError(
            "tuning_data features must match training features; "
            + "; ".join(details)
        )

    return validation_X.loc[:, train_columns].reset_index(drop=True)


def _validate_holdout_frac(holdout_frac: float) -> float:
    value = float(holdout_frac)
    if not 0.0 < value < 1.0:
        raise ValueError("holdout_frac must be between 0 and 1.")
    return value
