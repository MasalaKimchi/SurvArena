from __future__ import annotations

from dataclasses import dataclass, replace
import importlib
from typing import Any

import numpy as np
import pandas as pd

from survarena.data.preprocess import TabularPreprocessor
from survarena.data.schema import SurvivalDataset
from survarena.methods.preprocessing import finalize_preprocessed_features, method_preprocessor_kwargs


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


@dataclass(frozen=True, slots=True)
class ResampledFold:
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


def prepare_resampled_fold_cache(*, method_id: str, folds: list[ResampledFold]) -> list[dict[str, Any]]:
    fold_cache: list[dict[str, Any]] = []
    for fold in folds:
        preprocessor = TabularPreprocessor(**method_preprocessor_kwargs(method_id))
        X_train = finalize_preprocessed_features(method_id, preprocessor.fit_transform(fold.train_X))
        X_validation = finalize_preprocessed_features(method_id, preprocessor.transform(fold.validation_X))
        fold_cache.append(
            {
                "X_train": X_train,
                "X_val": X_validation,
                "time_train": fold.train_time,
                "event_train": fold.train_event,
                "time_val": fold.validation_time,
                "event_val": fold.validation_event,
            }
        )
    return fold_cache


def build_bagging_folds(
    dataset: SurvivalDataset,
    *,
    num_bag_folds: int,
    num_bag_sets: int,
    seed: int,
) -> list[ResampledFold]:
    resolved_num_bag_folds = _validate_num_bag_folds(num_bag_folds)
    resolved_num_bag_sets = _validate_num_bag_sets(num_bag_sets)

    event = np.asarray(dataset.event, dtype=int)
    unique_events, counts = np.unique(event, return_counts=True)
    if unique_events.size < 2 or int(counts.min()) < resolved_num_bag_folds:
        raise ValueError(
            "Bagged OOF validation requires at least num_bag_folds samples in each event class. "
            "Lower num_bag_folds or provide a larger training set."
        )

    StratifiedKFold = importlib.import_module("sklearn.model_selection").StratifiedKFold
    indices = np.arange(len(dataset.X))
    folds: list[ResampledFold] = []
    for bag_set in range(resolved_num_bag_sets):
        splitter = StratifiedKFold(
            n_splits=resolved_num_bag_folds,
            shuffle=True,
            random_state=int(seed) + bag_set,
        )
        for train_idx, validation_idx in splitter.split(indices, event):
            folds.append(
                ResampledFold(
                    train_X=dataset.X.iloc[train_idx].reset_index(drop=True),
                    train_time=np.asarray(dataset.time[train_idx], dtype=float),
                    train_event=np.asarray(dataset.event[train_idx], dtype=int),
                    validation_X=dataset.X.iloc[validation_idx].reset_index(drop=True),
                    validation_time=np.asarray(dataset.time[validation_idx], dtype=float),
                    validation_event=np.asarray(dataset.event[validation_idx], dtype=int),
                )
            )
    return folds


def bagging_row_summary(folds: list[ResampledFold]) -> tuple[int | None, int]:
    if not folds:
        return None, 0
    average_train_rows = int(round(np.mean([len(fold.train_X) for fold in folds])))
    total_validation_rows = int(sum(len(fold.validation_X) for fold in folds))
    return average_train_rows, total_validation_rows


def validation_plan_to_fold(plan: ValidationPlan) -> ResampledFold:
    return ResampledFold(
        train_X=plan.train_X,
        train_time=plan.train_time,
        train_event=plan.train_event,
        validation_X=plan.validation_X,
        validation_time=plan.validation_time,
        validation_event=plan.validation_event,
    )


def prepare_validation_fold_cache(*, method_id: str, plan: ValidationPlan) -> list[dict[str, Any]]:
    return prepare_resampled_fold_cache(method_id=method_id, folds=[validation_plan_to_fold(plan)])


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


def build_refit_dataset(
    dataset: SurvivalDataset,
    *,
    validation_plan: ValidationPlan | None,
    tuning_dataset: SurvivalDataset | None = None,
    refit_full: bool,
) -> SurvivalDataset:
    if tuning_dataset is not None:
        if refit_full:
            aligned_tuning_X = _align_validation_frame(dataset.X, tuning_dataset.X)
            X = pd.concat([dataset.X.reset_index(drop=True), aligned_tuning_X], ignore_index=True)
            time = np.concatenate(
                [
                    np.asarray(dataset.time, dtype=float),
                    np.asarray(tuning_dataset.time, dtype=float),
                ]
            )
            event = np.concatenate(
                [
                    np.asarray(dataset.event, dtype=int),
                    np.asarray(tuning_dataset.event, dtype=int),
                ]
            )
        else:
            X = dataset.X.reset_index(drop=True)
            time = np.asarray(dataset.time, dtype=float)
            event = np.asarray(dataset.event, dtype=int)
    elif refit_full:
        X = dataset.X.reset_index(drop=True)
        time = np.asarray(dataset.time, dtype=float)
        event = np.asarray(dataset.event, dtype=int)
    elif validation_plan is not None:
        X = validation_plan.train_X.reset_index(drop=True)
        time = np.asarray(validation_plan.train_time, dtype=float)
        event = np.asarray(validation_plan.train_event, dtype=int)
    else:
        X = dataset.X.reset_index(drop=True)
        time = np.asarray(dataset.time, dtype=float)
        event = np.asarray(dataset.event, dtype=int)

    refit_dataset = SurvivalDataset(
        metadata=replace(
            dataset.metadata,
            dataset_id=f"{dataset.metadata.dataset_id}_refit",
            name=f"{dataset.metadata.name}_refit",
        ),
        X=X,
        time=time,
        event=event,
    )
    refit_dataset.validate()
    return refit_dataset


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


def _validate_num_bag_folds(num_bag_folds: int) -> int:
    value = int(num_bag_folds)
    if value == 0:
        raise ValueError("num_bag_folds must be at least 2 when bagging is enabled.")
    if value < 2:
        raise ValueError("num_bag_folds must be 0 or >= 2.")
    return value


def _validate_num_bag_sets(num_bag_sets: int) -> int:
    value = int(num_bag_sets)
    if value < 1:
        raise ValueError("num_bag_sets must be at least 1.")
    return value
