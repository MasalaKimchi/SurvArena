from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from survarena.methods.foundation.tabpfn_backbone import _kaplan_meier_survival_at

HAZARD_EPSILON = 1e-7
DEFAULT_TIME_FEATURES = ("interval_index", "log_interval_end", "interval_width", "km_survival")


@dataclass(frozen=True, slots=True)
class DiscreteTimeGrid:
    endpoints: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DiscreteHazardFrame:
    X_stacked: Any
    y_stacked: np.ndarray
    row_weights: np.ndarray | None
    subject_ids: np.ndarray
    interval_ids: np.ndarray
    metadata: dict[str, Any]


def parse_time_bin_quantiles(value: str | list[float]) -> np.ndarray:
    if isinstance(value, str):
        quantiles = [float(part.strip()) for part in value.split("-") if part.strip()]
    else:
        quantiles = [float(part) for part in value]
    if not quantiles:
        raise ValueError("time_bin_quantiles must contain at least one quantile.")
    array = np.asarray(quantiles, dtype=np.float64)
    if np.any((array <= 0.0) | (array > 1.0)):
        raise ValueError("time_bin_quantiles must be in the interval (0, 1].")
    return np.unique(array)


def _default_interval_quantiles(n_intervals: int) -> np.ndarray:
    requested = int(n_intervals)
    if requested not in {3, 5, 10}:
        raise ValueError("n_intervals must be one of {3, 5, 10}.")
    return np.linspace(1.0 / requested, 1.0, requested, dtype=np.float64)


def build_event_quantile_time_grid(
    time: np.ndarray,
    event: np.ndarray,
    *,
    time_grid: str = "event_quantile",
    n_intervals: int = 5,
    horizon_quantiles: str | list[float] | None = None,
    min_events_per_interval: int = 5,
) -> DiscreteTimeGrid:
    if str(time_grid) != "event_quantile":
        raise ValueError("Only time_grid='event_quantile' is currently supported.")
    time_np = np.asarray(time, dtype=np.float64).reshape(-1)
    event_np = np.asarray(event).astype(bool).reshape(-1)
    if time_np.size != event_np.size:
        raise ValueError("time and event must have the same length.")
    event_times = time_np[event_np & np.isfinite(time_np)]
    if event_times.size <= 0:
        raise ValueError("Discrete-time survival training requires at least one observed event.")

    quantiles = (
        parse_time_bin_quantiles(horizon_quantiles)
        if horizon_quantiles is not None
        else _default_interval_quantiles(int(n_intervals))
    )
    requested_endpoint_count = int(quantiles.size)
    raw_endpoints = np.quantile(event_times, quantiles).astype(np.float64)
    endpoints = np.unique(raw_endpoints[np.isfinite(raw_endpoints)])
    endpoints = endpoints[endpoints > 0.0]
    if endpoints.size == 0:
        endpoints = np.asarray([float(np.max(event_times))], dtype=np.float64)

    previous = np.concatenate(([0.0], endpoints[:-1]))
    event_counts = np.asarray(
        [int(np.sum((event_times > left) & (event_times <= right))) for left, right in zip(previous, endpoints, strict=False)],
        dtype=np.int32,
    )
    collapsed = int(requested_endpoint_count - endpoints.size)
    metadata = {
        "time_grid": "event_quantile",
        "requested_n_intervals": int(requested_endpoint_count),
        "actual_n_intervals": int(endpoints.size),
        "requested_quantiles": quantiles.astype(float).tolist(),
        "time_grid_endpoints": endpoints.astype(float).tolist(),
        "time_grid_duplicate_endpoint_count": collapsed,
        "min_events_per_interval": int(min_events_per_interval),
        "event_rows_per_interval": event_counts.astype(int).tolist(),
        "low_event_interval_count": int(np.sum(event_counts < int(min_events_per_interval))),
    }
    return DiscreteTimeGrid(endpoints=endpoints, metadata=metadata)


def event_quantile_time_bins(time: np.ndarray, event: np.ndarray, quantiles: str | list[float]) -> np.ndarray:
    return build_event_quantile_time_grid(time, event, horizon_quantiles=quantiles).endpoints


def interval_label_matrix(
    *,
    time: np.ndarray,
    event: np.ndarray,
    time_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    time_np = np.asarray(time, dtype=np.float64).reshape(-1)
    event_np = np.asarray(event).astype(bool).reshape(-1)
    bins = np.asarray(time_bins, dtype=np.float64).reshape(-1)
    previous = np.concatenate(([0.0], bins[:-1]))
    labels = np.zeros((time_np.size, bins.size), dtype=np.float32)
    known = np.zeros((time_np.size, bins.size), dtype=bool)
    for idx, (left, right) in enumerate(zip(previous, bins, strict=False)):
        positive = (time_np > left) & (time_np <= right) & event_np
        negative = time_np > right
        known[:, idx] = positive | negative
        labels[:, idx] = positive.astype(np.float32)
    return known, labels


def baseline_hazards_from_km(time: np.ndarray, event: np.ndarray, time_bins: np.ndarray) -> np.ndarray:
    bins = np.asarray(time_bins, dtype=np.float64).reshape(-1)
    survival = _kaplan_meier_survival_at(np.asarray(time, dtype=np.float64), np.asarray(event, dtype=np.int32), bins)
    previous_survival = np.concatenate(([1.0], survival[:-1]))
    hazards = 1.0 - (survival / np.clip(previous_survival, HAZARD_EPSILON, 1.0))
    return clean_hazards(hazards)


def clean_hazards(hazards: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(hazards, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(values, HAZARD_EPSILON, 1.0 - HAZARD_EPSILON)


def survival_from_hazards(hazards: np.ndarray, time_bins: np.ndarray, times: np.ndarray) -> np.ndarray:
    hazard_np = clean_hazards(hazards)
    if hazard_np.ndim == 1:
        hazard_np = hazard_np[None, :]
    bins = np.asarray(time_bins, dtype=np.float64).reshape(-1)
    eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
    survival_at_bins = np.cumprod(1.0 - hazard_np, axis=1)
    interval_positions = np.searchsorted(bins, eval_times, side="right") - 1
    survival = np.ones((hazard_np.shape[0], eval_times.size), dtype=np.float64)
    inside = interval_positions >= 0
    if np.any(inside):
        clipped_positions = np.clip(interval_positions[inside], 0, bins.size - 1)
        survival[:, inside] = survival_at_bins[:, clipped_positions]
    survival = np.nan_to_num(survival, nan=1.0, posinf=1.0, neginf=HAZARD_EPSILON)
    return np.minimum.accumulate(np.clip(survival, HAZARD_EPSILON, 1.0), axis=1).astype(np.float64)


def risk_from_hazards(hazards: np.ndarray, aggregate_risk: str = "cumulative_event_probability_at_last") -> np.ndarray:
    hazard_np = clean_hazards(hazards)
    if hazard_np.ndim == 1:
        hazard_np = hazard_np[None, :]
    event_probabilities = 1.0 - np.cumprod(1.0 - hazard_np, axis=1)
    if aggregate_risk in {"last_event_probability", "cumulative_event_probability_at_last"}:
        return event_probabilities[:, -1].astype(np.float64)
    if aggregate_risk in {"mean_event_probability", "mean_cumulative_event_probability"}:
        return event_probabilities.mean(axis=1).astype(np.float64)
    raise ValueError(
        "aggregate_risk must be one of "
        "{'cumulative_event_probability_at_last', 'mean_cumulative_event_probability'}."
    )


def _resolve_time_features(time_feature_set: str | None, time_features: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if time_features is not None:
        return tuple(str(feature) for feature in time_features)
    if str(time_feature_set or "km") == "minimal":
        return ("interval_index", "log_interval_end", "interval_width")
    return DEFAULT_TIME_FEATURES


def _interval_feature_matrix(
    interval_ids: np.ndarray,
    time_bins: np.ndarray,
    *,
    time_train: np.ndarray | None,
    event_train: np.ndarray | None,
    time_feature_set: str | None,
    time_features: list[str] | tuple[str, ...] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    interval_idx = np.asarray(interval_ids, dtype=np.int64).reshape(-1)
    bins = np.asarray(time_bins, dtype=np.float64).reshape(-1)
    previous = np.concatenate(([0.0], bins[:-1]))
    feature_names = _resolve_time_features(time_feature_set, time_features)
    columns: list[np.ndarray] = []
    for name in feature_names:
        if name == "interval_index":
            values = interval_idx.astype(np.float64)
        elif name == "interval_index_normalized":
            values = interval_idx.astype(np.float64) / max(float(len(bins) - 1), 1.0)
        elif name == "log_interval_end":
            values = np.log1p(bins[interval_idx])
        elif name == "interval_width":
            values = bins[interval_idx] - previous[interval_idx]
        elif name in {"km_survival", "km_survival_at_interval_end"}:
            if time_train is None or event_train is None:
                values = np.ones(interval_idx.size, dtype=np.float64)
            else:
                km = _kaplan_meier_survival_at(
                    np.asarray(time_train, dtype=np.float64),
                    np.asarray(event_train, dtype=np.int32),
                    bins,
                )
                values = km[interval_idx]
        elif name == "km_survival_at_interval_start":
            if time_train is None or event_train is None:
                values = np.ones(interval_idx.size, dtype=np.float64)
            else:
                start_values = _kaplan_meier_survival_at(
                    np.asarray(time_train, dtype=np.float64),
                    np.asarray(event_train, dtype=np.int32),
                    previous,
                )
                values = start_values[interval_idx]
        else:
            raise ValueError(f"Unknown discrete-time feature '{name}'.")
        columns.append(values.astype(np.float32))
    if not columns:
        return np.empty((interval_idx.size, 0), dtype=np.float32), feature_names
    return np.column_stack(columns).astype(np.float32), feature_names


def append_time_bin_features(
    X: Any,
    bin_indices: np.ndarray,
    time_bins: np.ndarray,
    *,
    time_train: np.ndarray | None = None,
    event_train: np.ndarray | None = None,
    time_feature_set: str | None = "km",
    time_features: list[str] | tuple[str, ...] | None = None,
) -> Any:
    bin_idx = np.asarray(bin_indices, dtype=np.int64).reshape(-1)
    features, feature_names = _interval_feature_matrix(
        bin_idx,
        time_bins,
        time_train=time_train,
        event_train=event_train,
        time_feature_set=time_feature_set,
        time_features=time_features,
    )
    if isinstance(X, pd.DataFrame):
        frame = X.reset_index(drop=True).copy()
        for offset, name in enumerate(feature_names):
            frame[f"__survarena_{name}__"] = features[:, offset].astype(np.float32)
        return frame
    X_np = np.asarray(X, dtype=np.float32)
    return np.column_stack([X_np, features]).astype(np.float32)


def build_discrete_hazard_frame(
    *,
    X: Any,
    time: np.ndarray,
    event: np.ndarray,
    time_grid: np.ndarray,
    time_feature_spec: str | list[str] | tuple[str, ...] = "km",
    subject_weighting: str = "normalized",
    censoring_weighting: str | None = None,
    max_stacked_rows: int | None = None,
    seed: int | None = None,
) -> DiscreteHazardFrame:
    if censoring_weighting not in {None, "none"}:
        raise ValueError("Only censoring_weighting='none' is currently supported; IPCW is reserved for future work.")
    time_np = np.asarray(time, dtype=np.float64).reshape(-1)
    event_np = np.asarray(event, dtype=np.int32).reshape(-1)
    bins = np.asarray(time_grid, dtype=np.float64).reshape(-1)
    known, labels = interval_label_matrix(time=time_np, event=event_np, time_bins=bins)
    subject_ids, interval_ids = np.nonzero(known)
    if isinstance(X, pd.DataFrame):
        expanded_X = X.iloc[subject_ids].reset_index(drop=True)
    else:
        expanded_X = np.asarray(X, dtype=np.float32)[subject_ids]

    if isinstance(time_feature_spec, str):
        time_feature_set = time_feature_spec
        time_features = None
    else:
        time_feature_set = None
        time_features = tuple(str(feature) for feature in time_feature_spec)

    expanded_X = append_time_bin_features(
        expanded_X,
        interval_ids,
        bins,
        time_train=time_np,
        event_train=event_np,
        time_feature_set=time_feature_set,
        time_features=time_features,
    )
    y = labels[subject_ids, interval_ids].astype(np.int32)

    row_weights: np.ndarray | None
    requested_subject_weighting = str(subject_weighting)
    if requested_subject_weighting == "normalized":
        counts = np.bincount(subject_ids, minlength=time_np.size).astype(np.float64)
        row_weights = (1.0 / np.clip(counts[subject_ids], 1.0, None)).astype(np.float64)
    elif requested_subject_weighting == "none":
        row_weights = None
    else:
        raise ValueError("subject_weighting must be one of {'normalized', 'none'}.")

    previous = np.concatenate(([0.0], bins[:-1]))
    excluded_censored_by_interval = [
        int(np.sum((event_np == 0) & (time_np > left) & (time_np <= right)))
        for left, right in zip(previous, bins, strict=False)
    ]
    original_rows = int(y.size)
    sampled = False
    if max_stacked_rows is not None and int(max_stacked_rows) > 0 and y.size > int(max_stacked_rows):
        selected = _sample_indices(y, max_rows=int(max_stacked_rows), seed=seed)
        if hasattr(expanded_X, "iloc"):
            expanded_X = expanded_X.iloc[selected].reset_index(drop=True)
        else:
            expanded_X = np.asarray(expanded_X)[selected]
        y = y[selected]
        subject_ids = subject_ids[selected]
        interval_ids = interval_ids[selected]
        row_weights = None if row_weights is None else row_weights[selected]
        sampled = True

    rows_per_interval = np.bincount(interval_ids, minlength=bins.size).astype(np.int32)
    positives_per_interval = np.bincount(interval_ids, weights=y, minlength=bins.size).astype(np.int32)

    metadata = {
        "stacked_rows": int(y.size),
        "original_stacked_rows": original_rows,
        "max_stacked_rows": None if max_stacked_rows is None else int(max_stacked_rows),
        "max_stacked_rows_applied": bool(sampled),
        "positive_rows": int(np.sum(y)),
        "negative_rows": int(y.size - np.sum(y)),
        "rows_per_interval": rows_per_interval.astype(int).tolist(),
        "positive_rows_per_interval": positives_per_interval.astype(int).tolist(),
        "excluded_censored_in_interval_rows": int(np.sum(excluded_censored_by_interval)),
        "excluded_censored_in_interval_rows_per_interval": excluded_censored_by_interval,
        "requested_subject_weighting": requested_subject_weighting,
        "censoring_weighting": "none",
        "ipcw_status": "not_implemented",
        "time_features": list(_resolve_time_features(time_feature_set, time_features)),
    }
    return DiscreteHazardFrame(
        X_stacked=expanded_X,
        y_stacked=y,
        row_weights=row_weights,
        subject_ids=subject_ids.astype(np.int32),
        interval_ids=interval_ids.astype(np.int32),
        metadata=metadata,
    )


def build_person_time_hazard_frame(
    *,
    X: Any,
    time: np.ndarray,
    event: np.ndarray,
    time_bins: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray]:
    frame = build_discrete_hazard_frame(
        X=X,
        time=time,
        event=event,
        time_grid=time_bins,
        time_feature_spec=("interval_index", "interval_index_normalized", "log_interval_end", "interval_width"),
        subject_weighting="none",
    )
    return frame.X_stacked, frame.y_stacked, frame.interval_ids


def _sample_indices(y: np.ndarray, *, max_rows: int, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y_np = np.asarray(y, dtype=np.int32)
    positives = np.flatnonzero(y_np == 1)
    negatives = np.flatnonzero(y_np == 0)
    positive_budget = min(len(positives), max(1, int(max_rows) // 2))
    negative_budget = int(max_rows) - positive_budget
    selected = []
    if positive_budget > 0 and len(positives) > 0:
        selected.append(rng.choice(positives, size=positive_budget, replace=False))
    if negative_budget > 0 and len(negatives) > 0:
        selected.append(rng.choice(negatives, size=min(negative_budget, len(negatives)), replace=False))
    if not selected:
        return np.arange(min(len(y_np), int(max_rows)))
    return np.sort(np.concatenate(selected))


def sample_person_time_rows(
    X: Any,
    y: np.ndarray,
    bin_indices: np.ndarray,
    *,
    max_rows: int | None,
    seed: int | None,
) -> tuple[Any, np.ndarray, np.ndarray]:
    if max_rows is None or int(max_rows) <= 0 or len(y) <= int(max_rows):
        return X, y, bin_indices
    indices = _sample_indices(np.asarray(y, dtype=np.int32), max_rows=int(max_rows), seed=seed)
    if hasattr(X, "iloc"):
        X_sampled = X.iloc[indices].reset_index(drop=True)
    else:
        X_sampled = np.asarray(X)[indices]
    return X_sampled, np.asarray(y, dtype=np.int32)[indices], np.asarray(bin_indices, dtype=np.int32)[indices]
