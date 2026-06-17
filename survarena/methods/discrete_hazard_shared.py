from __future__ import annotations

from typing import Any, Callable

import numpy as np

from survarena.methods.base import SurvivalPredictions
from survarena.methods.discrete_time import (
    DiscreteHazardFrame,
    append_time_bin_features,
    baseline_hazards_from_km,
    build_discrete_hazard_frame,
    build_event_quantile_time_grid,
    clean_hazards,
    risk_from_hazards,
    survival_from_hazards,
)


DISCRETE_HAZARD_DEFAULTS: dict[str, Any] = {
    "time_grid": "event_quantile",
    "n_intervals": 5,
    "horizon_quantiles": None,
    "min_events_per_interval": 5,
    "min_rows_per_interval": 20,
    "max_stacked_rows": None,
    "subject_weighting": "normalized",
    "censoring_weighting": "none",
    "aggregate_risk": "cumulative_event_probability_at_last",
    "time_feature_set": "km",
}


def apply_discrete_hazard_defaults(params: dict[str, Any]) -> None:
    for key, value in DISCRETE_HAZARD_DEFAULTS.items():
        params.setdefault(key, value)


def init_discrete_hazard_state(instance: Any) -> None:
    instance.time_grid_ = None
    instance.time_train_ = None
    instance.event_train_ = None
    instance.baseline_hazards_ = None
    instance.used_fallback_ = False
    instance.frame_metadata_ = {}
    instance.grid_metadata_ = {}
    instance.sample_weight_supported_ = False
    instance.sample_weight_applied_ = False
    instance.last_hazard_min_ = None
    instance.last_hazard_max_ = None


def build_discrete_hazard_training_frame(
    instance: Any,
    *,
    X_train: Any,
    time_train: np.ndarray,
    event_train: np.ndarray,
) -> DiscreteHazardFrame:
    params = instance.params
    instance.time_train_ = np.asarray(time_train, dtype=np.float64)
    instance.event_train_ = np.asarray(event_train, dtype=np.int32)
    grid = build_event_quantile_time_grid(
        instance.time_train_,
        instance.event_train_,
        time_grid=str(params["time_grid"]),
        n_intervals=int(params["n_intervals"]),
        horizon_quantiles=params.get("horizon_quantiles"),
        min_events_per_interval=int(params["min_events_per_interval"]),
    )
    instance.time_grid_ = grid.endpoints
    instance.grid_metadata_ = dict(grid.metadata)
    instance.baseline_hazards_ = baseline_hazards_from_km(
        instance.time_train_,
        instance.event_train_,
        instance.time_grid_,
    )
    frame = build_discrete_hazard_frame(
        X=X_train,
        time=instance.time_train_,
        event=instance.event_train_,
        time_grid=instance.time_grid_,
        time_feature_spec=str(params["time_feature_set"]),
        subject_weighting=str(params["subject_weighting"]),
        censoring_weighting=str(params["censoring_weighting"]),
        max_stacked_rows=params.get("max_stacked_rows"),
        seed=params.get("seed"),
    )
    instance.frame_metadata_ = {**instance.grid_metadata_, **frame.metadata}
    return frame


def should_use_discrete_hazard_fallback(instance: Any, frame: DiscreteHazardFrame) -> bool:
    return int(len(frame.y_stacked)) < int(instance.params["min_rows_per_interval"]) or np.unique(frame.y_stacked).size < 2


def predict_discrete_hazards(
    instance: Any,
    *,
    X: Any,
    row_count: int,
    fitted_model: Any,
    probability_fn: Callable[[Any], np.ndarray],
) -> np.ndarray:
    if instance.time_grid_ is None or instance.baseline_hazards_ is None:
        raise RuntimeError(f"{instance.__class__.__name__} must be fit before prediction.")
    if fitted_model is None:
        hazards = np.tile(instance.baseline_hazards_, (row_count, 1))
    else:
        columns: list[np.ndarray] = []
        for idx in range(len(instance.time_grid_)):
            query = append_time_bin_features(
                X,
                np.full(row_count, idx, dtype=np.int32),
                instance.time_grid_,
                time_train=instance.time_train_,
                event_train=instance.event_train_,
                time_feature_set=str(instance.params["time_feature_set"]),
            )
            columns.append(probability_fn(query))
        hazards = np.column_stack(columns)
    clean = clean_hazards(hazards)
    instance.last_hazard_min_ = float(np.min(clean)) if clean.size else None
    instance.last_hazard_max_ = float(np.max(clean)) if clean.size else None
    return clean


def discrete_hazard_foundation_metadata(
    instance: Any,
    *,
    backbone: str,
    training: str,
) -> dict[str, Any]:
    return {
        "foundation_backbone": backbone,
        "foundation_backbone_task": "censored_aware_pooled_discrete_time_hazard_classification",
        "foundation_backbone_training": training,
        "foundation_time_grid": instance.grid_metadata_.get("time_grid", instance.params.get("time_grid")),
        "foundation_time_grid_endpoints": instance.grid_metadata_.get("time_grid_endpoints", []),
        "foundation_requested_interval_count": int(instance.params["n_intervals"]),
        "foundation_interval_count": 0 if instance.time_grid_ is None else int(len(instance.time_grid_)),
        "foundation_stacked_rows": int(instance.frame_metadata_.get("stacked_rows", 0)),
        "foundation_positive_rows": int(instance.frame_metadata_.get("positive_rows", 0)),
        "foundation_rows_per_interval": instance.frame_metadata_.get("rows_per_interval", []),
        "foundation_positive_rows_per_interval": instance.frame_metadata_.get("positive_rows_per_interval", []),
        "foundation_excluded_censored_in_interval_rows": int(
            instance.frame_metadata_.get("excluded_censored_in_interval_rows", 0)
        ),
        "foundation_sample_weight_supported": bool(instance.sample_weight_supported_),
        "foundation_sample_weight_requested": instance.params.get("subject_weighting"),
        "foundation_sample_weight_applied": bool(instance.sample_weight_applied_),
        "foundation_censoring_weighting": instance.params.get("censoring_weighting"),
        "foundation_ipcw_status": instance.frame_metadata_.get("ipcw_status", "not_implemented"),
        "foundation_time_features": instance.frame_metadata_.get("time_features", []),
        "foundation_discrete_hazard_fallback": bool(instance.used_fallback_),
        "foundation_max_stacked_rows_applied": bool(instance.frame_metadata_.get("max_stacked_rows_applied", False)),
        "foundation_predicted_hazard_min": instance.last_hazard_min_,
        "foundation_predicted_hazard_max": instance.last_hazard_max_,
    }


def discrete_hazard_predictions(instance: Any, X: Any, times: np.ndarray, hazards: np.ndarray) -> SurvivalPredictions:
    return SurvivalPredictions(
        risk=risk_from_hazards(hazards, aggregate_risk=str(instance.params["aggregate_risk"])),
        survival=survival_from_hazards(hazards, instance.time_grid_, times),
    )
