from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from survarena.methods.base import BaseSurvivalMethod, SurvivalPredictions
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
from survarena.methods.foundation.inference import positive_class_probability_with_backoff
from survarena.methods.foundation.readiness import ensure_foundation_runtime_ready, rewrite_foundation_runtime_error
from survarena.methods.foundation.tabpfn_backbone import build_tabpfn_classifier


def _fit_supports_sample_weight(model: Any) -> bool:
    try:
        signature = inspect.signature(model.fit)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "sample_weight" or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _fit_classifier(model: Any, X: Any, y: np.ndarray, row_weights: np.ndarray | None) -> tuple[bool, bool]:
    supports_sample_weight = _fit_supports_sample_weight(model)
    if row_weights is not None and supports_sample_weight:
        model.fit(X, y, sample_weight=row_weights)
        return supports_sample_weight, True
    model.fit(X, y)
    return supports_sample_weight, False


class _PooledDiscreteTimeHazardSurvivalMethod(BaseSurvivalMethod):
    method_id = ""
    foundation_backbone = "DirectFoundationClassifier"
    foundation_training = "frozen"

    def __init__(
        self,
        time_grid: str = "event_quantile",
        n_intervals: int = 5,
        horizon_quantiles: str | list[float] | None = None,
        min_events_per_interval: int = 5,
        min_rows_per_interval: int = 20,
        max_stacked_rows: int | None = None,
        subject_weighting: str = "normalized",
        censoring_weighting: str = "none",
        aggregate_risk: str = "cumulative_event_probability_at_last",
        time_feature_set: str = "km",
        seed: int | None = None,
        predict_batch_size: int | None = None,
        **params: Any,
    ) -> None:
        super().__init__(
            time_grid=time_grid,
            n_intervals=n_intervals,
            horizon_quantiles=horizon_quantiles,
            min_events_per_interval=min_events_per_interval,
            min_rows_per_interval=min_rows_per_interval,
            max_stacked_rows=max_stacked_rows,
            subject_weighting=subject_weighting,
            censoring_weighting=censoring_weighting,
            aggregate_risk=aggregate_risk,
            time_feature_set=time_feature_set,
            seed=seed,
            predict_batch_size=predict_batch_size,
            **params,
        )
        self.model_: Any | None = None
        self.time_grid_: np.ndarray | None = None
        self.time_train_: np.ndarray | None = None
        self.event_train_: np.ndarray | None = None
        self.baseline_hazards_: np.ndarray | None = None
        self.used_fallback_: bool = False
        self.frame_metadata_: dict[str, Any] = {}
        self.grid_metadata_: dict[str, Any] = {}
        self.sample_weight_supported_: bool = False
        self.sample_weight_applied_: bool = False
        self.last_hazard_min_: float | None = None
        self.last_hazard_max_: float | None = None

    def _build_backbone(self) -> Any:
        raise NotImplementedError

    def foundation_metadata(self) -> dict[str, Any]:
        return {
            "foundation_backbone": self.foundation_backbone,
            "foundation_backbone_task": "censored_aware_pooled_discrete_time_hazard_classification",
            "foundation_backbone_training": self.foundation_training,
            "foundation_time_grid": self.grid_metadata_.get("time_grid", self.params.get("time_grid")),
            "foundation_time_grid_endpoints": self.grid_metadata_.get("time_grid_endpoints", []),
            "foundation_requested_interval_count": int(self.params["n_intervals"]),
            "foundation_interval_count": 0 if self.time_grid_ is None else int(len(self.time_grid_)),
            "foundation_stacked_rows": int(self.frame_metadata_.get("stacked_rows", 0)),
            "foundation_positive_rows": int(self.frame_metadata_.get("positive_rows", 0)),
            "foundation_rows_per_interval": self.frame_metadata_.get("rows_per_interval", []),
            "foundation_positive_rows_per_interval": self.frame_metadata_.get("positive_rows_per_interval", []),
            "foundation_excluded_censored_in_interval_rows": int(
                self.frame_metadata_.get("excluded_censored_in_interval_rows", 0)
            ),
            "foundation_sample_weight_supported": bool(self.sample_weight_supported_),
            "foundation_sample_weight_requested": self.params.get("subject_weighting"),
            "foundation_sample_weight_applied": bool(self.sample_weight_applied_),
            "foundation_censoring_weighting": self.params.get("censoring_weighting"),
            "foundation_ipcw_status": self.frame_metadata_.get("ipcw_status", "not_implemented"),
            "foundation_time_features": self.frame_metadata_.get("time_features", []),
            "foundation_discrete_hazard_fallback": bool(self.used_fallback_),
            "foundation_max_stacked_rows_applied": bool(self.frame_metadata_.get("max_stacked_rows_applied", False)),
            "foundation_predicted_hazard_min": self.last_hazard_min_,
            "foundation_predicted_hazard_max": self.last_hazard_max_,
        }

    def _build_training_frame(self, X_train: np.ndarray, time_train: np.ndarray, event_train: np.ndarray) -> DiscreteHazardFrame:
        grid = build_event_quantile_time_grid(
            time_train,
            event_train,
            time_grid=str(self.params["time_grid"]),
            n_intervals=int(self.params["n_intervals"]),
            horizon_quantiles=self.params.get("horizon_quantiles"),
            min_events_per_interval=int(self.params["min_events_per_interval"]),
        )
        self.time_grid_ = grid.endpoints
        self.grid_metadata_ = dict(grid.metadata)
        self.baseline_hazards_ = baseline_hazards_from_km(time_train, event_train, self.time_grid_)
        frame = build_discrete_hazard_frame(
            X=X_train,
            time=time_train,
            event=event_train,
            time_grid=self.time_grid_,
            time_feature_spec=str(self.params["time_feature_set"]),
            subject_weighting=str(self.params["subject_weighting"]),
            censoring_weighting=str(self.params["censoring_weighting"]),
            max_stacked_rows=self.params.get("max_stacked_rows"),
            seed=self.params.get("seed"),
        )
        self.frame_metadata_ = {**self.grid_metadata_, **frame.metadata}
        return frame

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_PooledDiscreteTimeHazardSurvivalMethod":
        del X_val, time_val, event_val
        method_id = self.method_id or self.__class__.__name__
        checkpoint_path = self.params.get("checkpoint_path")
        ensure_foundation_runtime_ready(method_id, checkpoint_path=checkpoint_path)
        try:
            X_train_np = np.asarray(X_train, dtype=np.float32)
            self.time_train_ = np.asarray(time_train, dtype=np.float64)
            self.event_train_ = np.asarray(event_train, dtype=np.int32)
            frame = self._build_training_frame(X_train_np, self.time_train_, self.event_train_)
            if int(frame.y_stacked.size) < int(self.params["min_rows_per_interval"]) or np.unique(frame.y_stacked).size < 2:
                self.model_ = None
                self.used_fallback_ = True
                return self

            self.model_ = self._build_backbone()
            self.sample_weight_supported_, self.sample_weight_applied_ = _fit_classifier(
                self.model_,
                np.asarray(frame.X_stacked, dtype=np.float32),
                np.asarray(frame.y_stacked, dtype=np.int32),
                frame.row_weights,
            )
            self.used_fallback_ = False
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(method_id, exc, checkpoint_path=checkpoint_path) from exc

    def _hazards(self, X: np.ndarray) -> np.ndarray:
        if self.time_grid_ is None or self.baseline_hazards_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        X_np = np.asarray(X, dtype=np.float32)
        if self.model_ is None:
            hazards = np.tile(self.baseline_hazards_, (X_np.shape[0], 1))
        else:
            columns: list[np.ndarray] = []
            batch_size = self.params.get("predict_batch_size")
            for idx in range(len(self.time_grid_)):
                query = append_time_bin_features(
                    X_np,
                    np.full(X_np.shape[0], idx, dtype=np.int32),
                    self.time_grid_,
                    time_train=self.time_train_,
                    event_train=self.event_train_,
                    time_feature_set=str(self.params["time_feature_set"]),
                )
                columns.append(positive_class_probability_with_backoff(self.model_, query, batch_size=batch_size))
            hazards = np.column_stack(columns)
        clean = clean_hazards(hazards)
        self.last_hazard_min_ = float(np.min(clean)) if clean.size else None
        self.last_hazard_max_ = float(np.max(clean)) if clean.size else None
        return clean

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return risk_from_hazards(self._hazards(X), aggregate_risk=str(self.params["aggregate_risk"]))

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.time_grid_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        return survival_from_hazards(self._hazards(X), self.time_grid_, times)

    def predict_bundle(self, X: np.ndarray, times: np.ndarray) -> SurvivalPredictions:
        if self.time_grid_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        hazards = self._hazards(X)
        return SurvivalPredictions(
            risk=risk_from_hazards(hazards, aggregate_risk=str(self.params["aggregate_risk"])),
            survival=survival_from_hazards(hazards, self.time_grid_, times),
        )


class TabPFNDiscreteHazardSurvivalMethod(_PooledDiscreteTimeHazardSurvivalMethod):
    method_id = "tabpfn_discrete_hazard_survival"
    foundation_backbone = "TabPFN"

    def __init__(
        self,
        n_estimators: int = 4,
        fit_mode: str = "fit_preprocessors",
        model_version: str = "v2.5",
        checkpoint_path: str | None = None,
        device: str = "auto",
        **params: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            fit_mode=fit_mode,
            model_version=model_version,
            checkpoint_path=checkpoint_path,
            device=device,
            **params,
        )

    def foundation_metadata(self) -> dict[str, Any]:
        metadata = super().foundation_metadata()
        metadata["foundation_n_estimators"] = int(self.params["n_estimators"])
        metadata["foundation_model_version"] = self.params["model_version"]
        return metadata

    def _build_backbone(self) -> Any:
        return build_tabpfn_classifier(
            n_estimators=int(self.params["n_estimators"]),
            fit_mode=str(self.params["fit_mode"]),
            model_version=str(self.params["model_version"]),
            checkpoint_path=self.params.get("checkpoint_path"),
            device=str(self.params["device"]),
            seed=self.params.get("seed"),
        )


class TabICLDiscreteHazardSurvivalMethod(_PooledDiscreteTimeHazardSurvivalMethod):
    method_id = "tabicl_discrete_hazard_survival"
    foundation_backbone = "TabICL"

    def __init__(
        self,
        n_estimators: int = 1,
        batch_size: int = 8,
        checkpoint_version: str = "tabicl-classifier-v1.1-0506.ckpt",
        device: str | None = None,
        use_amp: bool = False,
        allow_auto_download: bool = True,
        **params: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            batch_size=batch_size,
            checkpoint_version=checkpoint_version,
            device=device,
            use_amp=use_amp,
            allow_auto_download=allow_auto_download,
            **params,
        )

    def foundation_metadata(self) -> dict[str, Any]:
        metadata = super().foundation_metadata()
        metadata["foundation_n_estimators"] = int(self.params["n_estimators"])
        return metadata

    def _build_backbone(self) -> Any:
        from tabicl import TabICLClassifier

        return TabICLClassifier(
            n_estimators=int(self.params["n_estimators"]),
            batch_size=int(self.params["batch_size"]),
            checkpoint_version=str(self.params["checkpoint_version"]),
            device=self.params.get("device"),
            use_amp=bool(self.params["use_amp"]),
            allow_auto_download=bool(self.params["allow_auto_download"]),
            random_state=self.params.get("seed"),
            verbose=False,
        )


class TabICLSurvivalMethod(TabICLDiscreteHazardSurvivalMethod):
    """Compatibility ID for the default TabICL discrete-hazard survival adapter."""

    method_id = "tabicl_survival"


TabPFNPooledHazardSurvivalMethod = TabPFNDiscreteHazardSurvivalMethod
TabICLPooledHazardSurvivalMethod = TabICLDiscreteHazardSurvivalMethod
