from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from survarena.methods.base import BaseSurvivalMethod, SurvivalPredictions
from survarena.methods.discrete_hazard_shared import (
    build_discrete_hazard_training_frame,
    discrete_hazard_foundation_metadata,
    discrete_hazard_predictions,
    init_discrete_hazard_state,
    predict_discrete_hazards,
    should_use_discrete_hazard_fallback,
)
from survarena.methods.discrete_time import (
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
        init_discrete_hazard_state(self)

    def _build_backbone(self) -> Any:
        raise NotImplementedError

    def foundation_metadata(self) -> dict[str, Any]:
        return discrete_hazard_foundation_metadata(
            self,
            backbone=self.foundation_backbone,
            training=self.foundation_training,
        )

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
            frame = build_discrete_hazard_training_frame(
                self,
                X_train=X_train_np,
                time_train=time_train,
                event_train=event_train,
            )
            if should_use_discrete_hazard_fallback(self, frame):
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
        batch_size = self.params.get("predict_batch_size")
        return predict_discrete_hazards(
            self,
            X=X_np,
            row_count=X_np.shape[0],
            fitted_model=self.model_,
            probability_fn=lambda query: positive_class_probability_with_backoff(
                self.model_,
                query,
                batch_size=batch_size,
            ),
        )

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
        return discrete_hazard_predictions(self, X, times, hazards)


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
