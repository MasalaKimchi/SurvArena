from __future__ import annotations

from typing import Any

import numpy as np

from survarena.automl.autogluon_backend import (
    AutoGluonFitMetadata,
    fit_autogluon_event_predictor,
    predict_event_probability,
)
from survarena.methods.base import BaseSurvivalMethod, SurvivalPredictions
from survarena.methods.discrete_hazard_shared import (
    apply_discrete_hazard_defaults,
    build_discrete_hazard_training_frame,
    discrete_hazard_foundation_metadata,
    discrete_hazard_predictions,
    init_discrete_hazard_state,
    predict_discrete_hazards,
    should_use_discrete_hazard_fallback,
)
from survarena.methods.discrete_time import (
    build_discrete_hazard_frame,
    risk_from_hazards,
    survival_from_hazards,
)
from survarena.methods.foundation.readiness import ensure_foundation_runtime_ready, rewrite_foundation_runtime_error
from survarena.methods.survival_utils import fit_breslow_baseline_survival, predict_breslow_survival


class _AutoGluonEventRiskSurvivalBase(BaseSurvivalMethod):
    foundation_method_id = ""
    foundation_backbone = "AutoGluon"
    foundation_hyperparameter_key = "AG"
    foundation_training = "default"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.predictor_: Any | None = None
        self.fit_metadata_: AutoGluonFitMetadata | None = None
        self.baseline_event_times_: np.ndarray | None = None
        self.baseline_survival_: np.ndarray | None = None

    def fit(
        self,
        X_train: Any,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: Any | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_AutoGluonEventRiskSurvivalBase":
        params = dict(self.params)
        method_id = self.foundation_method_id or self.__class__.__name__
        try:
            ensure_foundation_runtime_ready(method_id)
            self.predictor_, self.fit_metadata_ = fit_autogluon_event_predictor(
                X_train=X_train,
                event_train=event_train,
                X_val=X_val,
                event_val=event_val,
                presets=params.get("presets", "medium"),
                time_limit=params.get("time_limit"),
                hyperparameters=params.get("hyperparameters"),
                hyperparameter_tune_kwargs=params.get("hyperparameter_tune_kwargs"),
                num_bag_folds=int(params.get("num_bag_folds", 0)),
                num_stack_levels=int(params.get("num_stack_levels", 0)),
                refit_full=params.get("refit_full", False),
                path=params.get("path"),
                verbosity=int(params.get("verbosity", 0)),
            )
            train_risk = self.predict_risk(X_train)
            self.baseline_event_times_, self.baseline_survival_ = fit_breslow_baseline_survival(
                time_train=np.asarray(time_train, dtype=float),
                event_train=np.asarray(event_train, dtype=int),
                train_risk_scores=train_risk,
            )
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(method_id, exc) from exc

    def predict_risk(self, X: Any) -> np.ndarray:
        if self.predictor_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        probabilities = predict_event_probability(self.predictor_, X)
        return np.asarray(probabilities, dtype=float)

    def predict_survival(self, X: Any, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before survival prediction.")
        return self._survival_from_risk(self.predict_risk(X), times)

    def _survival_from_risk(self, risk_scores: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before survival prediction.")
        return predict_breslow_survival(
            risk_scores=risk_scores,
            times=np.asarray(times, dtype=float),
            baseline_event_times=self.baseline_event_times_,
            baseline_survival=self.baseline_survival_,
        )

    def predict_bundle(self, X: Any, times: np.ndarray) -> SurvivalPredictions:
        risk = self.predict_risk(X)
        return SurvivalPredictions(risk=risk, survival=self._survival_from_risk(risk, times))

    def autogluon_metadata(self) -> dict[str, Any]:
        if self.fit_metadata_ is None:
            return {}
        return {
            "autogluon_best_model": self.fit_metadata_.best_model,
            "autogluon_model_count": self.fit_metadata_.model_count,
            "autogluon_path": self.fit_metadata_.path,
            "autogluon_leaderboard": list(self.fit_metadata_.leaderboard),
        }

    def foundation_metadata(self) -> dict[str, Any]:
        hyperparameters = dict(self.params.get("hyperparameters", {}) or {})
        backbone_params = dict(hyperparameters.get(self.foundation_hyperparameter_key, {}) or {})
        return {
            "foundation_backbone": self.foundation_backbone,
            "foundation_backbone_task": "classification_event",
            "foundation_backbone_training": self.foundation_training,
            "foundation_time_limit_sec": self.params.get("time_limit"),
            "foundation_autogluon_hyperparameter_key": self.foundation_hyperparameter_key,
            "foundation_autogluon_backbone_params": backbone_params,
        }


class _MitraSurvivalMethod(_AutoGluonEventRiskSurvivalBase):
    foundation_method_id = "mitra_survival_frozen"
    foundation_backbone = "Mitra"
    foundation_hyperparameter_key = "MITRA"

    def __init__(self, **params: Any) -> None:
        mitra_params = dict(params.pop("mitra_params", {}) or {})
        mitra_params.setdefault("fine_tune", False)
        resolved = {
            **params,
            "presets": params.pop("presets", None),
            "hyperparameters": {"MITRA": mitra_params},
        }
        super().__init__(**resolved)

    def fit(
        self,
        X_train: Any,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: Any | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_MitraSurvivalMethod":
        try:
            from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Mitra Survival requires AutoGluon's Mitra extra. "
                'Install it with `python -m pip install -e ".[foundation-mitra]"`.'
            ) from exc
        return super().fit(X_train, time_train, event_train, X_val, time_val, event_val)


class MitraSurvivalFrozenMethod(_MitraSurvivalMethod):
    foundation_training = "frozen"

    def __init__(self, **params: Any) -> None:
        mitra_params = dict(params.pop("mitra_params", {}) or {})
        mitra_params["fine_tune"] = False
        super().__init__(**params, mitra_params=mitra_params)

    def foundation_metadata(self) -> dict[str, Any]:
        metadata = super().foundation_metadata()
        hyperparameters = dict(self.params.get("hyperparameters", {}) or {})
        mitra_params = dict(hyperparameters.get("MITRA", {}) or {})
        metadata["foundation_mitra_fine_tune"] = bool(mitra_params.get("fine_tune", False))
        return metadata


class _AutoGluonFoundationSurvivalMethod(_AutoGluonEventRiskSurvivalBase):
    def __init__(self, **params: Any) -> None:
        backbone_params_key = f"{self.foundation_hyperparameter_key.lower().replace('-', '_')}_params"
        backbone_params = dict(params.pop(backbone_params_key, {}) or {})
        resolved = {
            **params,
            "presets": params.pop("presets", None),
            "hyperparameters": {self.foundation_hyperparameter_key: backbone_params},
        }
        super().__init__(**resolved)


def _row_count(X: Any) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return int(len(X))


class _AutoGluonDiscreteHazardSurvivalMethod(_AutoGluonFoundationSurvivalMethod):
    foundation_training = "default"

    def __init__(self, **params: Any) -> None:
        apply_discrete_hazard_defaults(params)
        super().__init__(**params)
        self.predictor_: Any | None = None
        self.fit_metadata_: AutoGluonFitMetadata | None = None
        init_discrete_hazard_state(self)

    def fit(
        self,
        X_train: Any,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: Any | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_AutoGluonDiscreteHazardSurvivalMethod":
        params = dict(self.params)
        method_id = self.foundation_method_id or self.__class__.__name__
        try:
            ensure_foundation_runtime_ready(method_id)
            frame = build_discrete_hazard_training_frame(
                self,
                X_train=X_train,
                time_train=time_train,
                event_train=event_train,
            )
            self.sample_weight_supported_ = False
            self.sample_weight_applied_ = False
            if should_use_discrete_hazard_fallback(self, frame):
                self.predictor_ = None
                self.fit_metadata_ = None
                self.used_fallback_ = True
                return self

            X_val_pt = None
            y_val_pt = None
            if X_val is not None and time_val is not None and event_val is not None:
                candidate_frame = build_discrete_hazard_frame(
                    X=X_val,
                    time=np.asarray(time_val, dtype=np.float64),
                    event=np.asarray(event_val, dtype=np.int32),
                    time_grid=self.time_grid_,
                    time_feature_spec=str(params["time_feature_set"]),
                    subject_weighting="none",
                    censoring_weighting="none",
                )
                if (
                    int(len(candidate_frame.y_stacked)) >= int(params["min_rows_per_interval"])
                    and np.unique(candidate_frame.y_stacked).size == 2
                ):
                    X_val_pt = candidate_frame.X_stacked
                    y_val_pt = candidate_frame.y_stacked

            self.predictor_, self.fit_metadata_ = fit_autogluon_event_predictor(
                X_train=frame.X_stacked,
                event_train=frame.y_stacked,
                X_val=X_val_pt,
                event_val=y_val_pt,
                presets=params.get("presets", "medium"),
                time_limit=params.get("time_limit"),
                hyperparameters=params.get("hyperparameters"),
                hyperparameter_tune_kwargs=params.get("hyperparameter_tune_kwargs"),
                num_bag_folds=int(params.get("num_bag_folds", 0)),
                num_stack_levels=int(params.get("num_stack_levels", 0)),
                refit_full=params.get("refit_full", False),
                path=params.get("path"),
                verbosity=int(params.get("verbosity", 0)),
            )
            self.used_fallback_ = False
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(method_id, exc) from exc

    def _hazards(self, X: Any) -> np.ndarray:
        if self.time_grid_ is None or self.baseline_hazards_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        row_count = _row_count(X)
        return predict_discrete_hazards(
            self,
            X=X,
            row_count=row_count,
            fitted_model=self.predictor_,
            probability_fn=lambda query: predict_event_probability(self.predictor_, query),
        )

    def predict_risk(self, X: Any) -> np.ndarray:
        return risk_from_hazards(self._hazards(X), aggregate_risk=str(self.params["aggregate_risk"]))

    def predict_survival(self, X: Any, times: np.ndarray) -> np.ndarray:
        if self.time_grid_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        return survival_from_hazards(self._hazards(X), self.time_grid_, times)

    def predict_bundle(self, X: Any, times: np.ndarray) -> SurvivalPredictions:
        hazards = self._hazards(X)
        return discrete_hazard_predictions(self, X, times, hazards)

    def autogluon_metadata(self) -> dict[str, Any]:
        if self.fit_metadata_ is None:
            return {}
        return {
            "autogluon_best_model": self.fit_metadata_.best_model,
            "autogluon_model_count": self.fit_metadata_.model_count,
            "autogluon_path": self.fit_metadata_.path,
            "autogluon_leaderboard": list(self.fit_metadata_.leaderboard),
        }

    def foundation_metadata(self) -> dict[str, Any]:
        metadata = discrete_hazard_foundation_metadata(
            self,
            backbone=self.foundation_backbone,
            training=self.foundation_training,
        )
        hyperparameters = dict(self.params.get("hyperparameters", {}) or {})
        backbone_params = dict(hyperparameters.get(self.foundation_hyperparameter_key, {}) or {})
        metadata["foundation_time_limit_sec"] = self.params.get("time_limit")
        metadata["foundation_autogluon_hyperparameter_key"] = self.foundation_hyperparameter_key
        metadata["foundation_autogluon_backbone_params"] = backbone_params
        return metadata


class TabMDiscreteHazardSurvivalMethod(_AutoGluonDiscreteHazardSurvivalMethod):
    foundation_method_id = "tabm_discrete_hazard_survival"
    foundation_backbone = "TabM"
    foundation_hyperparameter_key = "TABM"
    foundation_training = "fit"


class RealTabPFNV2DiscreteHazardSurvivalMethod(_AutoGluonDiscreteHazardSurvivalMethod):
    foundation_method_id = "realtabpfn_discrete_hazard_survival"
    foundation_backbone = "RealTabPFN-V2"
    foundation_hyperparameter_key = "REALTABPFN-V2"
    foundation_training = "in_context"


class TabMSurvivalMethod(TabMDiscreteHazardSurvivalMethod):
    foundation_method_id = "tabm_survival"


class RealTabPFNV2SurvivalMethod(RealTabPFNV2DiscreteHazardSurvivalMethod):
    foundation_method_id = "realtabpfn_survival"


TabMPooledHazardSurvivalMethod = TabMDiscreteHazardSurvivalMethod
RealTabPFNV2PooledHazardSurvivalMethod = RealTabPFNV2DiscreteHazardSurvivalMethod
