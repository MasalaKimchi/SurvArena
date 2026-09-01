from __future__ import annotations

from typing import Any

import numpy as np

from survarena.automl.autogluon_backend import (
    AutoGluonFitMetadata,
    fit_autogluon_event_predictor,
    horizon_event_labels,
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
    # fit() forwards the validation fold to AutoGluon as tuning_data for early
    # stopping / model selection (see fit_autogluon_event_predictor). All
    # subclasses (Mitra, TabM, RealTabPFN-V2, and their discrete-hazard variants)
    # inherit this behaviour.
    consumes_validation = True
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
        # M2: fixed horizon h (derived from training event times) used to define the
        # censoring-aware "event by h" classification target; stored so predict is
        # consistent with fit. `used_fallback_` honours the shared trivial-predictor contract.
        self.horizon_: float | None = None
        self.used_fallback_: bool = False

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
            time_train_arr = np.asarray(time_train, dtype=float)
            event_train_arr = np.asarray(event_train, dtype=int)

            # M2: Define the classification target as "event by a fixed horizon h" instead of
            # the raw (censoring-blind) event indicator. h is the configurable quantile
            # (default the median, 0.5) of the TRAINING event times only, so it never leaks
            # test/horizon information. Subjects censored before h have unknown status at h and
            # are dropped from the classifier's training frame.
            self.horizon_ = _training_event_horizon(
                time_train_arr,
                event_train_arr,
                quantile=float(params.get("event_horizon_quantile", 0.5)),
            )
            keep_mask, labels = horizon_event_labels(time_train_arr, event_train_arr, self.horizon_)
            X_train_h = _select_rows(X_train, keep_mask)
            labels_h = labels[keep_mask]

            # Degenerate target (too few usable rows or a single class after dropping subjects
            # censored before h) -> trivial KM-baseline fallback (per the shared contract).
            if int(labels_h.shape[0]) < 2 or np.unique(labels_h).size < 2:
                self._fit_trivial_fallback(time_train_arr, event_train_arr)
                return self

            # Build a validation frame with the SAME horizon and drop rule; only use it if it
            # is itself non-degenerate (>= 2 rows and both classes present).
            X_val_h = None
            labels_val_h = None
            if X_val is not None and time_val is not None and event_val is not None:
                time_val_arr = np.asarray(time_val, dtype=float)
                event_val_arr = np.asarray(event_val, dtype=int)
                val_mask, val_labels = horizon_event_labels(time_val_arr, event_val_arr, self.horizon_)
                if int(val_mask.sum()) >= 2 and np.unique(val_labels[val_mask]).size == 2:
                    X_val_h = _select_rows(X_val, val_mask)
                    labels_val_h = val_labels[val_mask]

            self.predictor_, self.fit_metadata_ = fit_autogluon_event_predictor(
                X_train=X_train_h,
                event_train=labels_h,
                X_val=X_val_h,
                event_val=labels_val_h,
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
            # Predicted risk = P(event by h) (higher = worse). Fit the Breslow baseline from
            # these corrected risk scores over all training subjects to obtain a monotone
            # survival curve consistent with the corrected target (the representation the
            # method already used); only the risk definition changes.
            train_risk = self.predict_risk(X_train)
            self.baseline_event_times_, self.baseline_survival_ = fit_breslow_baseline_survival(
                time_train=time_train_arr,
                event_train=event_train_arr,
                train_risk_scores=train_risk,
            )
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(method_id, exc) from exc

    def _fit_trivial_fallback(self, time_train: np.ndarray, event_train: np.ndarray) -> None:
        # Trivial KM-baseline fallback for degenerate horizon targets: emit a constant risk
        # (=> C-index ~ 0.5) and a KM-style baseline survival curve so the run is scored but
        # unambiguously flagged. Per the shared contract, expose `used_fallback_ = True`.
        self.predictor_ = None
        self.fit_metadata_ = None
        self.used_fallback_ = True
        time_train = np.asarray(time_train, dtype=float)
        event_train = np.asarray(event_train, dtype=int)
        # Constant (zero) risk => the Breslow baseline reduces to the shared KM-style curve.
        self.baseline_event_times_, self.baseline_survival_ = fit_breslow_baseline_survival(
            time_train=time_train,
            event_train=event_train,
            train_risk_scores=np.zeros(int(time_train.shape[0]), dtype=float),
        )

    def predict_risk(self, X: Any) -> np.ndarray:
        if self.used_fallback_:
            # Trivial predictor: constant risk, consistent with the KM baseline survival.
            return np.zeros(_row_count(X), dtype=float)
        if self.predictor_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        # Risk = P(event by h) from the censoring-aware classifier (positive-class probability).
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
            # M2: horizon h = quantile (default median) of training EVENT times; risk = P(event by h).
            "foundation_event_horizon": self.horizon_,
            "foundation_event_horizon_quantile": float(self.params.get("event_horizon_quantile", 0.5)),
            "foundation_used_fallback": bool(self.used_fallback_),
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


def _training_event_horizon(time: np.ndarray, event: np.ndarray, *, quantile: float) -> float:
    # Fixed horizon h derived from TRAINING data only: a quantile (default the median) of the
    # observed training EVENT times. If no events are observed, fall back to the quantile of all
    # observed times so a finite horizon is still available.
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    event_times = time[event.astype(bool)]
    if event_times.size == 0:
        event_times = time
    if event_times.size == 0:
        return 0.0
    return float(np.quantile(event_times, quantile))


def _select_rows(X: Any, mask: np.ndarray) -> Any:
    # Row subset that works for both pandas frames and numpy arrays. `mask` is a boolean array
    # aligned positionally to the rows of X.
    mask = np.asarray(mask, dtype=bool)
    if hasattr(X, "iloc"):
        return X[mask]
    return np.asarray(X)[mask]


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
