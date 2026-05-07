from __future__ import annotations

from typing import Any

import numpy as np

from survarena.automl.autogluon_backend import (
    AutoGluonFitMetadata,
    fit_autogluon_event_predictor,
    predict_event_probability,
)
from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.survival_utils import fit_breslow_baseline_survival, predict_breslow_survival


class _AutoGluonEventRiskSurvivalBase(BaseSurvivalMethod):
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

    def predict_risk(self, X: Any) -> np.ndarray:
        if self.predictor_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before prediction.")
        probabilities = predict_event_probability(self.predictor_, X)
        return np.asarray(probabilities, dtype=float)

    def predict_survival(self, X: Any, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be fit before survival prediction.")
        return predict_breslow_survival(
            risk_scores=self.predict_risk(X),
            times=np.asarray(times, dtype=float),
            baseline_event_times=self.baseline_event_times_,
            baseline_survival=self.baseline_survival_,
        )

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
        mitra_params = dict(hyperparameters.get("MITRA", {}) or {})
        return {
            "foundation_backbone": "Mitra",
            "foundation_backbone_task": "classification_event",
            "foundation_backbone_training": "finetune" if bool(mitra_params.get("fine_tune", False)) else "frozen",
            "foundation_time_limit_sec": self.params.get("time_limit"),
            "foundation_mitra_fine_tune": bool(mitra_params.get("fine_tune", False)),
        }


class MitraSurvivalMethod(_AutoGluonEventRiskSurvivalBase):
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
    ) -> "MitraSurvivalMethod":
        try:
            from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Mitra Survival requires AutoGluon's Mitra extra. "
                'Install it with `python -m pip install -e ".[foundation-mitra]"`.'
            ) from exc
        return super().fit(X_train, time_train, event_train, X_val, time_val, event_val)


class MitraSurvivalFrozenMethod(MitraSurvivalMethod):
    def __init__(self, **params: Any) -> None:
        mitra_params = dict(params.pop("mitra_params", {}) or {})
        mitra_params["fine_tune"] = False
        super().__init__(**params, mitra_params=mitra_params)


class MitraSurvivalFineTuneMethod(MitraSurvivalMethod):
    def __init__(self, **params: Any) -> None:
        mitra_params = dict(params.pop("mitra_params", {}) or {})
        mitra_params["fine_tune"] = True
        super().__init__(**params, mitra_params=mitra_params)
