from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.survival_utils import risk_from_survival_frame, survival_frame_to_array


class _BaseLifelinesMethod(BaseSurvivalMethod, ABC):
    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.model = None
        self.feature_columns_: list[str] | None = None

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_BaseLifelinesMethod":
        self.feature_columns_ = [f"feature_{idx}" for idx in range(X_train.shape[1])]
        train_frame = self._frame_from_array(X_train)
        train_frame["time"] = np.asarray(time_train, dtype=np.float64)
        train_frame["event"] = np.asarray(event_train, dtype=np.int32)
        if int(train_frame["event"].sum()) <= 0:
            raise ValueError(f"{type(self).__name__} requires at least one observed event in the training data.")

        self.model = self._build_model()
        scipy_method = getattr(self, "_lifelines_scipy_fit_method", None)
        if scipy_method and hasattr(self.model, "_scipy_fit_method"):
            self.model._scipy_fit_method = str(scipy_method)
        self.model.fit(train_frame, duration_col="time", event_col="event")
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must be fit before prediction.")
        survival_frame = self.model.predict_survival_function(self._frame_from_array(X))
        return risk_from_survival_frame(survival_frame)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must be fit before prediction.")
        survival_frame = self.model.predict_survival_function(self._frame_from_array(X))
        return survival_frame_to_array(survival_frame, times)

    def _frame_from_array(self, X: np.ndarray) -> pd.DataFrame:
        if self.feature_columns_ is None:
            raise RuntimeError(f"{type(self).__name__} must be fit before prediction.")
        return pd.DataFrame(np.asarray(X, dtype=np.float64), columns=self.feature_columns_)

    @abstractmethod
    def _build_model(self) -> object:
        raise NotImplementedError


class WeibullAFTMethod(_BaseLifelinesMethod):
    _lifelines_scipy_fit_method = "SLSQP"

    def __init__(
        self,
        penalizer: float = 0.01,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        model_ancillary: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            model_ancillary=model_ancillary,
            seed=seed,
        )

    def _build_model(self) -> object:
        from lifelines import WeibullAFTFitter

        return WeibullAFTFitter(
            penalizer=float(self.params["penalizer"]),
            l1_ratio=float(self.params["l1_ratio"]),
            fit_intercept=bool(self.params["fit_intercept"]),
            model_ancillary=bool(self.params["model_ancillary"]),
        )


class LogNormalAFTMethod(_BaseLifelinesMethod):
    _lifelines_scipy_fit_method = "SLSQP"

    def __init__(
        self,
        penalizer: float = 0.01,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        model_ancillary: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            model_ancillary=model_ancillary,
            seed=seed,
        )

    def _build_model(self) -> object:
        from lifelines import LogNormalAFTFitter

        return LogNormalAFTFitter(
            penalizer=float(self.params["penalizer"]),
            l1_ratio=float(self.params["l1_ratio"]),
            fit_intercept=bool(self.params["fit_intercept"]),
            model_ancillary=bool(self.params["model_ancillary"]),
        )


class LogLogisticAFTMethod(_BaseLifelinesMethod):
    _lifelines_scipy_fit_method = "SLSQP"

    def __init__(
        self,
        penalizer: float = 0.01,
        l1_ratio: float = 0.0,
        fit_intercept: bool = True,
        model_ancillary: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            penalizer=penalizer,
            l1_ratio=l1_ratio,
            fit_intercept=fit_intercept,
            model_ancillary=model_ancillary,
            seed=seed,
        )

    def _build_model(self) -> object:
        from lifelines import LogLogisticAFTFitter

        return LogLogisticAFTFitter(
            penalizer=float(self.params["penalizer"]),
            l1_ratio=float(self.params["l1_ratio"]),
            fit_intercept=bool(self.params["fit_intercept"]),
            model_ancillary=bool(self.params["model_ancillary"]),
        )


class AalenAdditiveMethod(_BaseLifelinesMethod):
    def __init__(
        self,
        coef_penalizer: float = 0.001,
        smoothing_penalizer: float = 0.0,
        fit_intercept: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            coef_penalizer=coef_penalizer,
            smoothing_penalizer=smoothing_penalizer,
            fit_intercept=fit_intercept,
            seed=seed,
        )

    def _build_model(self) -> object:
        from lifelines import AalenAdditiveFitter

        return AalenAdditiveFitter(
            fit_intercept=bool(self.params["fit_intercept"]),
            coef_penalizer=float(self.params["coef_penalizer"]),
            smoothing_penalizer=float(self.params["smoothing_penalizer"]),
        )
