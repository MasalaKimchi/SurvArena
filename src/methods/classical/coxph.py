from __future__ import annotations

import numpy as np

from src.methods.base import BaseSurvivalMethod, to_structured_y


class CoxPHMethod(BaseSurvivalMethod):
    def __init__(self, alpha: float = 0.0001) -> None:
        super().__init__(alpha=alpha)
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "CoxPHMethod":
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        self.model = CoxPHSurvivalAnalysis(alpha=float(self.params["alpha"]))
        self.model.fit(X_train, to_structured_y(time_train, event_train))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CoxPHMethod must be fit before prediction.")
        return self.model.predict(X)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CoxPHMethod must be fit before prediction.")
        fns = self.model.predict_survival_function(X)
        return np.vstack([fn(times) for fn in fns])
