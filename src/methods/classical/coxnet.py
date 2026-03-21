from __future__ import annotations

import numpy as np

from src.methods.base import BaseSurvivalMethod, to_structured_y


class CoxNetMethod(BaseSurvivalMethod):
    def __init__(
        self,
        n_alphas: int = 100,
        l1_ratio: float = 0.5,
        alpha_min_ratio: float = 0.001,
    ) -> None:
        super().__init__(
            n_alphas=n_alphas,
            l1_ratio=l1_ratio,
            alpha_min_ratio=alpha_min_ratio,
        )
        self.model = None

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "CoxNetMethod":
        from sksurv.linear_model import CoxnetSurvivalAnalysis

        self.model = CoxnetSurvivalAnalysis(
            n_alphas=int(self.params["n_alphas"]),
            l1_ratio=float(self.params["l1_ratio"]),
            alpha_min_ratio=float(self.params["alpha_min_ratio"]),
            fit_baseline_model=True,
        )
        self.model.fit(X_train, to_structured_y(time_train, event_train))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CoxNetMethod must be fit before prediction.")
        return self.model.predict(X)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CoxNetMethod must be fit before prediction.")
        fns = self.model.predict_survival_function(X)
        return np.vstack([fn(times) for fn in fns])
