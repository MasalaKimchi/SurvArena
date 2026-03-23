from __future__ import annotations

import numpy as np

from survarena.methods.base import BaseSurvivalMethod, to_structured_y


class CoxNetMethod(BaseSurvivalMethod):
    def __init__(
        self,
        alpha: float | None = 0.001,
        l1_ratio: float = 0.5,
        n_alphas: int = 100,
        alpha_min_ratio: float = 0.001,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            alpha=alpha,
            n_alphas=n_alphas,
            l1_ratio=l1_ratio,
            alpha_min_ratio=alpha_min_ratio,
            seed=seed,
        )
        self.model = None
        self.selected_alpha_: float | None = float(alpha) if alpha is not None else None

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

        alpha = self.params.get("alpha")
        if alpha is None:
            self.model = CoxnetSurvivalAnalysis(
                n_alphas=int(self.params["n_alphas"]),
                l1_ratio=float(self.params["l1_ratio"]),
                alpha_min_ratio=float(self.params["alpha_min_ratio"]),
                fit_baseline_model=True,
            )
        else:
            self.model = CoxnetSurvivalAnalysis(
                alphas=np.asarray([float(alpha)], dtype=np.float64),
                l1_ratio=float(self.params["l1_ratio"]),
                fit_baseline_model=True,
            )
        self.model.fit(X_train, to_structured_y(time_train, event_train))
        if alpha is None:
            self.selected_alpha_ = float(self.model.alphas_[-1])
        else:
            self.selected_alpha_ = float(alpha)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CoxNetMethod must be fit before prediction.")
        return self.model.predict(X, alpha=self.selected_alpha_)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("CoxNetMethod must be fit before prediction.")
        fns = self.model.predict_survival_function(X, alpha=self.selected_alpha_)
        return np.vstack([fn(times) for fn in fns])
