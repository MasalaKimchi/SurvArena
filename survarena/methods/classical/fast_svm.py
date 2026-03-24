from __future__ import annotations

import numpy as np

from survarena.methods.base import BaseSurvivalMethod, to_structured_y
from survarena.methods.survival_utils import fit_breslow_baseline_survival, predict_breslow_survival


class FastSurvivalSVMMethod(BaseSurvivalMethod):
    def __init__(
        self,
        alpha: float = 1.0,
        rank_ratio: float = 1.0,
        fit_intercept: bool = False,
        max_iter: int = 100,
        tol: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            alpha=alpha,
            rank_ratio=rank_ratio,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        self.model = None
        self.baseline_event_times_: np.ndarray | None = None
        self.baseline_survival_: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "FastSurvivalSVMMethod":
        from sksurv.svm import FastSurvivalSVM

        self.model = FastSurvivalSVM(
            alpha=float(self.params["alpha"]),
            rank_ratio=float(self.params["rank_ratio"]),
            fit_intercept=bool(self.params["fit_intercept"]),
            max_iter=int(self.params["max_iter"]),
            tol=self.params["tol"],
            random_state=None if self.params.get("seed") is None else int(self.params["seed"]),
        )
        self.model.fit(X_train, to_structured_y(time_train, event_train))
        train_risk = self.predict_risk(X_train)
        self.baseline_event_times_, self.baseline_survival_ = fit_breslow_baseline_survival(
            time_train=np.asarray(time_train, dtype=np.float64),
            event_train=np.asarray(event_train, dtype=np.int32),
            train_risk_scores=train_risk,
        )
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FastSurvivalSVMMethod must be fit before prediction.")
        return (-self.model.predict(X)).astype(np.float64)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError("FastSurvivalSVMMethod must be fit before prediction.")
        return predict_breslow_survival(
            risk_scores=self.predict_risk(X),
            times=np.asarray(times, dtype=np.float64),
            baseline_event_times=self.baseline_event_times_,
            baseline_survival=self.baseline_survival_,
        )
