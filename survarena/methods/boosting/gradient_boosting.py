from __future__ import annotations

import numpy as np

from survarena.methods.base import BaseSurvivalMethod, to_structured_y


class GradientBoostingSurvivalMethod(BaseSurvivalMethod):
    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_depth: int = 3,
        max_features: str | None = None,
        dropout_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            dropout_rate=dropout_rate,
            seed=seed,
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
    ) -> "GradientBoostingSurvivalMethod":
        from sksurv.ensemble import GradientBoostingSurvivalAnalysis

        self.model = GradientBoostingSurvivalAnalysis(
            loss="coxph",
            learning_rate=float(self.params["learning_rate"]),
            n_estimators=int(self.params["n_estimators"]),
            subsample=float(self.params["subsample"]),
            min_samples_split=int(self.params["min_samples_split"]),
            min_samples_leaf=int(self.params["min_samples_leaf"]),
            max_depth=int(self.params["max_depth"]),
            max_features=self.params["max_features"],
            dropout_rate=float(self.params["dropout_rate"]),
            random_state=None if self.params.get("seed") is None else int(self.params["seed"]),
        )
        self.model.fit(X_train, to_structured_y(time_train, event_train))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GradientBoostingSurvivalMethod must be fit before prediction.")
        return self.model.predict(X)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GradientBoostingSurvivalMethod must be fit before prediction.")
        fns = self.model.predict_survival_function(X)
        return np.vstack([fn(times) for fn in fns])


class ComponentwiseGradientBoostingMethod(BaseSurvivalMethod):
    def __init__(
        self,
        learning_rate: float = 0.05,
        n_estimators: int = 300,
        subsample: float = 1.0,
        dropout_rate: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            dropout_rate=dropout_rate,
            seed=seed,
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
    ) -> "ComponentwiseGradientBoostingMethod":
        from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis

        self.model = ComponentwiseGradientBoostingSurvivalAnalysis(
            loss="coxph",
            learning_rate=float(self.params["learning_rate"]),
            n_estimators=int(self.params["n_estimators"]),
            subsample=float(self.params["subsample"]),
            dropout_rate=float(self.params["dropout_rate"]),
            random_state=None if self.params.get("seed") is None else int(self.params["seed"]),
        )
        self.model.fit(X_train, to_structured_y(time_train, event_train))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ComponentwiseGradientBoostingMethod must be fit before prediction.")
        return self.model.predict(X)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ComponentwiseGradientBoostingMethod must be fit before prediction.")
        fns = self.model.predict_survival_function(X)
        return np.vstack([fn(times) for fn in fns])
