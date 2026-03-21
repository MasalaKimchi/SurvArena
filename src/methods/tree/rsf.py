from __future__ import annotations

import numpy as np

from src.methods.base import BaseSurvivalMethod, to_structured_y


class RSFMethod(BaseSurvivalMethod):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str | None = "sqrt",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
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
    ) -> "RSFMethod":
        from sksurv.ensemble import RandomSurvivalForest

        self.model = RandomSurvivalForest(
            n_estimators=int(self.params["n_estimators"]),
            max_depth=self.params["max_depth"],
            min_samples_split=int(self.params["min_samples_split"]),
            min_samples_leaf=int(self.params["min_samples_leaf"]),
            max_features=self.params["max_features"],
            n_jobs=-1,
            random_state=None if self.params.get("seed") is None else int(self.params["seed"]),
        )
        self.model.fit(X_train, to_structured_y(time_train, event_train))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RSFMethod must be fit before prediction.")
        return self.model.predict(X)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("RSFMethod must be fit before prediction.")
        fns = self.model.predict_survival_function(X)
        return np.vstack([fn(times) for fn in fns])
