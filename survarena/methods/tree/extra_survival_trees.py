from __future__ import annotations

import numpy as np

from survarena.methods.base import BaseSurvivalMethod, to_structured_y


class ExtraSurvivalTreesMethod(BaseSurvivalMethod):
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: str | None = "sqrt",
        bootstrap: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
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
    ) -> "ExtraSurvivalTreesMethod":
        from sksurv.ensemble import ExtraSurvivalTrees

        self.model = ExtraSurvivalTrees(
            n_estimators=int(self.params["n_estimators"]),
            max_depth=self.params["max_depth"],
            min_samples_split=int(self.params["min_samples_split"]),
            min_samples_leaf=int(self.params["min_samples_leaf"]),
            max_features=self.params["max_features"],
            bootstrap=bool(self.params["bootstrap"]),
            n_jobs=-1,
            random_state=None if self.params.get("seed") is None else int(self.params["seed"]),
        )
        self.model.fit(X_train, to_structured_y(time_train, event_train))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ExtraSurvivalTreesMethod must be fit before prediction.")
        return self.model.predict(X)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ExtraSurvivalTreesMethod must be fit before prediction.")
        fns = self.model.predict_survival_function(X)
        return np.vstack([fn(times) for fn in fns])
