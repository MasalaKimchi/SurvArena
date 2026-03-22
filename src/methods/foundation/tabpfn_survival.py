from __future__ import annotations

import numpy as np

from src.methods.base import BaseSurvivalMethod, to_structured_y


class TabPFNSurvivalMethod(BaseSurvivalMethod):
    def __init__(
        self,
        n_estimators: int = 8,
        fit_mode: str = "fit_preprocessors",
        model_version: str = "auto",
        checkpoint_path: str | None = None,
        fine_tune: bool = False,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            fit_mode=fit_mode,
            model_version=model_version,
            checkpoint_path=checkpoint_path,
            fine_tune=fine_tune,
            device=device,
            seed=seed,
        )
        self.backbone = None
        self.survival_head = None
        self.embedding_dim_: int | None = None

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "TabPFNSurvivalMethod":
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        X_train_f = np.asarray(X_train, dtype=np.float32)
        target_time = np.log1p(np.asarray(time_train, dtype=np.float32))

        if bool(self.params["fine_tune"]):
            raise NotImplementedError(
                "TabPFN backbone fine-tuning is not exposed by the installed tabpfn package. "
                "Use model_version/checkpoint_path to choose official or custom pretrained weights."
            )

        base_kwargs = {
            "n_estimators": int(self.params["n_estimators"]),
            "fit_mode": str(self.params["fit_mode"]),
            "device": str(self.params["device"]),
            "random_state": self.params.get("seed"),
        }
        checkpoint_path = self.params.get("checkpoint_path")
        if checkpoint_path:
            self.backbone = TabPFNRegressor(model_path=str(checkpoint_path), **base_kwargs)
        else:
            model_version = str(self.params["model_version"]).lower()
            if model_version in {"auto", "default"}:
                self.backbone = TabPFNRegressor(**base_kwargs)
            else:
                version_map = {
                    "v2": ModelVersion.V2,
                    "v2.5": ModelVersion.V2_5,
                    "v2_5": ModelVersion.V2_5,
                }
                if model_version not in version_map:
                    raise ValueError("model_version must be one of {'auto', 'v2', 'v2.5'}.")
                self.backbone = TabPFNRegressor.create_default_for_version(
                    version=version_map[model_version],
                    **base_kwargs,
                )
        self.backbone.fit(X_train_f, target_time)
        train_embeddings = self._extract_embeddings(X_train_f)
        self.embedding_dim_ = int(train_embeddings.shape[1])

        self.survival_head = CoxPHSurvivalAnalysis(alpha=0.0001)
        self.survival_head.fit(train_embeddings, to_structured_y(time_train, event_train))
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.backbone is None or self.survival_head is None:
            raise RuntimeError("TabPFNSurvivalMethod must be fit before prediction.")
        embeddings = self._extract_embeddings(np.asarray(X, dtype=np.float32))
        return self.survival_head.predict(embeddings)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.backbone is None or self.survival_head is None:
            raise RuntimeError("TabPFNSurvivalMethod must be fit before prediction.")
        embeddings = self._extract_embeddings(np.asarray(X, dtype=np.float32))
        fns = self.survival_head.predict_survival_function(embeddings)
        eval_times = np.asarray(times, dtype=np.float64)
        return np.vstack([fn(eval_times) for fn in fns])

    def _extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        if self.backbone is None:
            raise RuntimeError("Backbone must be fit before extracting embeddings.")
        embeddings = self.backbone.get_embeddings(X, data_source="test")
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(axis=0)
        if embeddings.ndim != 2:
            raise ValueError(f"Unexpected embedding shape from TabPFN backbone: {embeddings.shape}")
        return embeddings
