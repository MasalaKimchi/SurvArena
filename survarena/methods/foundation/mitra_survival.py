from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from typing import Any

import numpy as np
import pandas as pd

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.foundation.readiness import ensure_foundation_runtime_ready, rewrite_foundation_runtime_error


def _import_torch():
    import torch

    return torch


def _train_neural_cox_head(**kwargs):
    from survarena.methods.foundation.neural_cox import train_neural_cox_head

    return train_neural_cox_head(**kwargs)


def _predict_head_risk(*, head: Any, features: np.ndarray, device: Any, batch_size: int) -> np.ndarray:
    torch = _import_torch()
    from survarena.methods.foundation.neural_cox import forward_in_chunks

    features_t = torch.as_tensor(features, dtype=torch.float32, device=device)
    head.eval()
    with torch.no_grad():
        risk = forward_in_chunks(head, features_t, batch_size=batch_size).detach().cpu().numpy()
    return np.asarray(risk, dtype=np.float64)


def _array_to_feature_frame(X: np.ndarray) -> pd.DataFrame:
    array = np.asarray(X, dtype=np.float32)
    columns = [f"f{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


class MitraSurvivalMethod(BaseSurvivalMethod):
    def __init__(
        self,
        backbone_training: str = "frozen",
        fine_tune_steps: int | None = None,
        presets: str = "medium",
        time_limit: int | None = None,
        path: str | None = None,
        hidden_layers: str | list[int] = "128-64",
        activation: str = "relu",
        dropout: float = 0.1,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 150,
        patience: int = 20,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            backbone_training=backbone_training,
            fine_tune_steps=fine_tune_steps,
            presets=presets,
            time_limit=time_limit,
            path=path,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            seed=seed,
        )
        self.backbone = None
        self.head = None
        self.survival_head = None
        self.path_: str | None = None
        self.device_: Any | None = None
        self.head_input_dim_: int | None = None
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
    ) -> "MitraSurvivalMethod":
        ensure_foundation_runtime_ready("mitra_survival")
        try:
            from autogluon.tabular import TabularPredictor

            frame = _array_to_feature_frame(X_train)
            label = "__survarena_log_time__"
            train_frame = frame.copy()
            train_frame[label] = np.log1p(np.asarray(time_train, dtype=np.float32))

            self.path_ = self.params["path"] or mkdtemp(prefix="survarena_mitra_")
            self.backbone = TabularPredictor(
                label=label,
                problem_type="regression",
                eval_metric="rmse",
                path=str(Path(self.path_)),
            )

            fit_kwargs: dict[str, object] = {
                "train_data": train_frame,
                "hyperparameters": {
                    "MITRA": {
                        "fine_tune": str(self.params["backbone_training"]).lower() == "finetune",
                    }
                },
                "presets": str(self.params["presets"]),
                "verbosity": 0,
            }
            if self.params["fine_tune_steps"] is not None:
                fit_kwargs["hyperparameters"]["MITRA"]["fine_tune_steps"] = int(self.params["fine_tune_steps"])  # type: ignore[index]
            if self.params["time_limit"] is not None:
                fit_kwargs["time_limit"] = int(self.params["time_limit"])

            self.backbone.fit(**fit_kwargs)

            train_backbone_feature = self._backbone_features(frame)
            val_backbone_feature = None
            if X_val is not None and time_val is not None and event_val is not None:
                val_backbone_feature = self._backbone_features(_array_to_feature_frame(X_val))

            artifacts = _train_neural_cox_head(
                train_features=train_backbone_feature,
                time_train=time_train,
                event_train=event_train,
                hidden_layers=self.params["hidden_layers"],
                activation=str(self.params["activation"]),
                dropout=float(self.params["dropout"]),
                optimizer=str(self.params["optimizer"]),
                lr=float(self.params["lr"]),
                weight_decay=float(self.params["weight_decay"]),
                batch_size=int(self.params["batch_size"]),
                max_epochs=int(self.params["max_epochs"]),
                patience=int(self.params["patience"]),
                device=str(self.params["device"]),
                seed=self.params.get("seed"),
                val_features=val_backbone_feature,
                time_val=time_val,
                event_val=event_val,
            )
            self.head = artifacts.head
            self.survival_head = self.head
            self.device_ = artifacts.device
            self.head_input_dim_ = artifacts.input_dim
            self.baseline_event_times_ = artifacts.baseline_event_times
            self.baseline_survival_ = artifacts.baseline_survival
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error("mitra_survival", exc) from exc

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.backbone is None or self.head is None or self.device_ is None:
            raise RuntimeError("MitraSurvivalMethod must be fit before prediction.")
        features = self._backbone_features(_array_to_feature_frame(X))
        return _predict_head_risk(
            head=self.head,
            features=features,
            device=self.device_,
            batch_size=int(self.params["batch_size"]),
        )

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError("MitraSurvivalMethod must be fit before prediction.")
        eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
        risk_scores = self.predict_risk(X)
        rel_risk = np.exp(risk_scores)

        last_surv = float(self.baseline_survival_[-1]) if self.baseline_survival_.size else 1.0
        baseline_at_times = np.interp(
            eval_times,
            self.baseline_event_times_,
            self.baseline_survival_,
            left=1.0,
            right=last_surv,
        )
        survival = np.power(np.clip(baseline_at_times, 1e-8, 1.0)[None, :], rel_risk[:, None])
        return np.clip(survival, 1e-8, 1.0).astype(np.float64)

    def _backbone_features(self, frame: pd.DataFrame) -> np.ndarray:
        if self.backbone is None:
            raise RuntimeError("Backbone must be fit before prediction.")
        predicted_log_time = np.asarray(self.backbone.predict(frame), dtype=np.float32).reshape(-1, 1)
        return predicted_log_time
