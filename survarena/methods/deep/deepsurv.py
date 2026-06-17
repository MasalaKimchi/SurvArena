from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torchsurv.loss.cox import neg_partial_log_likelihood

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.deep.batching import batch_norm_safe_batch_size, resolve_torch_training_device
from survarena.methods.deep.common import (
    activation_cls as _activation_cls,
    build_mlp,
    parse_hidden_layers as _parse_hidden_layers,
    predict_log_risk_survival,
    set_torch_seed,
)
from survarena.methods.survival_utils import fit_breslow_baseline_survival

__all__ = ["DeepSurvMethod", "_activation_cls", "_parse_hidden_layers"]


class DeepSurvMethod(BaseSurvivalMethod):
    def __init__(
        self,
        hidden_layers: str | list[int] = "128-64",
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = True,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 200,
        patience: int = 20,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        super().__init__(
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
            device=device,
        )
        self.model: nn.Module | None = None
        self.device: torch.device | None = None
        self.baseline_event_times_: np.ndarray | None = None
        self.baseline_survival_: np.ndarray | None = None

    def _build_network(self, in_features: int) -> nn.Module:
        return build_mlp(
            in_features=in_features,
            out_features=1,
            hidden_layers=self.params["hidden_layers"],
            activation=str(self.params["activation"]),
            dropout=float(self.params["dropout"]),
            batch_norm=bool(self.params["batch_norm"]),
            output_bias=False,
        )

    @staticmethod
    def _cox_partial_log_likelihood_loss(log_hazard: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return neg_partial_log_likelihood(log_hazard, event, time)

    def _forward_in_chunks(self, X: torch.Tensor, batch_size: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            outputs.append(self.model(X[start:end]).squeeze(-1))  # type: ignore[misc]
        return torch.cat(outputs, dim=0)

    def _fit_baseline_survival(
        self,
        time_train: np.ndarray,
        event_train: np.ndarray,
        train_log_risk: np.ndarray,
    ) -> None:
        self.baseline_event_times_, self.baseline_survival_ = fit_breslow_baseline_survival(
            time_train=time_train,
            event_train=event_train,
            train_risk_scores=train_log_risk,
        )

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "DeepSurvMethod":
        self.device = resolve_torch_training_device(str(self.params["device"]))
        set_torch_seed(int(self.params["seed"]))

        X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        t_train_t = torch.as_tensor(time_train.astype(np.float32), dtype=torch.float32, device=self.device)
        e_train_t = torch.as_tensor(event_train.astype(bool), dtype=torch.bool, device=self.device)
        if int(e_train_t.sum().item()) <= 0:
            raise ValueError("DeepSurv requires at least one observed event in the training data.")

        X_val_t = None
        t_val_t = None
        e_val_t = None
        if X_val is not None and time_val is not None and event_val is not None:
            X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            t_val_t = torch.as_tensor(time_val.astype(np.float32), dtype=torch.float32, device=self.device)
            e_val_t = torch.as_tensor(event_val.astype(bool), dtype=torch.bool, device=self.device)

        self.model = self._build_network(X_train.shape[1]).to(self.device)
        opt_name = str(self.params["optimizer"]).lower()
        optimizer_cls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(
            self.model.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        batch_size = batch_norm_safe_batch_size(
            len(X_train_t),
            int(self.params["batch_size"]),
            batch_norm=bool(self.params["batch_norm"]),
        )
        max_epochs = int(self.params["max_epochs"])
        patience = int(self.params["patience"])

        best_state = deepcopy(self.model.state_dict())
        best_score = float("inf")
        stale_epochs = 0

        for _ in range(max_epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            train_log_hazard = self._forward_in_chunks(X_train_t, batch_size=batch_size)
            train_loss = self._cox_partial_log_likelihood_loss(train_log_hazard, e_train_t, t_train_t)
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                if X_val_t is not None and t_val_t is not None and e_val_t is not None:
                    self.model.eval()
                    val_log_hazard = self._forward_in_chunks(X_val_t, batch_size=batch_size)
                    monitor = float(self._cox_partial_log_likelihood_loss(val_log_hazard, e_val_t, t_val_t).item())
                else:
                    monitor = float(train_loss.item())

            if monitor + 1e-8 < best_score:
                best_score = monitor
                best_state = deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()
        with torch.no_grad():
            train_log_hazard = self._forward_in_chunks(X_train_t, batch_size=batch_size).detach().cpu().numpy()
        self._fit_baseline_survival(
            time_train=np.asarray(time_train, dtype=np.float64),
            event_train=np.asarray(event_train, dtype=np.int32),
            train_log_risk=train_log_hazard,
        )
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("DeepSurvMethod must be fit before prediction.")
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        batch_size = int(self.params["batch_size"])
        self.model.eval()
        with torch.no_grad():
            risk = self._forward_in_chunks(X_t, batch_size=batch_size).detach().cpu().numpy()
        return risk.astype(np.float64)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError("DeepSurvMethod must be fit before prediction.")
        return predict_log_risk_survival(
            risk_scores=self.predict_risk(X),
            times=times,
            baseline_event_times=self.baseline_event_times_,
            baseline_survival=self.baseline_survival_,
        )
