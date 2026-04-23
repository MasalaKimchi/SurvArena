from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from torch import nn
from torchsurv.loss.cox import neg_partial_log_likelihood
from torchsurv.loss.momentum import Momentum

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.deep.deepsurv import _activation_cls, _parse_hidden_layers


class DeepSurvMomentumMethod(BaseSurvivalMethod):
    """DeepSurv variant using torchsurv Momentum with Cox loss."""

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
        use_momentum_encoder: bool = True,
        momentum: float = 0.999,
        queue_size: int = 512,
        temperature: float = 1.0,
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
            use_momentum_encoder=use_momentum_encoder,
            momentum=momentum,
            queue_size=queue_size,
            temperature=temperature,
        )
        self.model: Momentum | None = None
        self.device: torch.device | None = None
        self.baseline_event_times_: np.ndarray | None = None
        self.baseline_survival_: np.ndarray | None = None

    def _resolve_device(self) -> torch.device:
        raw_device = str(self.params["device"])
        if raw_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(raw_device)

    @staticmethod
    def _set_torch_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_network(self, in_features: int) -> nn.Module:
        hidden_layers = _parse_hidden_layers(self.params["hidden_layers"])
        activation = _activation_cls(str(self.params["activation"]))
        dropout = float(self.params["dropout"])
        use_batch_norm = bool(self.params["batch_norm"])
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden_layers:
            layers.append(nn.Linear(prev, int(width)))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(int(width)))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = int(width)
        layers.append(nn.Linear(prev, 1, bias=False))
        return nn.Sequential(*layers)

    def _loss_fn(self, temperature: float) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        scale = max(float(temperature), 1e-6)

        def _scaled_loss(log_hz: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return neg_partial_log_likelihood(log_hz / scale, event, time)

        return _scaled_loss

    def _inference_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("DeepSurvMomentumMethod must be fit before prediction.")
        return self.model.target if bool(self.params["use_momentum_encoder"]) else self.model.online

    @staticmethod
    def _fit_baseline_survival(time_train: np.ndarray, event_train: np.ndarray, train_log_risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        event_mask = event_train.astype(bool)
        event_times = np.unique(time_train[event_mask])
        if event_times.size == 0:
            return np.asarray([1.0], dtype=np.float64), np.asarray([1.0], dtype=np.float64)
        exp_risk = np.exp(train_log_risk.astype(np.float64))
        hazards: list[float] = []
        for event_time in event_times:
            d_j = float(np.sum((time_train == event_time) & event_mask))
            r_j = float(np.sum(exp_risk[time_train >= event_time]))
            hazards.append(d_j / max(r_j, 1e-12))
        cumulative_hazard = np.cumsum(np.asarray(hazards, dtype=np.float64))
        baseline_survival = np.exp(-cumulative_hazard)
        return event_times.astype(np.float64), baseline_survival.astype(np.float64)

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "DeepSurvMomentumMethod":
        self.device = self._resolve_device()
        self._set_torch_seed(int(self.params["seed"]))
        X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        t_train_t = torch.as_tensor(time_train.astype(np.float32), dtype=torch.float32, device=self.device)
        e_train_t = torch.as_tensor(event_train.astype(bool), dtype=torch.bool, device=self.device)
        if int(e_train_t.sum().item()) <= 0:
            raise ValueError("DeepSurvMomentum requires at least one observed event in the training data.")

        X_val_t = None
        t_val_t = None
        e_val_t = None
        if X_val is not None and time_val is not None and event_val is not None:
            X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            t_val_t = torch.as_tensor(time_val.astype(np.float32), dtype=torch.float32, device=self.device)
            e_val_t = torch.as_tensor(event_val.astype(bool), dtype=torch.bool, device=self.device)

        batch_size = int(self.params["batch_size"])
        queue_size = int(self.params["queue_size"])
        steps = max(1, int(np.ceil(queue_size / max(batch_size, 1))))
        backbone = self._build_network(X_train.shape[1]).to(self.device)
        loss_fn = self._loss_fn(float(self.params["temperature"]))
        self.model = Momentum(
            backbone=backbone,
            loss=loss_fn,
            batchsize=batch_size,
            steps=steps,
            rate=float(self.params["momentum"]),
        ).to(self.device)

        opt_name = str(self.params["optimizer"]).lower()
        optimizer_cls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(
            self.model.online.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        max_epochs = int(self.params["max_epochs"])
        patience = int(self.params["patience"])
        use_momentum = bool(self.params["use_momentum_encoder"])

        best_online = deepcopy(self.model.online.state_dict())
        best_target = deepcopy(self.model.target.state_dict())
        best_score = float("inf")
        stale_epochs = 0

        for _ in range(max_epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            online_log_hz = self.model.online(X_train_t).squeeze(-1)
            train_loss = loss_fn(online_log_hz, e_train_t, t_train_t)
            train_loss.backward()
            optimizer.step()
            if use_momentum:
                rate = float(self.params["momentum"])
                with torch.no_grad():
                    for target_param, online_param in zip(self.model.target.parameters(), self.model.online.parameters()):
                        target_param.data.mul_(rate).add_(online_param.data, alpha=1.0 - rate)
            else:
                self.model.target.load_state_dict(self.model.online.state_dict())

            with torch.no_grad():
                if X_val_t is not None and t_val_t is not None and e_val_t is not None:
                    self.model.eval()
                    val_log_hz = self._inference_model()(X_val_t).squeeze(-1)
                    monitor = float(loss_fn(val_log_hz, e_val_t, t_val_t).item())
                else:
                    monitor = float(train_loss.item())

            if monitor + 1e-8 < best_score:
                best_score = monitor
                best_online = deepcopy(self.model.online.state_dict())
                best_target = deepcopy(self.model.target.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        self.model.online.load_state_dict(best_online)
        self.model.target.load_state_dict(best_target)
        self.model.eval()
        with torch.no_grad():
            train_log_hz = self._inference_model()(X_train_t).squeeze(-1).detach().cpu().numpy()
        self.baseline_event_times_, self.baseline_survival_ = self._fit_baseline_survival(
            time_train=np.asarray(time_train, dtype=np.float64),
            event_train=np.asarray(event_train, dtype=np.int32),
            train_log_risk=train_log_hz,
        )
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self.device is None:
            raise RuntimeError("DeepSurvMomentumMethod must be fit before prediction.")
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            risk = self._inference_model()(X_t).squeeze(-1).detach().cpu().numpy()
        return risk.astype(np.float64)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError("DeepSurvMomentumMethod must be fit before prediction.")
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
