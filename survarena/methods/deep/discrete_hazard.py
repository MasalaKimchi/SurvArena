from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from survarena.methods.base import BaseSurvivalMethod, SurvivalPredictions
from survarena.methods.deep.batching import batch_norm_safe_batch_size, resolve_torch_training_device
from survarena.methods.deep.deepsurv import _activation_cls, _parse_hidden_layers
from survarena.methods.discrete_time import (
    clean_hazards,
    event_quantile_time_bins,
    interval_label_matrix,
    risk_from_hazards,
    survival_from_hazards,
)


class SharedDiscreteHazardMethod(BaseSurvivalMethod):
    def __init__(
        self,
        time_bin_quantiles: str | list[float] = "0.25-0.5-0.75",
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
        aggregate_risk: str = "mean_event_probability",
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        super().__init__(
            time_bin_quantiles=time_bin_quantiles,
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
            aggregate_risk=aggregate_risk,
            seed=seed,
            device=device,
        )
        self.model_: nn.Module | None = None
        self.device_: torch.device | None = None
        self.time_bins_: np.ndarray | None = None
        self.train_loss_trace_: list[float] = []
        self.val_loss_trace_: list[float] = []

    @staticmethod
    def _set_torch_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_network(self, in_features: int, out_features: int) -> nn.Module:
        hidden_layers = _parse_hidden_layers(self.params["hidden_layers"])
        activation = _activation_cls(str(self.params["activation"]))
        dropout = float(self.params["dropout"])
        use_batch_norm = bool(self.params["batch_norm"])
        layers: list[nn.Module] = []
        previous = int(in_features)
        for width in hidden_layers:
            layers.append(nn.Linear(previous, int(width)))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(int(width)))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous = int(width)
        layers.append(nn.Linear(previous, int(out_features)))
        return nn.Sequential(*layers)

    @staticmethod
    def _masked_bce(logits: torch.Tensor, labels: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        weighted = loss * known.to(dtype=loss.dtype)
        return weighted.sum() / torch.clamp(known.to(dtype=loss.dtype).sum(), min=1.0)

    def _forward_in_chunks(self, X: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.model_ is None:
            raise RuntimeError("SharedDiscreteHazardMethod must be fit before prediction.")
        outputs: list[torch.Tensor] = []
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            outputs.append(self.model_(X[start:end]))
        return torch.cat(outputs, dim=0)

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "SharedDiscreteHazardMethod":
        self.device_ = resolve_torch_training_device(str(self.params["device"]))
        self._set_torch_seed(int(self.params["seed"]))
        time_train_np = np.asarray(time_train, dtype=np.float64)
        event_train_np = np.asarray(event_train, dtype=np.int32)
        self.time_bins_ = event_quantile_time_bins(
            time_train_np,
            event_train_np,
            self.params["time_bin_quantiles"],
        )
        train_known_np, train_labels_np = interval_label_matrix(
            time=time_train_np,
            event=event_train_np,
            time_bins=self.time_bins_,
        )
        if int(train_known_np.sum()) <= 0 or int(train_labels_np.sum()) <= 0:
            raise ValueError("SharedDiscreteHazardMethod requires observed person-time event labels.")

        X_train_t = torch.as_tensor(np.asarray(X_train, dtype=np.float32), dtype=torch.float32, device=self.device_)
        labels_train_t = torch.as_tensor(train_labels_np, dtype=torch.float32, device=self.device_)
        known_train_t = torch.as_tensor(train_known_np, dtype=torch.bool, device=self.device_)

        X_val_t = None
        labels_val_t = None
        known_val_t = None
        if X_val is not None and time_val is not None and event_val is not None:
            val_known_np, val_labels_np = interval_label_matrix(
                time=np.asarray(time_val, dtype=np.float64),
                event=np.asarray(event_val, dtype=np.int32),
                time_bins=self.time_bins_,
            )
            if int(val_known_np.sum()) > 0 and int(val_labels_np.sum()) > 0:
                X_val_t = torch.as_tensor(np.asarray(X_val, dtype=np.float32), dtype=torch.float32, device=self.device_)
                labels_val_t = torch.as_tensor(val_labels_np, dtype=torch.float32, device=self.device_)
                known_val_t = torch.as_tensor(val_known_np, dtype=torch.bool, device=self.device_)

        self.model_ = self._build_network(X_train_t.shape[1], len(self.time_bins_)).to(self.device_)
        optimizer_cls = torch.optim.AdamW if str(self.params["optimizer"]).lower() == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(
            self.model_.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        batch_size = batch_norm_safe_batch_size(
            len(X_train_t),
            int(self.params["batch_size"]),
            batch_norm=bool(self.params["batch_norm"]),
        )
        best_state = deepcopy(self.model_.state_dict())
        best_score = float("inf")
        stale_epochs = 0
        self.train_loss_trace_ = []
        self.val_loss_trace_ = []

        for _ in range(int(self.params["max_epochs"])):
            self.model_.train()
            optimizer.zero_grad(set_to_none=True)
            train_logits = self._forward_in_chunks(X_train_t, batch_size=batch_size)
            train_loss = self._masked_bce(train_logits, labels_train_t, known_train_t)
            train_loss.backward()
            optimizer.step()
            train_loss_value = float(train_loss.detach().cpu().item())
            self.train_loss_trace_.append(train_loss_value)

            with torch.no_grad():
                self.model_.eval()
                if X_val_t is not None and labels_val_t is not None and known_val_t is not None:
                    val_logits = self._forward_in_chunks(X_val_t, batch_size=batch_size)
                    monitor = float(self._masked_bce(val_logits, labels_val_t, known_val_t).detach().cpu().item())
                    self.val_loss_trace_.append(monitor)
                else:
                    monitor = train_loss_value

            if monitor + 1e-8 < best_score:
                best_score = monitor
                best_state = deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= int(self.params["patience"]):
                    break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def _hazards(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("SharedDiscreteHazardMethod must be fit before prediction.")
        X_t = torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            logits = self._forward_in_chunks(X_t, batch_size=int(self.params["batch_size"]))
            hazards = torch.sigmoid(logits).detach().cpu().numpy()
        return clean_hazards(hazards)

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return risk_from_hazards(self._hazards(X), aggregate_risk=str(self.params["aggregate_risk"]))

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.time_bins_ is None:
            raise RuntimeError("SharedDiscreteHazardMethod must be fit before prediction.")
        return survival_from_hazards(self._hazards(X), self.time_bins_, times)

    def predict_bundle(self, X: np.ndarray, times: np.ndarray) -> SurvivalPredictions:
        if self.time_bins_ is None:
            raise RuntimeError("SharedDiscreteHazardMethod must be fit before prediction.")
        hazards = self._hazards(X)
        return SurvivalPredictions(
            risk=risk_from_hazards(hazards, aggregate_risk=str(self.params["aggregate_risk"])),
            survival=survival_from_hazards(hazards, self.time_bins_, times),
        )

    def foundation_metadata(self) -> dict[str, Any]:
        return {
            "foundation_backbone": "PyTorch-MLP",
            "foundation_backbone_task": "shared_discrete_hazard_likelihood",
            "foundation_backbone_training": "supervised",
            "foundation_time_bin_count": 0 if self.time_bins_ is None else int(len(self.time_bins_)),
            "foundation_train_epochs": int(len(self.train_loss_trace_)),
            "foundation_best_train_loss": min(self.train_loss_trace_) if self.train_loss_trace_ else None,
        }
