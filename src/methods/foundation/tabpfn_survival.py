from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn
from torchsurv.loss.cox import neg_partial_log_likelihood

from src.methods.base import BaseSurvivalMethod


def _parse_hidden_layers(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("-") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    raise ValueError(f"Unsupported hidden_layers value: {value!r}")


def _activation_cls(name: str) -> type[nn.Module]:
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'. Choices: {sorted(mapping)}")
    return mapping[name]


class TabPFNSurvivalMethod(BaseSurvivalMethod):
    def __init__(
        self,
        n_estimators: int = 8,
        fit_mode: str = "fit_preprocessors",
        model_version: str = "auto",
        checkpoint_path: str | None = None,
        backbone_task: str = "classification_event",
        fine_tune: bool = False,
        aggregate_estimators: str = "mean",
        hidden_layers: str | list[int] = "64",
        activation: str = "relu",
        dropout: float = 0.1,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 200,
        patience: int = 20,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            fit_mode=fit_mode,
            model_version=model_version,
            checkpoint_path=checkpoint_path,
            backbone_task=backbone_task,
            fine_tune=fine_tune,
            aggregate_estimators=aggregate_estimators,
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
        self.head: nn.Module | None = None
        self.device_: torch.device | None = None
        self.embedding_dim_: int | None = None
        self.head_input_dim_: int | None = None
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

    def _build_backbone(self) -> Any:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion

        if bool(self.params["fine_tune"]):
            raise NotImplementedError(
                "TabPFN backbone fine-tuning is not exposed by the installed tabpfn package. "
                "This adapter supports a trainable MLP survival head on frozen TabPFN embeddings."
            )

        backbone_task = str(self.params["backbone_task"]).lower()
        if backbone_task == "classification_event":
            backbone_cls = TabPFNClassifier
        elif backbone_task == "regression_time":
            backbone_cls = TabPFNRegressor
        else:
            raise ValueError("backbone_task must be one of {'classification_event', 'regression_time'}.")

        base_kwargs = {
            "n_estimators": int(self.params["n_estimators"]),
            "fit_mode": str(self.params["fit_mode"]),
            "device": str(self.params["device"]),
            "random_state": self.params.get("seed"),
        }
        checkpoint_path = self.params.get("checkpoint_path")
        if checkpoint_path:
            return backbone_cls(model_path=str(checkpoint_path), **base_kwargs)

        model_version = str(self.params["model_version"]).lower()
        if model_version in {"auto", "default"}:
            return backbone_cls(**base_kwargs)

        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
            "v2_5": ModelVersion.V2_5,
        }
        if model_version not in version_map:
            raise ValueError("model_version must be one of {'auto', 'v2', 'v2.5'}.")
        return backbone_cls.create_default_for_version(version=version_map[model_version], **base_kwargs)

    def _fit_backbone(self, X_train: np.ndarray, time_train: np.ndarray, event_train: np.ndarray) -> None:
        self.backbone = self._build_backbone()
        backbone_task = str(self.params["backbone_task"]).lower()
        if backbone_task == "classification_event":
            target = np.asarray(event_train, dtype=np.int32)
        else:
            target = np.log1p(np.asarray(time_train, dtype=np.float32))
        self.backbone.fit(np.asarray(X_train, dtype=np.float32), target)

    def _extract_embeddings(self, X: np.ndarray, *, data_source: str = "test") -> np.ndarray:
        if self.backbone is None:
            raise RuntimeError("Backbone must be fit before extracting embeddings.")
        embeddings = self.backbone.get_embeddings(np.asarray(X, dtype=np.float32), data_source=data_source)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = embeddings[:, None, :]
        if embeddings.ndim != 3:
            raise ValueError(f"Unexpected embedding shape from TabPFN backbone: {embeddings.shape}")

        if str(self.params["aggregate_estimators"]) == "concat":
            embeddings = embeddings.transpose(1, 0, 2).reshape(embeddings.shape[1], -1)
        else:
            embeddings = embeddings.mean(axis=0)
        return np.asarray(embeddings, dtype=np.float32)

    def _build_head(self, in_features: int) -> nn.Module:
        hidden_layers = _parse_hidden_layers(self.params["hidden_layers"])
        activation = _activation_cls(str(self.params["activation"]))
        dropout = float(self.params["dropout"])

        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden_layers:
            width_i = int(width)
            layers.append(nn.Linear(prev, width_i))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = width_i
        layers.append(nn.Linear(prev, 1, bias=False))
        return nn.Sequential(*layers)

    @staticmethod
    def _cox_loss(log_risk: torch.Tensor, event: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return neg_partial_log_likelihood(log_risk, event, time)

    def _forward_in_chunks(self, X: torch.Tensor, batch_size: int) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("Survival head is unavailable before fit().")
        outputs: list[torch.Tensor] = []
        for start in range(0, X.shape[0], batch_size):
            end = min(start + batch_size, X.shape[0])
            outputs.append(self.head(X[start:end]).squeeze(-1))
        return torch.cat(outputs, dim=0)

    def _fit_baseline_survival(
        self,
        time_train: np.ndarray,
        event_train: np.ndarray,
        train_log_risk: np.ndarray,
    ) -> None:
        event_mask = event_train.astype(bool)
        event_times = np.unique(time_train[event_mask])
        if event_times.size == 0:
            self.baseline_event_times_ = np.asarray([1.0], dtype=np.float64)
            self.baseline_survival_ = np.asarray([1.0], dtype=np.float64)
            return

        exp_risk = np.exp(train_log_risk.astype(np.float64))
        hazards: list[float] = []
        for event_time in event_times:
            d_j = float(np.sum((time_train == event_time) & event_mask))
            r_j = float(np.sum(exp_risk[time_train >= event_time]))
            hazards.append(d_j / max(r_j, 1e-12))

        cumulative_hazard = np.cumsum(np.asarray(hazards, dtype=np.float64))
        self.baseline_event_times_ = event_times.astype(np.float64)
        self.baseline_survival_ = np.exp(-cumulative_hazard).astype(np.float64)

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "TabPFNSurvivalMethod":
        self.device_ = self._resolve_device()
        self._set_torch_seed(int(self.params["seed"] or 0))

        X_train_np = np.asarray(X_train, dtype=np.float32)
        time_train_np = np.asarray(time_train, dtype=np.float64)
        event_train_np = np.asarray(event_train, dtype=np.int32)
        if int(event_train_np.sum()) <= 0:
            raise ValueError("TabPFN survival head requires at least one observed event in the training data.")

        self._fit_backbone(X_train_np, time_train_np, event_train_np)

        train_embeddings_np = self._extract_embeddings(X_train_np, data_source="train")
        self.embedding_dim_ = int(train_embeddings_np.shape[-1]) if train_embeddings_np.ndim == 2 else None
        self.head_input_dim_ = int(train_embeddings_np.shape[1])

        train_embeddings = torch.as_tensor(train_embeddings_np, dtype=torch.float32, device=self.device_)
        train_time_t = torch.as_tensor(time_train_np.astype(np.float32), dtype=torch.float32, device=self.device_)
        train_event_t = torch.as_tensor(event_train_np.astype(bool), dtype=torch.bool, device=self.device_)

        val_embeddings = None
        val_time_t = None
        val_event_t = None
        if X_val is not None and time_val is not None and event_val is not None:
            val_embeddings_np = self._extract_embeddings(np.asarray(X_val, dtype=np.float32), data_source="test")
            val_embeddings = torch.as_tensor(val_embeddings_np, dtype=torch.float32, device=self.device_)
            val_time_t = torch.as_tensor(np.asarray(time_val, dtype=np.float32), dtype=torch.float32, device=self.device_)
            val_event_t = torch.as_tensor(np.asarray(event_val).astype(bool), dtype=torch.bool, device=self.device_)

        self.head = self._build_head(self.head_input_dim_).to(self.device_)
        opt_name = str(self.params["optimizer"]).lower()
        optimizer_cls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
        optimizer = optimizer_cls(
            self.head.parameters(),
            lr=float(self.params["lr"]),
            weight_decay=float(self.params["weight_decay"]),
        )

        batch_size = int(self.params["batch_size"])
        max_epochs = int(self.params["max_epochs"])
        patience = int(self.params["patience"])

        best_state = deepcopy(self.head.state_dict())
        best_loss = float("inf")
        stale_epochs = 0

        for _ in range(max_epochs):
            self.head.train()
            optimizer.zero_grad(set_to_none=True)
            train_log_risk = self._forward_in_chunks(train_embeddings, batch_size=batch_size)
            train_loss = self._cox_loss(train_log_risk, train_event_t, train_time_t)
            train_loss.backward()
            optimizer.step()

            with torch.no_grad():
                if val_embeddings is not None and val_time_t is not None and val_event_t is not None:
                    self.head.eval()
                    val_log_risk = self._forward_in_chunks(val_embeddings, batch_size=batch_size)
                    monitor = float(self._cox_loss(val_log_risk, val_event_t, val_time_t).item())
                else:
                    monitor = float(train_loss.item())

            if monitor + 1e-8 < best_loss:
                best_loss = monitor
                best_state = deepcopy(self.head.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    break

        self.head.load_state_dict(best_state)
        self.head.eval()
        with torch.no_grad():
            train_log_risk = self._forward_in_chunks(train_embeddings, batch_size=batch_size).detach().cpu().numpy()
        self._fit_baseline_survival(time_train_np, event_train_np, train_log_risk)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        if self.head is None or self.device_ is None:
            raise RuntimeError("TabPFNSurvivalMethod must be fit before prediction.")
        embeddings_np = self._extract_embeddings(np.asarray(X, dtype=np.float32), data_source="test")
        embeddings = torch.as_tensor(embeddings_np, dtype=torch.float32, device=self.device_)
        self.head.eval()
        with torch.no_grad():
            risk = self._forward_in_chunks(embeddings, batch_size=int(self.params["batch_size"])).detach().cpu().numpy()
        return risk.astype(np.float64)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.baseline_event_times_ is None or self.baseline_survival_ is None:
            raise RuntimeError("TabPFNSurvivalMethod must be fit before prediction.")
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
