from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchtuples as tt

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.survival_utils import risk_from_survival_frame, survival_frame_to_array


class _BasePyCoxMethod(BaseSurvivalMethod, ABC):
    model_kind: str = ""

    def __init__(
        self,
        *,
        hidden_layers: str | list[int] = "128-64",
        activation: str = "relu",
        dropout: float = 0.1,
        batch_norm: bool = True,
        optimizer: str = "adamw",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        max_epochs: int = 100,
        patience: int = 10,
        num_durations: int = 32,
        duration_scheme: str = "quantiles",
        alpha: float = 0.2,
        sigma: float = 0.1,
        shrink: float = 0.0,
        log_duration: bool = False,
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
            num_durations=num_durations,
            duration_scheme=duration_scheme,
            alpha=alpha,
            sigma=sigma,
            shrink=shrink,
            log_duration=log_duration,
            seed=seed,
            device=device,
        )
        self.model: Any | None = None
        self.device: torch.device | None = None

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

    def _build_optimizer(self) -> object:
        opt_name = str(self.params["optimizer"]).lower()
        lr = float(self.params["lr"])
        weight_decay = float(self.params["weight_decay"])
        if opt_name == "adamw":
            return tt.optim.AdamW(lr=lr, decoupled_weight_decay=weight_decay)
        return tt.optim.Adam(lr=lr, weight_decay=weight_decay)

    def _build_standard_mlp(self, *, in_features: int, out_features: int) -> nn.Module:
        hidden_layers = _parse_hidden_layers(self.params["hidden_layers"])
        activation = _activation_cls(str(self.params["activation"]))
        return tt.practical.MLPVanilla(
            in_features,
            hidden_layers,
            out_features,
            batch_norm=bool(self.params["batch_norm"]),
            dropout=float(self.params["dropout"]),
            activation=activation,
        )

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "_BasePyCoxMethod":
        if int(np.asarray(event_train, dtype=np.int32).sum()) <= 0:
            raise ValueError(f"{type(self).__name__} requires at least one observed event in the training data.")

        self.device = self._resolve_device()
        self._set_torch_seed(int(self.params["seed"]))
        X_train_f32 = np.asarray(X_train, dtype=np.float32)
        target_train = self._fit_label_transform(time_train=np.asarray(time_train), event_train=np.asarray(event_train))
        self.model = self._build_model(in_features=X_train_f32.shape[1])

        callbacks: list[object] = []
        val_data = None
        if X_val is not None and time_val is not None and event_val is not None:
            callbacks.append(tt.callbacks.EarlyStopping(patience=int(self.params["patience"])))
            X_val_f32 = np.asarray(X_val, dtype=np.float32)
            target_val = self._transform_targets(np.asarray(time_val), np.asarray(event_val))
            val_data = (X_val_f32, target_val)

        self.model.fit(
            X_train_f32,
            target_train,
            batch_size=min(int(self.params["batch_size"]), len(X_train_f32)),
            epochs=int(self.params["max_epochs"]),
            callbacks=callbacks if callbacks else None,
            verbose=False,
            val_data=val_data,
        )
        self._finalize_fit(X_train=X_train_f32, target_train=target_train)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        survival_frame = self._predict_survival_frame(X)
        return risk_from_survival_frame(survival_frame)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        survival_frame = self._predict_survival_frame(X)
        return survival_frame_to_array(survival_frame, times)

    def _fit_label_transform(self, *, time_train: np.ndarray, event_train: np.ndarray) -> Any:
        raise NotImplementedError

    def _transform_targets(self, time: np.ndarray, event: np.ndarray) -> Any:
        raise NotImplementedError

    def _build_model(self, *, in_features: int) -> Any:
        raise NotImplementedError

    def _finalize_fit(self, *, X_train: np.ndarray, target_train: Any) -> None:
        return None

    def _predict_survival_frame(self, X: np.ndarray) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError(f"{type(self).__name__} must be fit before prediction.")
        return self.model.predict_surv_df(np.asarray(X, dtype=np.float32))


def _parse_hidden_layers(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("-") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise ValueError(f"Unsupported hidden_layers value: {value!r}")


def _activation_cls(name: str) -> type[nn.Module]:
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'. Choices: {sorted(mapping.keys())}")
    return mapping[name]


class _BasePyCoxDiscreteMethod(_BasePyCoxMethod, ABC):
    model_cls: Any = None

    def _fit_label_transform(self, *, time_train: np.ndarray, event_train: np.ndarray) -> Any:
        self.labtrans_ = self.model_cls.label_transform(
            int(self.params["num_durations"]),
            scheme=str(self.params["duration_scheme"]),
        )
        return self.labtrans_.fit_transform(time_train.astype(np.float32), event_train.astype(np.float32))

    def _transform_targets(self, time: np.ndarray, event: np.ndarray) -> Any:
        return self.labtrans_.transform(time.astype(np.float32), event.astype(np.float32))

    def _build_model(self, *, in_features: int) -> Any:
        net = self._build_standard_mlp(in_features=in_features, out_features=int(self.labtrans_.out_features))
        kwargs: dict[str, Any] = {
            "optimizer": self._build_optimizer(),
            "device": self.device,
            "duration_index": self.labtrans_.cuts,
        }
        if self.model_cls.__name__ == "DeepHitSingle":
            kwargs["alpha"] = float(self.params["alpha"])
            kwargs["sigma"] = float(self.params["sigma"])
        return self.model_cls(net, **kwargs)


class LogisticHazardMethod(_BasePyCoxDiscreteMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def model_cls(self) -> Any:
        from pycox.models import LogisticHazard

        return LogisticHazard


class PMFMethod(_BasePyCoxDiscreteMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def model_cls(self) -> Any:
        from pycox.models import PMF

        return PMF


class MTLRMethod(_BasePyCoxDiscreteMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def model_cls(self) -> Any:
        from pycox.models import MTLR

        return MTLR


class DeepHitSingleMethod(_BasePyCoxDiscreteMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def model_cls(self) -> Any:
        from pycox.models import DeepHitSingle

        return DeepHitSingle


class PCHazardMethod(_BasePyCoxDiscreteMethod):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @property
    def model_cls(self) -> Any:
        from pycox.models import PCHazard

        return PCHazard


class CoxTimeMethod(_BasePyCoxMethod):
    def _fit_label_transform(self, *, time_train: np.ndarray, event_train: np.ndarray) -> Any:
        from pycox.models import CoxTime

        self.labtrans_ = CoxTime.label_transform(log_duration=bool(self.params["log_duration"]))
        return self.labtrans_.fit_transform(time_train.astype(np.float32), event_train.astype(np.float32))

    def _transform_targets(self, time: np.ndarray, event: np.ndarray) -> Any:
        return self.labtrans_.transform(time.astype(np.float32), event.astype(np.float32))

    def _build_model(self, *, in_features: int) -> Any:
        from pycox.models import CoxTime
        from pycox.models.cox_time import MLPVanillaCoxTime

        hidden_layers = _parse_hidden_layers(self.params["hidden_layers"])
        activation = _activation_cls(str(self.params["activation"]))
        net = MLPVanillaCoxTime(
            in_features,
            hidden_layers,
            batch_norm=bool(self.params["batch_norm"]),
            dropout=float(self.params["dropout"]),
            activation=activation,
        )
        return CoxTime(
            net,
            optimizer=self._build_optimizer(),
            device=self.device,
            shrink=float(self.params["shrink"]),
            labtrans=self.labtrans_,
        )

    def _finalize_fit(self, *, X_train: np.ndarray, target_train: Any) -> None:
        durations, _ = target_train
        order = np.argsort(np.asarray(durations, dtype=np.float32))
        X_sorted = np.asarray(X_train, dtype=np.float32)[order]
        target_sorted = tuple(np.asarray(values)[order] for values in target_train)
        self.model.compute_baseline_hazards(X_sorted, target_sorted)
