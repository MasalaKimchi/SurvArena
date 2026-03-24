from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn


def _import_torch():
    import torch

    return torch


def _import_nn():
    from torch import nn

    return nn


def _import_cox_loss():
    from torchsurv.loss.cox import neg_partial_log_likelihood

    return neg_partial_log_likelihood


def parse_hidden_layers(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("-") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    raise ValueError(f"Unsupported hidden_layers value: {value!r}")


def activation_cls(name: str) -> type[Any]:
    nn = _import_nn()
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'. Choices: {sorted(mapping)}")
    return mapping[name]


def resolve_device(raw_device: str) -> torch.device:
    torch = _import_torch()
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def set_torch_seed(seed: int) -> None:
    torch = _import_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_mlp_head(
    *,
    in_features: int,
    hidden_layers: Any,
    activation: str,
    dropout: float,
) -> nn.Module:
    nn = _import_nn()
    hidden_layer_sizes = parse_hidden_layers(hidden_layers)
    activation_module = activation_cls(activation)

    layers: list[nn.Module] = []
    prev = int(in_features)
    for width in hidden_layer_sizes:
        width_i = int(width)
        layers.append(nn.Linear(prev, width_i))
        layers.append(activation_module())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = width_i
    layers.append(nn.Linear(prev, 1, bias=False))
    return nn.Sequential(*layers)


def forward_in_chunks(head: nn.Module, X: torch.Tensor, batch_size: int) -> torch.Tensor:
    torch = _import_torch()
    outputs: list[torch.Tensor] = []
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        outputs.append(head(X[start:end]).squeeze(-1))
    return torch.cat(outputs, dim=0)


def fit_baseline_survival(
    *,
    time_train: np.ndarray,
    event_train: np.ndarray,
    train_log_risk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    return event_times.astype(np.float64), np.exp(-cumulative_hazard).astype(np.float64)


@dataclass(frozen=True, slots=True)
class NeuralCoxArtifacts:
    head: Any
    device: Any
    input_dim: int
    baseline_event_times: np.ndarray
    baseline_survival: np.ndarray


def train_neural_cox_head(
    *,
    train_features: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    hidden_layers: Any,
    activation: str,
    dropout: float,
    optimizer: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
    seed: int | None = None,
    val_features: np.ndarray | None = None,
    time_val: np.ndarray | None = None,
    event_val: np.ndarray | None = None,
) -> NeuralCoxArtifacts:
    torch = _import_torch()
    neg_partial_log_likelihood = _import_cox_loss()
    train_features_np = np.asarray(train_features, dtype=np.float32)
    if train_features_np.ndim != 2:
        raise ValueError(f"Neural Cox head expects 2D train features, received {train_features_np.shape}.")

    time_train_np = np.asarray(time_train, dtype=np.float64)
    event_train_np = np.asarray(event_train, dtype=np.int32)
    if int(event_train_np.sum()) <= 0:
        raise ValueError("Neural Cox head requires at least one observed event in the training data.")

    device_obj = resolve_device(str(device))
    set_torch_seed(int(seed or 0))

    train_features_t = torch.as_tensor(train_features_np, dtype=torch.float32, device=device_obj)
    train_time_t = torch.as_tensor(time_train_np.astype(np.float32), dtype=torch.float32, device=device_obj)
    train_event_t = torch.as_tensor(event_train_np.astype(bool), dtype=torch.bool, device=device_obj)

    val_features_t = None
    val_time_t = None
    val_event_t = None
    if val_features is not None and time_val is not None and event_val is not None:
        val_features_np = np.asarray(val_features, dtype=np.float32)
        if val_features_np.ndim != 2:
            raise ValueError(f"Neural Cox head expects 2D validation features, received {val_features_np.shape}.")
        val_features_t = torch.as_tensor(val_features_np, dtype=torch.float32, device=device_obj)
        val_time_t = torch.as_tensor(np.asarray(time_val, dtype=np.float32), dtype=torch.float32, device=device_obj)
        val_event_t = torch.as_tensor(np.asarray(event_val).astype(bool), dtype=torch.bool, device=device_obj)

    head = build_mlp_head(
        in_features=int(train_features_np.shape[1]),
        hidden_layers=hidden_layers,
        activation=str(activation),
        dropout=float(dropout),
    ).to(device_obj)
    optimizer_name = str(optimizer).lower()
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer_obj = optimizer_cls(
        head.parameters(),
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

    best_state = deepcopy(head.state_dict())
    best_loss = float("inf")
    stale_epochs = 0

    for _ in range(int(max_epochs)):
        head.train()
        optimizer_obj.zero_grad(set_to_none=True)
        train_log_risk = forward_in_chunks(head, train_features_t, batch_size=int(batch_size))
        train_loss = neg_partial_log_likelihood(train_log_risk, train_event_t, train_time_t)
        train_loss.backward()
        optimizer_obj.step()

        with torch.no_grad():
            if val_features_t is not None and val_time_t is not None and val_event_t is not None:
                head.eval()
                val_log_risk = forward_in_chunks(head, val_features_t, batch_size=int(batch_size))
                monitor = float(neg_partial_log_likelihood(val_log_risk, val_event_t, val_time_t).item())
            else:
                monitor = float(train_loss.item())

        if monitor + 1e-8 < best_loss:
            best_loss = monitor
            best_state = deepcopy(head.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        train_log_risk_np = forward_in_chunks(head, train_features_t, batch_size=int(batch_size)).detach().cpu().numpy()

    baseline_event_times, baseline_survival = fit_baseline_survival(
        time_train=time_train_np,
        event_train=event_train_np,
        train_log_risk=train_log_risk_np,
    )
    return NeuralCoxArtifacts(
        head=head,
        device=device_obj,
        input_dim=int(train_features_np.shape[1]),
        baseline_event_times=baseline_event_times,
        baseline_survival=baseline_survival,
    )
