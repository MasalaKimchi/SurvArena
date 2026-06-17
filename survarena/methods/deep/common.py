from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from survarena.methods.survival_utils import predict_breslow_survival


def parse_hidden_layers(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("-") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    raise ValueError(f"Unsupported hidden_layers value: {value!r}")


def activation_cls(name: str) -> type[nn.Module]:
    mapping: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "gelu": nn.GELU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation '{name}'. Choices: {sorted(mapping.keys())}")
    return mapping[name]


def set_torch_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_mlp(
    *,
    in_features: int,
    out_features: int,
    hidden_layers: Any,
    activation: str,
    dropout: float,
    batch_norm: bool,
    output_bias: bool = True,
) -> nn.Module:
    layers: list[nn.Module] = []
    previous = int(in_features)
    activation_type = activation_cls(str(activation))
    for width in parse_hidden_layers(hidden_layers):
        layers.append(nn.Linear(previous, int(width)))
        if batch_norm:
            layers.append(nn.BatchNorm1d(int(width)))
        layers.append(activation_type())
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        previous = int(width)
    layers.append(nn.Linear(previous, int(out_features), bias=output_bias))
    return nn.Sequential(*layers)


def predict_log_risk_survival(
    *,
    risk_scores: np.ndarray,
    times: np.ndarray,
    baseline_event_times: np.ndarray,
    baseline_survival: np.ndarray,
) -> np.ndarray:
    return predict_breslow_survival(
        risk_scores=np.asarray(risk_scores, dtype=np.float64),
        times=np.asarray(times, dtype=np.float64),
        baseline_event_times=np.asarray(baseline_event_times, dtype=np.float64),
        baseline_survival=np.asarray(baseline_survival, dtype=np.float64),
    )
