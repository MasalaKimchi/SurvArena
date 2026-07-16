"""Reference-Scaled Bregman Survival (RSBS) losses.

Architecture-agnostic PyTorch losses for networks that output a positive
piecewise-constant hazard. The reference intensity must be training-only and
fixed with respect to the optimized network parameters.
"""
from __future__ import annotations

from typing import Literal
import torch
from torch import Tensor

Reduction = Literal["none", "mean", "sum"]


def _reduce(per_subject: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return per_subject
    if reduction == "mean":
        return per_subject.mean()
    if reduction == "sum":
        return per_subject.sum()
    raise ValueError(f"Unknown reduction: {reduction!r}")


def _validate(hazard: Tensor, exposure: Tensor, events: Tensor, reference: Tensor) -> Tensor:
    if hazard.shape != exposure.shape or hazard.shape != events.shape:
        raise ValueError("hazard, exposure, and events must have identical shapes")
    if hazard.ndim < 2:
        raise ValueError("Expected [batch, time] tensors")
    if torch.any(exposure < 0) or torch.any(events < 0):
        raise ValueError("Exposure and events must be non-negative")
    ref = torch.as_tensor(reference, dtype=hazard.dtype, device=hazard.device)
    try:
        return torch.broadcast_to(ref, hazard.shape)
    except RuntimeError as exc:
        raise ValueError("reference is not broadcastable to hazard") from exc


def censored_log_intensity_score(
    hazard: Tensor,
    exposure: Tensor,
    events: Tensor,
    *,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
) -> Tensor:
    """Piecewise-event-intensity negative log-likelihood."""
    if hazard.shape != exposure.shape or hazard.shape != events.shape:
        raise ValueError("hazard, exposure, and events must have identical shapes")
    q = hazard.clamp_min(eps)
    return _reduce((exposure * q - events * torch.log(q)).sum(dim=-1), reduction)


def reference_scaled_bregman_score(
    hazard: Tensor,
    exposure: Tensor,
    events: Tensor,
    reference: Tensor,
    *,
    beta: float = 2.0,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Reference-scaled beta-Bregman intensity score.

    For beta != 1:

      sum_k E_k a_k (q_k/a_k)^beta / beta
          - D_k (q_k/a_k)^(beta-1)/(beta-1).

    beta=1 is the logarithmic score up to a candidate-independent constant.
    beta=2 gives sum E q^2/(2a) - D q/a.
    """
    ref = _validate(hazard, exposure, events, reference).clamp_min(eps)
    q = hazard.clamp_min(eps)
    if abs(float(beta) - 1.0) < 1e-8:
        per_subject = (exposure * q - events * torch.log(q / ref)).sum(dim=-1)
        return _reduce(per_subject, reduction)
    if beta <= 0:
        raise ValueError("beta must be positive")
    ratio = q / ref
    score = exposure * ref * ratio.pow(beta) / beta
    score = score - events * ratio.pow(beta - 1.0) / (beta - 1.0)
    return _reduce(score.sum(dim=-1), reduction)


def rsbs_loss(
    hazard: Tensor,
    exposure: Tensor,
    events: Tensor,
    reference: Tensor,
    *,
    eta: float = 0.1,
    beta: float = 2.0,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
) -> Tensor:
    r"""Likelihood-anchored RSBS loss.

      L_RSBS = L_log + eta L_beta,a, eta >= 0.

    A non-negative mixture with the strictly proper log-intensity score remains
    strictly proper on the observable support. The reference is detached
    defensively; estimate it from training data or by cross-fitting.
    """
    if eta < 0:
        raise ValueError("eta must be non-negative")
    ref = torch.as_tensor(reference, dtype=hazard.dtype, device=hazard.device).detach()
    log_part = censored_log_intensity_score(hazard, exposure, events, eps=eps, reduction="none")
    beta_part = reference_scaled_bregman_score(
        hazard, exposure, events, ref, beta=beta, eps=eps, reduction="none"
    )
    return _reduce(log_part + float(eta) * beta_part, reduction)


def hazard_from_logits(logits: Tensor, *, link: Literal["softplus", "exp"] = "softplus") -> Tensor:
    """Map unconstrained network outputs to positive hazards."""
    if link == "softplus":
        return torch.nn.functional.softplus(logits) + torch.finfo(logits.dtype).eps
    if link == "exp":
        return torch.exp(logits)
    raise ValueError(f"Unknown link: {link!r}")


def kmd_u_statistic(moment_residuals: Tensor) -> Tensor:
    r"""Unbiased off-diagonal estimate of 0.5 ||E[xi]||^2.

    This can be negative in finite samples. It is primarily a held-out model
    specification diagnostic until its covariance and scale are stabilized.
    """
    if moment_residuals.ndim < 2 or moment_residuals.shape[0] < 2:
        raise ValueError("Expected at least two subjects and one moment dimension")
    flat = moment_residuals.reshape(moment_residuals.shape[0], -1)
    n = flat.shape[0]
    total = flat.sum(dim=0)
    return 0.5 * (total.square().sum() - flat.square().sum()) / (n * (n - 1))


__all__ = [
    "censored_log_intensity_score",
    "reference_scaled_bregman_score",
    "rsbs_loss",
    "hazard_from_logits",
    "kmd_u_statistic",
]
