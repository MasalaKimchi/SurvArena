from __future__ import annotations

import torch


def resolve_torch_training_device(raw_device: str) -> torch.device:
    normalized = str(raw_device).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "mps":
        raise ValueError(
            "MPS is not supported for SurvArena deep survival training. "
            "torchsurv Cox losses require aten::_logcumsumexp, which is unavailable on MPS in this environment; "
            "use device='cpu' on macOS or device='cuda' on CUDA-capable Linux."
        )
    return torch.device(normalized)


def batch_norm_safe_batch_size(n_samples: int, requested_batch_size: int, *, batch_norm: bool) -> int:
    """Return a batch size that avoids singleton training batches when batch norm is enabled."""
    n_samples = int(n_samples)
    requested_batch_size = int(requested_batch_size)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if requested_batch_size <= 0:
        raise ValueError("requested_batch_size must be positive.")

    batch_size = min(requested_batch_size, n_samples)
    if not batch_norm or n_samples <= 2 or n_samples % batch_size != 1:
        return batch_size

    if batch_size >= n_samples - 1:
        return n_samples

    for candidate in range(batch_size - 1, 1, -1):
        if n_samples % candidate != 1:
            return candidate
    for candidate in range(batch_size + 1, n_samples + 1):
        if n_samples % candidate != 1:
            return candidate
    return n_samples
