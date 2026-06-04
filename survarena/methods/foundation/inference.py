from __future__ import annotations

import gc
from typing import Any

import numpy as np


def positive_class_probability_with_backoff(
    model: Any,
    X: np.ndarray,
    *,
    batch_size: int | None,
) -> np.ndarray:
    X_np = np.asarray(X, dtype=np.float32)
    if X_np.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    cached_batch_size = getattr(model, "_survarena_safe_predict_batch_size", None)
    configured_batch_size = None if batch_size is None else max(1, int(batch_size))
    resolved_batch_size = min(
        X_np.shape[0],
        int(cached_batch_size or configured_batch_size or X_np.shape[0]),
    )

    while True:
        try:
            chunks: list[np.ndarray] = []
            for start in range(0, X_np.shape[0], resolved_batch_size):
                probabilities = np.asarray(model.predict_proba(X_np[start : start + resolved_batch_size]), dtype=np.float64)
                if probabilities.ndim == 1:
                    chunks.append(probabilities.reshape(-1))
                    continue
                classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))
                positive_positions = np.flatnonzero(classes.astype(str) == "1")
                positive_position = int(positive_positions[-1]) if positive_positions.size else -1
                chunks.append(probabilities[:, positive_position])
            return np.concatenate(chunks)
        except Exception as exc:
            if not _is_out_of_memory_error(exc) or resolved_batch_size <= 1:
                raise
            resolved_batch_size = max(1, resolved_batch_size // 2)
            setattr(model, "_survarena_safe_predict_batch_size", resolved_batch_size)
            _clear_accelerator_memory()


def _is_out_of_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory",
            "cannot allocate memory",
            "allocation failed",
            "mps backend out of memory",
        )
    )


def _clear_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        return
