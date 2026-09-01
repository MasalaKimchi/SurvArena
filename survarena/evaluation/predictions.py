from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PredictionValidationError(ValueError):
    """A fail-closed prediction-contract violation with a stable reason code."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"{reason_code}: {message}")


@dataclass(frozen=True, slots=True)
class ValidatedPredictions:
    risk: np.ndarray
    survival: np.ndarray | None
    survival_times: np.ndarray | None

    @property
    def has_survival_distribution(self) -> bool:
        return self.survival is not None and self.survival_times is not None


def _reject(reason_code: str, message: str) -> None:
    raise PredictionValidationError(reason_code, message)


def validate_prediction_bundle(
    *,
    risk_scores: np.ndarray,
    survival_probs: np.ndarray | None,
    survival_times: np.ndarray | None,
    n_rows: int,
    require_survival: bool = True,
    survival_capable: bool = True,
    monotonicity_tolerance: float = 1e-8,
) -> ValidatedPredictions:
    """Validate model output without sorting, clipping, transposing, or imputing it.

    ``monotonicity_tolerance`` only absorbs floating-point noise in adjacent
    survival probabilities. The returned arrays are otherwise value-identical
    NumPy views/copies of the supplied output.
    """
    risk = np.asarray(risk_scores, dtype=float)
    if risk.ndim != 1:
        _reject("risk_dimensionality", f"expected a 1-D risk vector; received shape {risk.shape}")
    if risk.shape[0] != int(n_rows):
        _reject("risk_row_count", f"expected {int(n_rows)} risk rows; received {risk.shape[0]}")
    if not bool(np.isfinite(risk).all()):
        _reject("risk_nonfinite", "risk vector contains NaN or infinite values")

    if not survival_capable:
        if require_survival:
            _reject("survival_capability_mismatch", "full-distribution metrics require survival capability")
        return ValidatedPredictions(risk=risk, survival=None, survival_times=None)
    if survival_probs is None or survival_times is None:
        if require_survival:
            _reject("survival_capability_mismatch", "full-distribution metrics require probabilities and a time grid")
        return ValidatedPredictions(risk=risk, survival=None, survival_times=None)

    survival = np.asarray(survival_probs, dtype=float)
    grid = np.asarray(survival_times, dtype=float)
    if survival.ndim != 2:
        _reject("survival_dimensionality", f"expected an N-by-T matrix; received shape {survival.shape}")
    if survival.shape[0] != int(n_rows):
        _reject("survival_row_count", f"expected {int(n_rows)} survival rows; received {survival.shape[0]}")
    if grid.ndim != 1 or survival.shape[1] != grid.shape[0]:
        _reject(
            "survival_grid_length",
            f"survival columns ({survival.shape[1]}) must equal the 1-D time-grid length ({grid.size})",
        )
    if not bool(np.isfinite(grid).all()):
        _reject("survival_grid_nonfinite", "survival time grid contains NaN or infinite values")
    if grid.size == 0 or bool(np.any(np.diff(grid) <= 0.0)):
        _reject("survival_grid_not_strictly_increasing", "survival time grid must be unique and increasing")
    if not bool(np.isfinite(survival).all()):
        _reject("survival_nonfinite", "survival probabilities contain NaN or infinite values")
    if bool(np.any((survival < 0.0) | (survival > 1.0))):
        _reject("survival_out_of_bounds", "survival probabilities must remain within [0, 1]")
    if monotonicity_tolerance < 0.0:
        raise ValueError("monotonicity_tolerance must be non-negative.")
    if survival.shape[1] > 1 and bool(np.any(np.diff(survival, axis=1) > float(monotonicity_tolerance))):
        _reject(
            "survival_nonmonotone",
            f"survival probabilities increase by more than tolerance {float(monotonicity_tolerance):g}",
        )
    return ValidatedPredictions(risk=risk, survival=survival, survival_times=grid)
