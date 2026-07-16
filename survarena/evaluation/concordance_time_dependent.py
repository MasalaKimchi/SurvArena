"""Transparent concordance estimators for static and time-varying survival predictions."""
from __future__ import annotations
from typing import Optional
import numpy as np


def _at_time(survival: np.ndarray, grid: np.ndarray, time: float) -> np.ndarray:
    index = int(np.clip(np.searchsorted(grid, time, side="right") - 1, 0, len(grid) - 1))
    return survival[:, index]


def harrell_c_index(time, event, risk, *, tau: Optional[float] = None) -> float:
    """Observed comparable-pair concordance for a static risk score."""
    time, event, risk = np.asarray(time, float), np.asarray(event, bool), np.asarray(risk, float)
    numerator = denominator = 0.0
    for i in np.flatnonzero(event):
        if tau is not None and time[i] > tau:
            continue
        js = np.flatnonzero(time > time[i])
        if not js.size:
            continue
        numerator += np.sum(risk[i] > risk[js]) + 0.5 * np.sum(np.isclose(risk[i], risk[js]))
        denominator += js.size
    return float(numerator / denominator) if denominator else float("nan")


def antolini_c_index(time, event, survival, grid, *, tau: Optional[float] = None) -> float:
    """Antolini concordance, comparing survival curves at the earlier event time."""
    time, event = np.asarray(time, float), np.asarray(event, bool)
    survival, grid = np.asarray(survival, float), np.asarray(grid, float)
    numerator = denominator = 0.0
    for i in np.flatnonzero(event):
        if tau is not None and time[i] > tau:
            continue
        js = np.flatnonzero(time > time[i])
        if not js.size:
            continue
        values = _at_time(survival, grid, time[i])
        numerator += np.sum(values[i] < values[js]) + 0.5 * np.sum(np.isclose(values[i], values[js]))
        denominator += js.size
    return float(numerator / denominator) if denominator else float("nan")


def time_dependent_uno_c_index(
    time,
    event,
    survival,
    grid,
    *,
    censor_survival,
    tau: Optional[float] = None,
) -> float:
    """Marginal-IPCW dynamic concordance for subject-specific survival curves.

    ``censor_survival[i]`` estimates G(T_i-) from training data. Event-subject
    weights are 1/G(T_i-)^2.
    """
    time, event = np.asarray(time, float), np.asarray(event, bool)
    survival, grid = np.asarray(survival, float), np.asarray(grid, float)
    censor_survival = np.clip(np.asarray(censor_survival, float), 1e-6, 1.0)
    numerator = denominator = 0.0
    for i in np.flatnonzero(event):
        if tau is not None and time[i] > tau:
            continue
        js = np.flatnonzero(time > time[i])
        if not js.size:
            continue
        values = _at_time(survival, grid, time[i])
        weight = 1.0 / censor_survival[i] ** 2
        numerator += weight * (
            np.sum(values[i] < values[js]) + 0.5 * np.sum(np.isclose(values[i], values[js]))
        )
        denominator += weight * js.size
    return float(numerator / denominator) if denominator else float("nan")


def conditional_time_dependent_uno_c_index(
    time,
    event,
    survival,
    grid,
    *,
    conditional_censor_survival,
    tau: Optional[float] = None,
) -> float:
    """Pairwise conditional-IPCW dynamic concordance.

    Matrix entry [i,j] estimates G(T_i- | X_j). Pair weight is
    1/[G(T_i-|X_i)G(T_i-|X_j)]. Cross-fit the censoring model.
    """
    time, event = np.asarray(time, float), np.asarray(event, bool)
    survival, grid = np.asarray(survival, float), np.asarray(grid, float)
    conditional = np.clip(np.asarray(conditional_censor_survival, float), 1e-6, 1.0)
    if conditional.shape != (len(time), len(time)):
        raise ValueError("conditional_censor_survival must have shape [subjects, subjects]")
    numerator = denominator = 0.0
    for i in np.flatnonzero(event):
        if tau is not None and time[i] > tau:
            continue
        js = np.flatnonzero(time > time[i])
        if not js.size:
            continue
        values = _at_time(survival, grid, time[i])
        weights = 1.0 / (conditional[i, i] * conditional[i, js])
        comparisons = (values[i] < values[js]).astype(float) + 0.5 * np.isclose(values[i], values[js])
        numerator += float(np.sum(weights * comparisons))
        denominator += float(np.sum(weights))
    return float(numerator / denominator) if denominator else float("nan")
