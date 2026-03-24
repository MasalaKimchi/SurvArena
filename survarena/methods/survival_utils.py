from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, ndtr


def fit_breslow_baseline_survival(
    *,
    time_train: np.ndarray,
    event_train: np.ndarray,
    train_risk_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    event_mask = event_train.astype(bool)
    event_times = np.unique(time_train[event_mask])
    if event_times.size == 0:
        return np.asarray([1.0], dtype=np.float64), np.asarray([1.0], dtype=np.float64)

    exp_risk = np.exp(train_risk_scores.astype(np.float64))
    hazards: list[float] = []
    for event_time in event_times:
        d_j = float(np.sum((time_train == event_time) & event_mask))
        r_j = float(np.sum(exp_risk[time_train >= event_time]))
        hazards.append(d_j / max(r_j, 1e-12))

    cumulative_hazard = np.cumsum(np.asarray(hazards, dtype=np.float64))
    baseline_survival = np.exp(-cumulative_hazard)
    return event_times.astype(np.float64), baseline_survival.astype(np.float64)


def predict_breslow_survival(
    *,
    risk_scores: np.ndarray,
    times: np.ndarray,
    baseline_event_times: np.ndarray,
    baseline_survival: np.ndarray,
) -> np.ndarray:
    eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
    clipped_baseline = np.clip(np.asarray(baseline_survival, dtype=np.float64), 1e-8, 1.0)
    last_survival = float(clipped_baseline[-1]) if clipped_baseline.size else 1.0
    baseline_at_times = np.interp(
        eval_times,
        np.asarray(baseline_event_times, dtype=np.float64),
        clipped_baseline,
        left=1.0,
        right=last_survival,
    )
    survival = np.power(baseline_at_times[None, :], np.exp(np.asarray(risk_scores, dtype=np.float64))[:, None])
    return _clean_survival_array(survival, time_axis=1)


def survival_frame_to_array(survival_frame: pd.DataFrame, times: np.ndarray) -> np.ndarray:
    frame = _normalize_survival_frame(survival_frame)
    source_times = frame.index.to_numpy(dtype=np.float64)
    source_values = _clean_survival_array(frame.to_numpy(dtype=np.float64), time_axis=0)
    eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
    if source_times.size == 0:
        return np.ones((source_values.shape[1], eval_times.size), dtype=np.float64)

    last_values = source_values[-1]
    curves = np.vstack(
        [
            np.interp(eval_times, source_times, source_values[:, col], left=1.0, right=last_values[col])
            for col in range(source_values.shape[1])
        ]
    )
    return _clean_survival_array(curves, time_axis=1)


def risk_from_survival_frame(survival_frame: pd.DataFrame) -> np.ndarray:
    frame = _normalize_survival_frame(survival_frame)
    times = frame.index.to_numpy(dtype=np.float64)
    values = _clean_survival_array(frame.to_numpy(dtype=np.float64), time_axis=0)
    if values.shape[1] == 0:
        return np.asarray([], dtype=np.float64)
    if times.size <= 1:
        return (1.0 - values.reshape(times.size, -1).mean(axis=0)).astype(np.float64)
    area = np.trapz(values, x=times, axis=0)
    return (-area).astype(np.float64)


def normalize_aft_distribution_name(distribution: str) -> str:
    normalized = str(distribution).strip().lower()
    if normalized not in {"normal", "logistic", "extreme"}:
        raise ValueError(
            f"Unsupported AFT distribution '{distribution}'. "
            "Expected one of: normal, logistic, extreme."
        )
    return normalized


def predict_aft_survival(
    *,
    location_scores: np.ndarray,
    times: np.ndarray,
    distribution: str,
    scale: float,
) -> np.ndarray:
    resolved_scale = float(scale)
    if resolved_scale <= 0.0:
        raise ValueError("AFT scale must be positive.")

    resolved_distribution = normalize_aft_distribution_name(distribution)
    eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
    safe_times = np.maximum(eval_times, 1e-12)
    z = (np.log(safe_times)[None, :] - np.asarray(location_scores, dtype=np.float64)[:, None]) / resolved_scale
    survival = 1.0 - _aft_cdf(z, distribution=resolved_distribution)
    if np.any(eval_times <= 0.0):
        survival[:, eval_times <= 0.0] = 1.0
    return _clean_survival_array(survival, time_axis=1)


def _normalize_survival_frame(survival_frame: pd.DataFrame) -> pd.DataFrame:
    if survival_frame.empty:
        return survival_frame.copy()
    frame = survival_frame.copy()
    frame.index = frame.index.astype(float)
    frame = frame.sort_index()
    if not frame.index.is_unique:
        frame = frame.groupby(level=0, sort=True).last()
    return frame


def _aft_cdf(values: np.ndarray, *, distribution: str) -> np.ndarray:
    if distribution == "normal":
        return ndtr(values)
    if distribution == "logistic":
        return expit(values)
    if distribution == "extreme":
        return 1.0 - np.exp(-np.exp(values))
    raise ValueError(f"Unsupported AFT distribution '{distribution}'.")


def _clean_survival_array(values: np.ndarray, *, time_axis: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    array = np.nan_to_num(array, nan=1.0, posinf=1.0, neginf=1e-8)
    array = np.clip(array, 1e-8, 1.0)
    return np.minimum.accumulate(array, axis=time_axis).astype(np.float64)
