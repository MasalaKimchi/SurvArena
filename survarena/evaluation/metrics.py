from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(slots=True)
class MetricBundle:
    uno_c: float
    harrell_c: float
    ibs: float
    td_auc_25: float
    td_auc_50: float
    td_auc_75: float
    brier_25: float = float("nan")
    brier_50: float = float("nan")
    brier_75: float = float("nan")
    calibration_slope_50: float = float("nan")
    calibration_intercept_50: float = float("nan")
    net_benefit_50: float = float("nan")

    def to_dict(self) -> dict[str, float]:
        return {
            "uno_c": float(self.uno_c),
            "harrell_c": float(self.harrell_c),
            "ibs": float(self.ibs),
            "td_auc_25": float(self.td_auc_25),
            "td_auc_50": float(self.td_auc_50),
            "td_auc_75": float(self.td_auc_75),
            "brier_25": float(self.brier_25),
            "brier_50": float(self.brier_50),
            "brier_75": float(self.brier_75),
            "calibration_slope_50": float(self.calibration_slope_50),
            "calibration_intercept_50": float(self.calibration_intercept_50),
            "net_benefit_50": float(self.net_benefit_50),
        }


def _safe_float(value: float | np.ndarray) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        return float(value.item() if value.size == 1 else np.mean(value))
    return float(value)


def _ipcw_estimable_mask(train_time: np.ndarray, train_event: np.ndarray, eval_time: np.ndarray) -> np.ndarray:
    observed_train_times = np.asarray(train_time)[np.asarray(train_event).astype(bool)]
    if observed_train_times.size == 0:
        return np.zeros_like(np.asarray(eval_time), dtype=bool)
    max_observed_time = float(np.max(observed_train_times))
    return np.asarray(eval_time, dtype=float) <= max_observed_time


def compute_survival_metrics(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk_scores: np.ndarray,
    survival_probs: np.ndarray,
    survival_times: np.ndarray,
    horizons: tuple[float, float, float],
) -> MetricBundle:
    import torch
    from torchsurv.metrics.auc import Auc
    from torchsurv.metrics.brier_score import BrierScore
    from torchsurv.metrics.cindex import ConcordanceIndex
    from torchsurv.stats.ipcw import get_ipcw

    estimable_mask = _ipcw_estimable_mask(train_time, train_event, test_time)
    if estimable_mask.sum() < 2:
        return MetricBundle(
            uno_c=float("nan"),
            harrell_c=compute_harrell_c_index(
                eval_time=test_time,
                eval_event=test_event,
                eval_risk_scores=risk_scores,
            ),
            ibs=float("nan"),
            td_auc_25=float("nan"),
            td_auc_50=float("nan"),
            td_auc_75=float("nan"),
        )

    test_time = np.asarray(test_time)[estimable_mask]
    test_event = np.asarray(test_event)[estimable_mask]
    risk_scores = np.asarray(risk_scores)[estimable_mask]
    survival_probs = np.asarray(survival_probs)[estimable_mask]

    train_event_t = torch.as_tensor(train_event.astype(bool))
    train_time_t = torch.as_tensor(train_time.astype(np.float32))
    test_event_t = torch.as_tensor(test_event.astype(bool))
    test_time_t = torch.as_tensor(test_time.astype(np.float32))
    risk_t = torch.as_tensor(risk_scores.astype(np.float32))
    survival_probs_t = torch.as_tensor(survival_probs.astype(np.float32))
    survival_times_t = torch.as_tensor(survival_times.astype(np.float32))
    horizons_t = torch.as_tensor(np.asarray(horizons, dtype=np.float32))

    ipcw_test = get_ipcw(train_event_t, train_time_t, test_time_t)
    ipcw_survival_times = get_ipcw(train_event_t, train_time_t, survival_times_t)
    ipcw_horizons = get_ipcw(train_event_t, train_time_t, horizons_t)

    cindex = ConcordanceIndex()
    uno = cindex(risk_t, test_event_t, test_time_t, weight=ipcw_test)
    harrell = cindex(risk_t, test_event_t, test_time_t)

    brier = BrierScore()
    _ = brier(
        survival_probs_t,
        test_event_t,
        test_time_t,
        new_time=survival_times_t,
        weight=ipcw_test,
        weight_new_time=ipcw_survival_times,
    )
    ibs = brier.integral()

    auc = Auc()
    aucs = auc(
        risk_t,
        test_event_t,
        test_time_t,
        auc_type="cumulative",
        new_time=horizons_t,
        weight=ipcw_test,
        weight_new_time=ipcw_horizons,
    )
    horizon_survival = _survival_at_times(survival_probs, survival_times, horizons)
    horizon_event_probs = 1.0 - horizon_survival
    horizon_survival_t = torch.as_tensor(horizon_survival.astype(np.float32))
    brier_at_horizons_t = BrierScore()(
        horizon_survival_t,
        test_event_t,
        test_time_t,
        new_time=horizons_t,
        weight=ipcw_test,
        weight_new_time=ipcw_horizons,
    )
    horizon_observed, horizon_known = _event_status_at_horizons(test_time, test_event, horizons)
    calibration_slope, calibration_intercept = _calibration_line(
        predicted=horizon_event_probs[:, 1],
        observed=horizon_observed[:, 1],
        known=horizon_known[:, 1],
    )
    net_benefit = _net_benefit(
        predicted=horizon_event_probs[:, 1],
        observed=horizon_observed[:, 1],
        known=horizon_known[:, 1],
        threshold=0.2,
    )

    return MetricBundle(
        uno_c=_safe_float(uno),
        harrell_c=_safe_float(harrell),
        ibs=_safe_float(ibs),
        td_auc_25=_safe_float(aucs[0]),
        td_auc_50=_safe_float(aucs[1]),
        td_auc_75=_safe_float(aucs[2]),
        brier_25=_safe_float(brier_at_horizons_t[0]),
        brier_50=_safe_float(brier_at_horizons_t[1]),
        brier_75=_safe_float(brier_at_horizons_t[2]),
        calibration_slope_50=calibration_slope,
        calibration_intercept_50=calibration_intercept,
        net_benefit_50=net_benefit,
    )


def _survival_at_times(
    survival_probs: np.ndarray,
    survival_times: np.ndarray,
    horizons: tuple[float, float, float],
) -> np.ndarray:
    probs = np.asarray(survival_probs, dtype=float)
    times = np.asarray(survival_times, dtype=float)
    if probs.ndim != 2 or times.size == 0:
        return np.full((len(probs), len(horizons)), np.nan, dtype=float)
    return np.vstack(
        [
            np.interp(np.asarray(horizons, dtype=float), times, row, left=1.0, right=float(row[-1]))
            for row in probs
        ]
    )


def _event_status_at_horizons(
    test_time: np.ndarray,
    test_event: np.ndarray,
    horizons: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(test_time, dtype=float)
    event = np.asarray(test_event, dtype=bool)
    observed_rows: list[np.ndarray] = []
    known_rows: list[np.ndarray] = []
    for horizon in horizons:
        case = (time <= horizon) & event
        control = time > horizon
        known = case | control
        observed_rows.append(case.astype(float))
        known_rows.append(known)
    return np.vstack(observed_rows).T, np.vstack(known_rows).T


def _calibration_line(*, predicted: np.ndarray, observed: np.ndarray, known: np.ndarray | None = None) -> tuple[float, float]:
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(obs)
    if known is not None:
        mask &= np.asarray(known, dtype=bool)
    if int(mask.sum()) < 2 or float(np.std(pred[mask])) == 0.0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(pred[mask], obs[mask], deg=1)
    return float(slope), float(intercept)


def _net_benefit(
    *,
    predicted: np.ndarray,
    observed: np.ndarray,
    threshold: float,
    known: np.ndarray | None = None,
) -> float:
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=bool)
    mask = np.isfinite(pred)
    if known is not None:
        mask &= np.asarray(known, dtype=bool)
    if not mask.any() or not 0.0 < threshold < 1.0:
        return float("nan")
    pred = pred[mask]
    obs = obs[mask]
    selected = pred >= threshold
    n = float(len(pred))
    true_positive = float(np.sum(selected & obs)) / n
    false_positive = float(np.sum(selected & ~obs)) / n
    return true_positive - false_positive * (threshold / (1.0 - threshold))


def compute_uno_c_index(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    eval_time: np.ndarray,
    eval_event: np.ndarray,
    eval_risk_scores: np.ndarray,
) -> float:
    import torch
    from torchsurv.metrics.cindex import ConcordanceIndex
    from torchsurv.stats.ipcw import get_ipcw

    estimable_mask = _ipcw_estimable_mask(train_time, train_event, eval_time)
    if estimable_mask.sum() < 2:
        return float("nan")
    eval_time = np.asarray(eval_time)[estimable_mask]
    eval_event = np.asarray(eval_event)[estimable_mask]
    eval_risk_scores = np.asarray(eval_risk_scores)[estimable_mask]

    train_event_t = torch.as_tensor(train_event.astype(bool))
    train_time_t = torch.as_tensor(train_time.astype(np.float32))
    eval_event_t = torch.as_tensor(eval_event.astype(bool))
    eval_time_t = torch.as_tensor(eval_time.astype(np.float32))
    eval_risk_t = torch.as_tensor(eval_risk_scores.astype(np.float32))

    ipcw_eval = get_ipcw(train_event_t, train_time_t, eval_time_t)
    uno = ConcordanceIndex()(eval_risk_t, eval_event_t, eval_time_t, weight=ipcw_eval)
    return _safe_float(uno)


def compute_harrell_c_index(
    *,
    eval_time: np.ndarray,
    eval_event: np.ndarray,
    eval_risk_scores: np.ndarray,
) -> float:
    import torch
    from torchsurv.metrics.cindex import ConcordanceIndex

    eval_event_t = torch.as_tensor(eval_event.astype(bool))
    eval_time_t = torch.as_tensor(eval_time.astype(np.float32))
    eval_risk_t = torch.as_tensor(eval_risk_scores.astype(np.float32))
    harrell = ConcordanceIndex()(eval_risk_t, eval_event_t, eval_time_t)
    return _safe_float(harrell)


def compute_primary_metric_score(
    *,
    primary_metric: str,
    train_time: np.ndarray,
    train_event: np.ndarray,
    eval_time: np.ndarray,
    eval_event: np.ndarray,
    eval_risk_scores: np.ndarray,
) -> float:
    if primary_metric == "harrell_c":
        return compute_harrell_c_index(
            eval_time=eval_time,
            eval_event=eval_event,
            eval_risk_scores=eval_risk_scores,
        )
    if primary_metric == "uno_c":
        return compute_uno_c_index(
            train_time=train_time,
            train_event=train_event,
            eval_time=eval_time,
            eval_event=eval_event,
            eval_risk_scores=eval_risk_scores,
        )
    raise ValueError(f"Unsupported primary metric for selection: {primary_metric}")


def horizons_from_train_event_times(
    time: np.ndarray,
    event: np.ndarray,
    quantiles: tuple[float, float, float] = (0.25, 0.5, 0.75),
) -> tuple[float, float, float]:
    event_times = time[event.astype(bool)]
    if event_times.size == 0:
        return (1.0, 2.0, 3.0)
    return tuple(float(np.quantile(event_times, q)) for q in quantiles)
