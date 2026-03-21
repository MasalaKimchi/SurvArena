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

    def to_dict(self) -> dict[str, float]:
        return {
            "uno_c": float(self.uno_c),
            "harrell_c": float(self.harrell_c),
            "ibs": float(self.ibs),
            "td_auc_25": float(self.td_auc_25),
            "td_auc_50": float(self.td_auc_50),
            "td_auc_75": float(self.td_auc_75),
        }


def _safe_float(value: float | np.ndarray) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        return float(value.item() if value.size == 1 else np.mean(value))
    return float(value)


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

    return MetricBundle(
        uno_c=_safe_float(uno),
        harrell_c=_safe_float(harrell),
        ibs=_safe_float(ibs),
        td_auc_25=_safe_float(aucs[0]),
        td_auc_50=_safe_float(aucs[1]),
        td_auc_75=_safe_float(aucs[2]),
    )


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
