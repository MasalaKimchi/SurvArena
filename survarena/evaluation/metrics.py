from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from survarena.evaluation.predictions import validate_prediction_bundle


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
    net_benefit_50: float = float("nan")
    extra_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "uno_c": float(self.uno_c),
            "harrell_c": float(self.harrell_c),
            "ibs": float(self.ibs),
            "td_auc_25": float(self.td_auc_25),
            "td_auc_50": float(self.td_auc_50),
            "td_auc_75": float(self.td_auc_75),
            "brier_25": float(self.brier_25),
            "brier_50": float(self.brier_50),
            "brier_75": float(self.brier_75),
            "net_benefit_50": float(self.net_benefit_50),
        }
        for key, value in self.extra_metrics.items():
            payload[str(key)] = float(value)
        payload.update(self.metadata)
        return payload


def _safe_float(value: float | np.ndarray) -> float:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return float("nan")
        return float(value.item() if value.size == 1 else np.mean(value))
    return float(value)


def _ipcw_estimable_mask(train_time: np.ndarray, train_event: np.ndarray, eval_time: np.ndarray) -> np.ndarray:
    support_limit = _ipcw_support_limit(train_time, train_event)
    if support_limit is None:
        return np.zeros_like(np.asarray(eval_time), dtype=bool)
    return np.asarray(eval_time, dtype=float) <= support_limit


def _ipcw_support_limit(train_time: np.ndarray, train_event: np.ndarray) -> float | None:
    time = np.asarray(train_time, dtype=float)
    event = np.asarray(train_event, dtype=bool)
    if time.size == 0 or not bool(np.isfinite(time).all()) or not bool(event.any()):
        return None
    from sksurv.nonparametric import CensoringDistributionEstimator

    outcome = np.empty(time.size, dtype=[("event", "?"), ("time", "f8")])
    outcome["event"] = event
    outcome["time"] = time
    estimator = CensoringDistributionEstimator().fit(outcome)
    unique_time = np.asarray(estimator.unique_time_[1:], dtype=float)
    probabilities = np.asarray(estimator.prob_[1:], dtype=float)
    zero_indices = np.flatnonzero(probabilities <= 0.0)
    boundary = float(unique_time[zero_indices[0]]) if zero_indices.size else float(unique_time[-1])
    return max(1e-8, _strictly_below(boundary))


def _strictly_below(value: float) -> float:
    margin = max(1e-6, abs(float(value)) * 1e-6)
    return float(value) - margin


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
    decision_thresholds: tuple[float, ...] = (0.2,),
) -> MetricBundle:
    import torch
    from torchsurv.metrics.auc import Auc
    from torchsurv.metrics.brier_score import BrierScore
    from torchsurv.metrics.cindex import ConcordanceIndex
    from torchsurv.stats.ipcw import get_ipcw

    validated = validate_prediction_bundle(
        risk_scores=risk_scores,
        survival_probs=survival_probs,
        survival_times=survival_times,
        n_rows=len(test_time),
        require_survival=True,
    )
    assert validated.survival is not None
    assert validated.survival_times is not None
    risk_scores = validated.risk
    survival_probs = validated.survival
    survival_times = validated.survival_times

    estimable_mask = _ipcw_estimable_mask(train_time, train_event, test_time)
    harrell_full = compute_harrell_c_index(
        eval_time=np.asarray(test_time),
        eval_event=np.asarray(test_event),
        eval_risk_scores=risk_scores,
    )
    requested_horizons = tuple(float(horizon) for horizon in horizons)
    support_limit = _ipcw_support_limit(train_time, train_event)
    test_time_array = np.asarray(test_time, dtype=float)
    estimable_test_times = test_time_array[estimable_mask]
    support_lower = float(np.min(estimable_test_times)) if estimable_test_times.size else float("nan")
    support_upper = (
        min(float(support_limit), _strictly_below(float(np.max(estimable_test_times))))
        if support_limit is not None and estimable_test_times.size
        else float("nan")
    )
    support_metadata: dict[str, object] = {
        "metric_support_policy": "censoring_distribution_fixed_horizons_v1",
        "metric_support_lower": support_lower,
        "metric_support_upper": support_upper,
        "evaluation_window_lower": float("nan"),
        "evaluation_window_upper": float("nan"),
        "full_distribution_eligible": True,
    }
    labels = ("25", "50", "75")
    ipcw_supported = np.asarray(
        [
            np.isfinite(support_lower)
            and np.isfinite(support_upper)
            and support_lower <= horizon <= support_upper
            for horizon in requested_horizons
        ],
        dtype=bool,
    )
    grid_supported = np.asarray(
        [float(survival_times[0]) <= horizon <= float(survival_times[-1]) for horizon in requested_horizons],
        dtype=bool,
    )
    horizon_supported = ipcw_supported & grid_supported
    for index, (label, horizon, supported) in enumerate(
        zip(labels, requested_horizons, horizon_supported, strict=True)
    ):
        reason = ""
        if not ipcw_supported[index]:
            reason = "outside_ipcw_support"
        elif not grid_supported[index]:
            reason = "outside_prediction_grid"
        support_metadata[f"horizon_requested_{label}"] = horizon
        support_metadata[f"horizon_used_{label}"] = horizon if supported else float("nan")
        support_metadata[f"horizon_eligible_{label}"] = bool(supported)
        support_metadata[f"horizon_reason_{label}"] = reason

    if estimable_mask.sum() < 2:
        return MetricBundle(
            uno_c=float("nan"),
            harrell_c=harrell_full,
            ibs=float("nan"),
            td_auc_25=float("nan"),
            td_auc_50=float("nan"),
            td_auc_75=float("nan"),
            metadata=support_metadata,
        )

    test_time = np.asarray(test_time)[estimable_mask]
    test_event = np.asarray(test_event)[estimable_mask]
    risk_scores = np.asarray(risk_scores)[estimable_mask]
    survival_probs = np.asarray(survival_probs)[estimable_mask]

    supported_time_mask = (survival_times >= support_lower) & (survival_times <= support_upper)
    supported_survival_times = survival_times[supported_time_mask]
    supported_survival_probs = np.asarray(survival_probs)[:, supported_time_mask]

    train_event_t = torch.as_tensor(train_event.astype(bool))
    train_time_t = torch.as_tensor(train_time.astype(np.float32))
    test_event_t = torch.as_tensor(test_event.astype(bool))
    test_time_t = torch.as_tensor(test_time.astype(np.float32))
    risk_t = torch.as_tensor(risk_scores.astype(np.float32))

    ipcw_test = get_ipcw(train_event_t, train_time_t, test_time_t)

    cindex = ConcordanceIndex()
    uno = cindex(risk_t, test_event_t, test_time_t, weight=ipcw_test)

    ibs = float("nan")
    if supported_survival_times.size >= 2:
        survival_probs_t = torch.as_tensor(supported_survival_probs.astype(np.float32))
        survival_times_t = torch.as_tensor(supported_survival_times.astype(np.float32))
        ipcw_survival_times = get_ipcw(train_event_t, train_time_t, survival_times_t)
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

    aucs = np.full(3, np.nan, dtype=float)
    brier_at_horizons = np.full(3, np.nan, dtype=float)
    horizon_survival = np.full((len(test_time), 3), np.nan, dtype=float)
    horizon_observed = np.zeros((len(test_time), 3), dtype=float)
    horizon_known = np.zeros((len(test_time), 3), dtype=bool)
    horizon_weights = np.zeros((len(test_time), 3), dtype=float)
    if bool(horizon_supported.any()):
        supported_indices = np.flatnonzero(horizon_supported)
        supported_horizons = tuple(requested_horizons[index] for index in supported_indices)
        unique_horizons, horizon_inverse = _unique_horizons(supported_horizons)
        unique_horizons_t = torch.as_tensor(unique_horizons.astype(np.float32))
        ipcw_horizons = get_ipcw(train_event_t, train_time_t, unique_horizons_t)
        unique_survival = _survival_at_times(survival_probs, survival_times, tuple(unique_horizons.tolist()))
        supported_event_probs_t = torch.as_tensor((1.0 - unique_survival).astype(np.float32))
        supported_aucs = Auc()(
            supported_event_probs_t,
            test_event_t,
            test_time_t,
            auc_type="cumulative",
            new_time=unique_horizons_t,
            weight=ipcw_test,
            weight_new_time=ipcw_horizons,
        )
        supported_brier = BrierScore()(
            torch.as_tensor(unique_survival.astype(np.float32)),
            test_event_t,
            test_time_t,
            new_time=unique_horizons_t,
            weight=ipcw_test,
            weight_new_time=ipcw_horizons,
        )
        unique_observed, unique_known = _event_status_at_horizons(
            test_time,
            test_event,
            tuple(unique_horizons.tolist()),
        )
        unique_weights = _ipcw_weights_at_horizons(
            ipcw_at_time=ipcw_test.detach().cpu().numpy(),
            ipcw_at_horizons=ipcw_horizons.detach().cpu().numpy(),
            test_time=test_time,
            test_event=test_event,
            horizons=tuple(unique_horizons.tolist()),
        )
        for local_index, output_index in enumerate(supported_indices):
            unique_index = int(horizon_inverse[local_index])
            aucs[output_index] = _safe_float(supported_aucs[unique_index])
            brier_at_horizons[output_index] = _safe_float(supported_brier[unique_index])
            horizon_survival[:, output_index] = unique_survival[:, unique_index]
            horizon_observed[:, output_index] = unique_observed[:, unique_index]
            horizon_known[:, output_index] = unique_known[:, unique_index]
            horizon_weights[:, output_index] = unique_weights[:, unique_index]

    extra_metrics: dict[str, float] = {}
    horizon_event_probs = 1.0 - horizon_survival
    for idx, label in enumerate(labels):
        slope, intercept = _calibration_line(
            predicted=horizon_event_probs[:, idx],
            observed=horizon_observed[:, idx],
            known=horizon_known[:, idx],
            sample_weight=horizon_weights[:, idx],
        )
        extra_metrics[f"calibration_slope_abs_error_{label}"] = (
            float(abs(slope - 1.0)) if np.isfinite(slope) else float("nan")
        )
        extra_metrics[f"calibration_intercept_abs_error_{label}"] = (
            float(abs(intercept)) if np.isfinite(intercept) else float("nan")
        )
        threshold_scores: list[float] = []
        for threshold in decision_thresholds:
            nb = _net_benefit(
                predicted=horizon_event_probs[:, idx],
                observed=horizon_observed[:, idx],
                known=horizon_known[:, idx],
                sample_weight=horizon_weights[:, idx],
                threshold=float(threshold),
            )
            if np.isfinite(nb):
                threshold_scores.append(float(nb))
        # M13: The canonical `net_benefit_50` is the single-threshold net benefit
        # at the median horizon carried by the dataclass field (set below). The
        # threshold-averaged quantity is a distinct diagnostic, so for the "50"
        # horizon it is exported under `net_benefit_mean_thresholds_50` to avoid
        # colliding with (and silently overwriting) the field in ``to_dict``.
        averaged_key = "net_benefit_mean_thresholds_50" if label == "50" else f"net_benefit_{label}"
        extra_metrics[averaged_key] = float(np.mean(threshold_scores)) if threshold_scores else float("nan")

    net_benefit = _net_benefit(
        predicted=horizon_event_probs[:, 1],
        observed=horizon_observed[:, 1],
        known=horizon_known[:, 1],
        sample_weight=horizon_weights[:, 1],
        threshold=0.2,
    )
    d_calibration = _d_calibration(
        test_time=test_time,
        test_event=test_event,
        survival_probs=survival_probs,
        survival_times=survival_times,
    )
    extra_metrics.update(d_calibration)
    support_metadata["evaluation_window_lower"] = (
        float(supported_survival_times[0]) if supported_survival_times.size else float("nan")
    )
    support_metadata["evaluation_window_upper"] = (
        float(supported_survival_times[-1]) if supported_survival_times.size else float("nan")
    )

    return MetricBundle(
        uno_c=_safe_float(uno),
        harrell_c=harrell_full,
        ibs=_safe_float(ibs),
        td_auc_25=_safe_float(aucs[0]),
        td_auc_50=_safe_float(aucs[1]),
        td_auc_75=_safe_float(aucs[2]),
        brier_25=_safe_float(brier_at_horizons[0]),
        brier_50=_safe_float(brier_at_horizons[1]),
        brier_75=_safe_float(brier_at_horizons[2]),
        net_benefit_50=net_benefit,
        extra_metrics=extra_metrics,
        metadata=support_metadata,
    )


def _survival_at_times(
    survival_probs: np.ndarray,
    survival_times: np.ndarray,
    horizons: tuple[float, ...],
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


def _unique_horizons(horizons: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(horizons, dtype=float)
    unique_values, inverse = np.unique(values, return_inverse=True)
    return unique_values.astype(float), inverse.astype(int)


def _event_status_at_horizons(
    test_time: np.ndarray,
    test_event: np.ndarray,
    horizons: tuple[float, ...],
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


def _ipcw_weights_at_horizons(
    *,
    ipcw_at_time: np.ndarray,
    ipcw_at_horizons: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    horizons: tuple[float, ...],
) -> np.ndarray:
    time = np.asarray(test_time, dtype=float)
    event = np.asarray(test_event, dtype=bool)
    event_weights = np.asarray(ipcw_at_time, dtype=float)
    horizon_weights = np.asarray(ipcw_at_horizons, dtype=float)
    columns: list[np.ndarray] = []
    for idx, horizon in enumerate(horizons):
        case = (time <= float(horizon)) & event
        control = time > float(horizon)
        weights = np.zeros_like(time, dtype=float)
        weights[case] = event_weights[case]
        weights[control] = horizon_weights[idx]
        columns.append(weights)
    return np.vstack(columns).T


def _calibration_line(
    *,
    predicted: np.ndarray,
    observed: np.ndarray,
    known: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    """Standard calibration slope/intercept on the logit link scale.

    The observed binary outcome is regressed on the *linear predictor*
    ``logit(risk_hat)`` via a small, dependency-free weighted single-covariate
    logistic regression (Newton-Raphson / IRLS). This is the standard calibration
    slope: a perfectly calibrated model yields slope ~= 1 and intercept ~= 0
    (unlike an OLS fit of the 0/1 indicator on the raw probability). Predicted
    probabilities are clamped away from 0/1 before the logit transform so the
    linear predictor stays finite. Degenerate cases -- fewer than two eligible
    rows, all-events / all-censored outcomes, zero covariate variance, or
    non-convergence -- return NaN.
    """
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(obs)
    if known is not None:
        mask &= np.asarray(known, dtype=bool)
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        mask &= np.isfinite(weights) & (weights > 0.0)
    if int(mask.sum()) < 2:
        return float("nan"), float("nan")
    pred = pred[mask]
    obs_bin = (obs[mask] > 0.5).astype(float)
    w = np.ones_like(pred) if weights is None else weights[mask]

    # Transform predicted risk to the linear-predictor (logit) scale, clamping
    # away from 0/1 so the logit is finite.
    eps = 1e-6
    p_clamped = np.clip(pred, eps, 1.0 - eps)
    x = np.log(p_clamped / (1.0 - p_clamped))

    # Guard degenerate designs: no variation in the covariate, or an outcome that
    # is all-events / all-censored (a slope is then not identifiable).
    if float(np.std(x)) == 0.0:
        return float("nan"), float("nan")
    positives = float(np.sum(w * obs_bin))
    total_weight = float(np.sum(w))
    if positives <= 0.0 or positives >= total_weight:
        return float("nan"), float("nan")

    # Weighted logistic regression logit(P(Y=1)) = intercept + slope * x via IRLS.
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2, dtype=float)
    for _ in range(50):
        eta = np.clip(design @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        var = np.clip(mu * (1.0 - mu), 1e-9, None)
        gradient = design.T @ (w * (obs_bin - mu))
        hessian = design.T @ (design * (w * var)[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        beta = beta + step
        if not np.all(np.isfinite(beta)):
            return float("nan"), float("nan")
        if float(np.max(np.abs(step))) < 1e-8:
            break
    intercept, slope = float(beta[0]), float(beta[1])
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return float("nan"), float("nan")
    return slope, intercept


def _net_benefit(
    *,
    predicted: np.ndarray,
    observed: np.ndarray,
    threshold: float,
    known: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> float:
    pred = np.asarray(predicted, dtype=float)
    obs = np.asarray(observed, dtype=bool)
    mask = np.isfinite(pred)
    denominator = float(mask.sum())
    if known is not None:
        mask &= np.asarray(known, dtype=bool)
    weights = None
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        mask &= np.isfinite(weights) & (weights > 0.0)
    if not mask.any() or not 0.0 < threshold < 1.0:
        return float("nan")
    pred = pred[mask]
    obs = obs[mask]
    if weights is None:
        weights = np.ones_like(pred, dtype=float)
        denominator = float(len(pred))
    else:
        weights = weights[mask]
    selected = pred >= threshold
    if denominator <= 0.0:
        return float("nan")
    true_positive = float(np.sum(weights * (selected & obs))) / denominator
    false_positive = float(np.sum(weights * (selected & ~obs))) / denominator
    return true_positive - false_positive * (threshold / (1.0 - threshold))


def _d_calibration(
    *,
    test_time: np.ndarray,
    test_event: np.ndarray,
    survival_probs: np.ndarray,
    survival_times: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Return censored D-calibration using uniform redistribution below S(C).

    For an event, all mass enters the bin containing ``S(T)``. For a censored
    observation, the conditional mass is spread uniformly over ``[0, S(C)]``.
    Curves outside their supplied time grid are excluded rather than extrapolated.
    """
    time = np.asarray(test_time, dtype=float)
    event = np.asarray(test_event, dtype=bool)
    probs = np.asarray(survival_probs, dtype=float)
    grid = np.asarray(survival_times, dtype=float)
    eligible = np.isfinite(time) & (time >= grid[0]) & (time <= grid[-1])
    if not bool(eligible.any()) or n_bins < 2:
        return {
            "d_calibration_chi2": float("nan"),
            "d_calibration_p_value": float("nan"),
            "d_calibration_n": 0.0,
        }
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    observed = np.zeros(int(n_bins), dtype=float)
    for row_index in np.flatnonzero(eligible):
        survival_at_observation = float(
            np.interp(time[row_index], grid, probs[row_index], left=1.0, right=float(probs[row_index, -1]))
        )
        if event[row_index]:
            bin_index = min(int(np.searchsorted(edges, survival_at_observation, side="right") - 1), n_bins - 1)
            observed[max(bin_index, 0)] += 1.0
        elif survival_at_observation <= 0.0:
            observed[0] += 1.0
        else:
            for bin_index in range(int(n_bins)):
                overlap = max(
                    0.0,
                    min(survival_at_observation, float(edges[bin_index + 1])) - float(edges[bin_index]),
                )
                observed[bin_index] += overlap / survival_at_observation
    n_eligible = int(eligible.sum())
    expected = np.full(int(n_bins), n_eligible / float(n_bins), dtype=float)
    from scipy.stats import chisquare

    result = chisquare(observed, expected)
    return {
        "d_calibration_chi2": float(result.statistic),
        "d_calibration_p_value": float(result.pvalue),
        "d_calibration_n": float(n_eligible),
    }


def compute_risk_metrics(
    *,
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    risk_scores: np.ndarray,
) -> MetricBundle:
    """Compute risk-only metrics while explicitly withholding curve metrics."""
    validated = validate_prediction_bundle(
        risk_scores=risk_scores,
        survival_probs=None,
        survival_times=None,
        n_rows=len(test_time),
        require_survival=False,
        survival_capable=False,
    )
    harrell = compute_harrell_c_index(
        eval_time=np.asarray(test_time),
        eval_event=np.asarray(test_event),
        eval_risk_scores=validated.risk,
    )
    uno = compute_uno_c_index(
        train_time=np.asarray(train_time),
        train_event=np.asarray(train_event),
        eval_time=np.asarray(test_time),
        eval_event=np.asarray(test_event),
        eval_risk_scores=validated.risk,
    )
    return MetricBundle(
        uno_c=uno,
        harrell_c=harrell,
        ibs=float("nan"),
        td_auc_25=float("nan"),
        td_auc_50=float("nan"),
        td_auc_75=float("nan"),
        metadata={
            "metric_support_policy": "risk_only_v1",
            "full_distribution_eligible": False,
            "full_distribution_ineligible_reason": "survival_capability_mismatch",
        },
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
    # L2 (note only): horizons are per-split quantiles of the split's own training
    # event times, so aggregating horizon-indexed metrics (td_auc_50, brier_50,
    # ...) across splits mixes slightly different absolute times. The concrete
    # grid used per split is recorded via the horizon_used_* columns.
    event_times = time[event.astype(bool)]
    if event_times.size == 0:
        return (1.0, 2.0, 3.0)
    return tuple(float(np.quantile(event_times, q)) for q in quantiles)
