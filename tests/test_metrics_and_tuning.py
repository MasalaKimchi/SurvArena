from __future__ import annotations

import math

import numpy as np
import pytest

from survarena.benchmark import tuning
from survarena.evaluation.metrics import (
    MetricBundle,
    _event_status_at_horizons,
    compute_primary_metric_score,
    horizons_from_train_event_times,
)


def test_metric_bundle_to_dict_casts_all_metrics_to_plain_floats() -> None:
    bundle = MetricBundle(
        uno_c=np.float32(0.61),
        harrell_c=np.float64(0.63),
        ibs=np.float32(0.18),
        td_auc_25=np.float32(0.66),
        td_auc_50=np.float32(0.67),
        td_auc_75=np.float32(0.68),
    )

    as_dict = bundle.to_dict()

    assert {key: as_dict[key] for key in ["uno_c", "harrell_c", "ibs", "td_auc_25", "td_auc_50", "td_auc_75"]} == {
        "uno_c": 0.6100000143051147,
        "harrell_c": 0.63,
        "ibs": 0.18000000715255737,
        "td_auc_25": 0.6600000262260437,
        "td_auc_50": 0.6700000166893005,
        "td_auc_75": 0.6800000071525574,
    }
    for key in ["brier_25", "brier_50", "brier_75", "calibration_slope_50", "calibration_intercept_50", "net_benefit_50"]:
        assert math.isnan(as_dict[key])
    assert all(isinstance(value, float) for value in as_dict.values())


def test_compute_primary_metric_score_dispatches_to_harrell(monkeypatch) -> None:
    called: dict[str, np.ndarray] = {}

    def fake_harrell(**kwargs) -> float:
        called.update(kwargs)
        return 0.73

    monkeypatch.setattr("survarena.evaluation.metrics.compute_harrell_c_index", fake_harrell)

    score = compute_primary_metric_score(
        primary_metric="harrell_c",
        train_time=np.asarray([1.0, 2.0]),
        train_event=np.asarray([1, 0]),
        eval_time=np.asarray([1.5, 2.5]),
        eval_event=np.asarray([1, 0]),
        eval_risk_scores=np.asarray([0.2, 0.4]),
    )

    assert score == 0.73
    np.testing.assert_allclose(called["eval_time"], np.asarray([1.5, 2.5]))


def test_compute_primary_metric_score_dispatches_to_uno(monkeypatch) -> None:
    called: dict[str, np.ndarray] = {}

    def fake_uno(**kwargs) -> float:
        called.update(kwargs)
        return 0.69

    monkeypatch.setattr("survarena.evaluation.metrics.compute_uno_c_index", fake_uno)

    score = compute_primary_metric_score(
        primary_metric="uno_c",
        train_time=np.asarray([1.0, 2.0]),
        train_event=np.asarray([1, 0]),
        eval_time=np.asarray([1.5, 2.5]),
        eval_event=np.asarray([1, 0]),
        eval_risk_scores=np.asarray([0.2, 0.4]),
    )

    assert score == 0.69
    np.testing.assert_allclose(called["train_time"], np.asarray([1.0, 2.0]))


def test_compute_primary_metric_score_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported primary metric"):
        compute_primary_metric_score(
            primary_metric="ibs",
            train_time=np.asarray([1.0, 2.0]),
            train_event=np.asarray([1, 0]),
            eval_time=np.asarray([1.5, 2.5]),
            eval_event=np.asarray([1, 0]),
            eval_risk_scores=np.asarray([0.2, 0.4]),
        )


def test_horizons_from_train_event_times_uses_defaults_without_events() -> None:
    horizons = horizons_from_train_event_times(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0, 0, 0]),
    )

    assert horizons == (1.0, 2.0, 3.0)


def test_horizons_from_train_event_times_uses_event_quantiles() -> None:
    horizons = horizons_from_train_event_times(
        np.asarray([1.0, 2.0, 4.0, 8.0]),
        np.asarray([0, 1, 1, 1]),
    )

    assert horizons == (3.0, 4.0, 6.0)


def test_horizon_event_status_marks_censored_before_horizon_unknown() -> None:
    observed, known = _event_status_at_horizons(
        np.asarray([1.0, 2.0, 4.0]),
        np.asarray([0, 1, 0]),
        (3.0, 5.0, 6.0),
    )

    assert observed[:, 0].tolist() == [0.0, 1.0, 0.0]
    assert known[:, 0].tolist() == [False, True, True]
    assert known[:, 1].tolist() == [False, True, False]


def test_selection_helpers_strip_runtime_only_defaults() -> None:
    assert tuning.resolve_runtime_method_params({"alpha": 0.1, "seed": 3}, seed=11) == {"alpha": 0.1, "seed": 11}
    assert tuning._searchable_default_params({"default_params": {"alpha": 0.1, "seed": 3}}) == {"alpha": 0.1}


def test_metric_bundle_float_conversion_stays_finite_for_normal_values() -> None:
    values = MetricBundle(0.6, 0.7, 0.2, 0.65, 0.66, 0.67).to_dict()

    assert math.isfinite(values["uno_c"])
    assert math.isfinite(values["harrell_c"])
