from __future__ import annotations

import math

import numpy as np
import pytest

from survarena.benchmark import tuning
from survarena.evaluation.metrics import MetricBundle, compute_primary_metric_score, horizons_from_train_event_times


class FakeTrial:
    def __init__(self, *, use_none: bool = False) -> None:
        self.use_none = use_none

    def suggest_int(self, name: str, low: int, high: int) -> int:
        assert low <= high
        return high

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float:
        assert low <= high
        return high if log else (low + high) / 2.0

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        if name.endswith("_is_none"):
            return self.use_none
        return choices[-1]


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

    assert as_dict == {
        "uno_c": 0.6100000143051147,
        "harrell_c": 0.63,
        "ibs": 0.18000000715255737,
        "td_auc_25": 0.6600000262260437,
        "td_auc_50": 0.6700000166893005,
        "td_auc_75": 0.6800000071525574,
    }
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


def test_method_param_suggestions_supports_declared_search_types() -> None:
    params = tuning.method_param_suggestions(
        FakeTrial(),
        {
            "search_space": {
                "max_depth": {"type": "int", "low": 2, "high": 5},
                "lr": {"type": "float", "low": 1e-3, "high": 1e-1, "log": True},
                "criterion": {"type": "categorical", "choices": ["a", "b"]},
                "min_leaf": {"type": "int_or_none", "low": 2, "high": 7},
            }
        },
    )

    assert params == {"max_depth": 5, "lr": 0.1, "criterion": "b", "min_leaf": 7}


def test_method_param_suggestions_allows_int_or_none_to_return_none() -> None:
    params = tuning.method_param_suggestions(
        FakeTrial(use_none=True),
        {"search_space": {"min_leaf": {"type": "int_or_none", "low": 2, "high": 7}}},
    )

    assert params == {"min_leaf": None}


def test_method_param_suggestions_falls_back_to_default_params_when_search_space_is_empty() -> None:
    params = tuning.method_param_suggestions(
        FakeTrial(),
        {"default_params": {"max_depth": 5, "seed": 17}},
    )

    assert params == {"max_depth": 5, "seed": 17}


def test_method_param_suggestions_rejects_unknown_search_spec_types() -> None:
    with pytest.raises(ValueError, match="Unsupported search spec type"):
        tuning.method_param_suggestions(
            FakeTrial(),
            {"search_space": {"oops": {"type": "matrix", "low": 0, "high": 1}}},
        )


def test_tuning_helpers_strip_runtime_only_defaults_and_validate_direction() -> None:
    assert tuning.resolve_runtime_method_params({"alpha": 0.1, "seed": 3}, seed=11) == {"alpha": 0.1, "seed": 11}
    assert tuning._searchable_default_params({"default_params": {"alpha": 0.1, "seed": 3}}) == {"alpha": 0.1}
    assert tuning._metric_optimization_direction("harrell_c") == "maximize"
    assert tuning._metric_optimization_direction("uno_c") == "maximize"
    assert tuning._is_better_score(0.7, 0.6, direction="maximize") is True
    assert tuning._is_better_score(0.3, 0.4, direction="minimize") is True
    with pytest.raises(ValueError, match="Unsupported primary metric"):
        tuning._metric_optimization_direction("ibs")
    with pytest.raises(ValueError, match="Unsupported optimization direction"):
        tuning._is_better_score(0.3, 0.4, direction="sideways")


def test_metric_bundle_float_conversion_stays_finite_for_normal_values() -> None:
    values = MetricBundle(0.6, 0.7, 0.2, 0.65, 0.66, 0.67).to_dict()

    assert math.isfinite(values["uno_c"])
    assert math.isfinite(values["harrell_c"])
