from __future__ import annotations

import numpy as np
import pytest

from survarena.evaluation.metrics import (
    _calibration_line,
    _survival_at_times,
    compute_harrell_c_index,
    compute_survival_metrics,
    compute_uno_c_index,
)
from survarena.evaluation.predictions import PredictionValidationError, validate_prediction_bundle


def _valid_bundle() -> dict[str, object]:
    return {
        "risk_scores": np.asarray([0.8, 0.2]),
        "survival_probs": np.asarray([[0.95, 0.8, 0.6], [0.99, 0.9, 0.85]]),
        "survival_times": np.asarray([1.0, 2.0, 3.0]),
        "n_rows": 2,
    }


@pytest.mark.parametrize(
    ("changes", "reason_code"),
    [
        ({"risk_scores": np.ones((2, 1))}, "risk_dimensionality"),
        ({"risk_scores": np.asarray([0.1])}, "risk_row_count"),
        ({"risk_scores": np.asarray([0.1, np.nan])}, "risk_nonfinite"),
        ({"survival_probs": np.ones(3)}, "survival_dimensionality"),
        ({"survival_probs": np.ones((1, 3))}, "survival_row_count"),
        ({"survival_times": np.asarray([1.0, 2.0])}, "survival_grid_length"),
        ({"survival_times": np.asarray([1.0, np.nan, 3.0])}, "survival_grid_nonfinite"),
        ({"survival_times": np.asarray([1.0, 1.0, 3.0])}, "survival_grid_not_strictly_increasing"),
        ({"survival_probs": np.asarray([[0.9, np.nan, 0.7], [0.9, 0.8, 0.7]])}, "survival_nonfinite"),
        ({"survival_probs": np.asarray([[1.01, 0.8, 0.7], [0.9, 0.8, 0.7]])}, "survival_out_of_bounds"),
        ({"survival_probs": np.asarray([[0.9, 0.95, 0.7], [0.9, 0.8, 0.7]])}, "survival_nonmonotone"),
    ],
)
def test_prediction_contract_rejects_invalid_bundles(
    changes: dict[str, object],
    reason_code: str,
) -> None:
    payload = {**_valid_bundle(), **changes}

    with pytest.raises(PredictionValidationError) as exc_info:
        validate_prediction_bundle(**payload)

    assert exc_info.value.reason_code == reason_code


def test_prediction_contract_accepts_constant_decreasing_and_numeric_tolerance() -> None:
    payload = _valid_bundle()
    payload["survival_probs"] = np.asarray(
        [
            [0.9, 0.9 + 5e-9, 0.7],
            [1.0, 0.8, 0.4],
        ]
    )

    validated = validate_prediction_bundle(**payload, monotonicity_tolerance=1e-8)

    assert validated.has_survival_distribution
    np.testing.assert_array_equal(validated.risk, payload["risk_scores"])


def test_prediction_contract_gates_risk_only_capability() -> None:
    with pytest.raises(PredictionValidationError) as exc_info:
        validate_prediction_bundle(
            risk_scores=np.asarray([0.8, 0.2]),
            survival_probs=None,
            survival_times=None,
            n_rows=2,
            require_survival=True,
            survival_capable=False,
        )
    assert exc_info.value.reason_code == "survival_capability_mismatch"

    validated = validate_prediction_bundle(
        risk_scores=np.asarray([0.8, 0.2]),
        survival_probs=None,
        survival_times=None,
        n_rows=2,
        require_survival=False,
        survival_capable=False,
    )
    assert not validated.has_survival_distribution
    assert validated.survival is None


def _structured_outcome(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    outcome = np.empty(len(time), dtype=[("event", "?"), ("time", "f8")])
    outcome["event"] = event.astype(bool)
    outcome["time"] = time.astype(float)
    return outcome


def _metric_fixture() -> dict[str, np.ndarray | tuple[float, float, float]]:
    train_time = np.arange(1.0, 13.0)
    train_event = np.asarray([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=bool)
    test_time = np.arange(1.5, 10.0)
    test_event = np.asarray([1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=bool)
    risk = np.asarray([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15])
    grid = np.arange(2.0, 10.0)
    survival = np.exp(-np.outer(risk, grid) / 8.0)
    return {
        "train_time": train_time,
        "train_event": train_event,
        "test_time": test_time,
        "test_event": test_event,
        "risk": risk,
        "grid": grid,
        "survival": survival,
        "horizons": (3.0, 5.0, 7.0),
    }


def test_metric_reference_matches_scikit_survival() -> None:
    from sksurv.metrics import (
        brier_score,
        concordance_index_censored,
        concordance_index_ipcw,
        cumulative_dynamic_auc,
        integrated_brier_score,
    )

    fixture = _metric_fixture()
    train_time = fixture["train_time"]
    train_event = fixture["train_event"]
    test_time = fixture["test_time"]
    test_event = fixture["test_event"]
    risk = fixture["risk"]
    grid = fixture["grid"]
    survival = fixture["survival"]
    horizons = fixture["horizons"]
    assert isinstance(train_time, np.ndarray)
    assert isinstance(train_event, np.ndarray)
    assert isinstance(test_time, np.ndarray)
    assert isinstance(test_event, np.ndarray)
    assert isinstance(risk, np.ndarray)
    assert isinstance(grid, np.ndarray)
    assert isinstance(survival, np.ndarray)
    assert isinstance(horizons, tuple)
    train_y = _structured_outcome(train_time, train_event)
    test_y = _structured_outcome(test_time, test_event)

    actual = compute_survival_metrics(
        train_time=train_time,
        train_event=train_event,
        test_time=test_time,
        test_event=test_event,
        risk_scores=risk,
        survival_probs=survival,
        survival_times=grid,
        horizons=horizons,
    ).to_dict()
    horizon_survival = _survival_at_times(survival, grid, horizons)
    _, expected_brier = brier_score(train_y, test_y, horizon_survival, np.asarray(horizons))
    expected_auc, _ = cumulative_dynamic_auc(train_y, test_y, 1.0 - horizon_survival, np.asarray(horizons))
    expected_ibs = integrated_brier_score(train_y, test_y, survival, grid)

    assert float(actual["harrell_c"]) == pytest.approx(
        concordance_index_censored(test_event, test_time, risk)[0], abs=1e-7
    )
    assert float(actual["uno_c"]) == pytest.approx(concordance_index_ipcw(train_y, test_y, risk)[0], abs=1e-7)
    np.testing.assert_allclose(
        [actual["brier_25"], actual["brier_50"], actual["brier_75"]],
        expected_brier,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [actual["td_auc_25"], actual["td_auc_50"], actual["td_auc_75"]],
        expected_auc,
        atol=1e-6,
    )
    assert float(actual["ibs"]) == pytest.approx(expected_ibs, abs=1e-6)
    assert np.isfinite(float(actual["d_calibration_chi2"]))
    assert np.isfinite(float(actual["d_calibration_p_value"]))


def test_ipcw_support_uses_censoring_distribution_not_last_event() -> None:
    metrics = compute_survival_metrics(
        train_time=np.asarray([1.0, 2.0, 3.0, 10.0]),
        train_event=np.asarray([1, 1, 1, 0]),
        test_time=np.asarray([1.5, 3.5, 5.5, 8.0]),
        test_event=np.asarray([1, 1, 0, 0]),
        risk_scores=np.asarray([0.9, 0.7, 0.3, 0.1]),
        survival_probs=np.asarray(
            [
                [0.9, 0.7, 0.5, 0.3],
                [0.95, 0.8, 0.6, 0.4],
                [0.99, 0.92, 0.82, 0.7],
                [0.99, 0.95, 0.9, 0.85],
            ]
        ),
        survival_times=np.asarray([1.0, 3.0, 5.0, 7.0]),
        horizons=(3.0, 5.0, 11.0),
    ).to_dict()

    assert bool(metrics["horizon_eligible_50"])
    assert np.isfinite(float(metrics["brier_50"]))
    assert float(metrics["horizon_requested_75"]) == 11.0
    assert np.isnan(float(metrics["horizon_used_75"]))
    assert metrics["horizon_reason_75"] == "outside_ipcw_support"
    assert float(metrics["metric_support_upper"]) > 5.0


def test_calibration_line_matches_independent_scipy_optimizer() -> None:
    from scipy.optimize import minimize

    predicted = np.asarray([0.08, 0.15, 0.27, 0.41, 0.58, 0.72, 0.84, 0.93])
    observed = np.asarray([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)
    slope, intercept = _calibration_line(predicted=predicted, observed=observed)
    x = np.log(predicted / (1.0 - predicted))

    def objective(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * x
        return float(np.sum(np.logaddexp(0.0, eta) - observed * eta))

    reference = minimize(objective, np.zeros(2), method="BFGS")
    assert reference.success
    assert intercept == pytest.approx(float(reference.x[0]), abs=1e-5)
    assert slope == pytest.approx(float(reference.x[1]), abs=1e-5)


def test_concordance_reference_handles_tied_risk_scores() -> None:
    from sksurv.metrics import concordance_index_censored

    time = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    event = np.asarray([1, 1, 0, 1, 0], dtype=bool)
    risk = np.asarray([0.8, 0.8, 0.4, 0.2, 0.2])

    actual = compute_harrell_c_index(eval_time=time, eval_event=event, eval_risk_scores=risk)

    assert actual == pytest.approx(concordance_index_censored(event, time, risk)[0], abs=1e-7)


def test_uno_reference_handles_heavy_censoring() -> None:
    from sksurv.metrics import concordance_index_ipcw

    train_time = np.arange(1.0, 11.0)
    train_event = np.asarray([1, 0, 0, 0, 1, 0, 0, 1, 0, 1], dtype=bool)
    test_time = np.asarray([1.5, 2.5, 4.5, 6.5, 8.5])
    test_event = np.asarray([1, 0, 0, 1, 0], dtype=bool)
    risk = np.asarray([0.9, 0.7, 0.4, 0.6, 0.2])
    expected = concordance_index_ipcw(
        _structured_outcome(train_time, train_event),
        _structured_outcome(test_time, test_event),
        risk,
    )[0]

    actual = compute_uno_c_index(
        train_time=train_time,
        train_event=train_event,
        eval_time=test_time,
        eval_event=test_event,
        eval_risk_scores=risk,
    )

    assert actual == pytest.approx(expected, abs=1e-6)
