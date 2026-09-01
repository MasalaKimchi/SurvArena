from __future__ import annotations

import numpy as np
import pytest

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
