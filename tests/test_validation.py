from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survarena.automl.validation import ValidationPlan, build_validation_plan, default_holdout_frac, prepare_validation_fold_cache
from survarena.data.schema import DatasetMetadata, SurvivalDataset


def _dataset(frame: pd.DataFrame, *, time: list[float], event: list[int]) -> SurvivalDataset:
    return SurvivalDataset(
        metadata=DatasetMetadata(dataset_id="toy", name="toy", source="unit_test"),
        X=frame.reset_index(drop=True),
        time=np.asarray(time, dtype=float),
        event=np.asarray(event, dtype=int),
    )


@pytest.mark.parametrize(
    ("n_rows", "expected"),
    [
        (10, 0.2),
        (499, 0.2),
        (500, 0.15),
        (4_999, 0.15),
        (5_000, 0.1),
        (24_999, 0.1),
        (25_000, 0.05),
    ],
)
def test_default_holdout_frac_uses_documented_thresholds(n_rows: int, expected: float) -> None:
    assert default_holdout_frac(n_rows) == expected


def test_build_validation_plan_aligns_tuning_columns_by_name() -> None:
    training = _dataset(
        pd.DataFrame({"age": [61, 57, 70, 66], "stage": ["i", "ii", "ii", "iii"]}),
        time=[1.0, 2.0, 3.0, 4.0],
        event=[1, 0, 1, 0],
    )
    tuning = _dataset(
        pd.DataFrame({"stage": ["ii", "i"], "age": [59, 63]}),
        time=[5.0, 6.0],
        event=[1, 0],
    )

    plan = build_validation_plan(training, tuning_dataset=tuning, seed=7)

    assert plan.source == "tuning_data"
    assert list(plan.validation_X.columns) == ["age", "stage"]
    assert plan.validation_X.to_dict(orient="list") == {"age": [59, 63], "stage": ["ii", "i"]}


def test_build_validation_plan_rejects_tuning_feature_mismatch() -> None:
    training = _dataset(
        pd.DataFrame({"age": [61, 57, 70, 66], "stage": ["i", "ii", "ii", "iii"]}),
        time=[1.0, 2.0, 3.0, 4.0],
        event=[1, 0, 1, 0],
    )
    tuning = _dataset(
        pd.DataFrame({"age": [59, 63], "grade": ["a", "b"]}),
        time=[5.0, 6.0],
        event=[1, 0],
    )

    with pytest.raises(ValueError, match="missing columns: \\['stage'\\].*extra columns: \\['grade'\\]"):
        build_validation_plan(training, tuning_dataset=tuning, seed=7)


def test_build_validation_plan_rejects_small_auto_holdout_dataset() -> None:
    dataset = _dataset(
        pd.DataFrame({"age": [61, 57, 70]}),
        time=[1.0, 2.0, 3.0],
        event=[1, 0, 1],
    )

    with pytest.raises(ValueError, match="Need at least 4 rows"):
        build_validation_plan(dataset, seed=0)


def test_build_validation_plan_rejects_low_count_event_classes() -> None:
    dataset = _dataset(
        pd.DataFrame({"age": [61, 57, 70, 66]}),
        time=[1.0, 2.0, 3.0, 4.0],
        event=[1, 0, 0, 0],
    )

    with pytest.raises(ValueError, match="requires at least two samples in each event class"):
        build_validation_plan(dataset, seed=0)


def test_prepare_validation_fold_cache_skips_numeric_scaling_for_rsf() -> None:
    plan = ValidationPlan(
        source="tuning_data",
        holdout_frac=None,
        train_X=pd.DataFrame({"age": [20.0, 40.0], "stage": ["i", "ii"]}),
        train_time=np.asarray([1.0, 2.0], dtype=float),
        train_event=np.asarray([1, 0], dtype=int),
        validation_X=pd.DataFrame({"age": [30.0], "stage": ["ii"]}),
        validation_time=np.asarray([1.5], dtype=float),
        validation_event=np.asarray([1], dtype=int),
    )

    rsf_fold = prepare_validation_fold_cache(method_id="rsf", plan=plan)[0]
    coxph_fold = prepare_validation_fold_cache(method_id="coxph", plan=plan)[0]

    np.testing.assert_allclose(rsf_fold["X_train"][:, 0], np.asarray([20.0, 40.0]))
    assert not np.allclose(coxph_fold["X_train"][:, 0], rsf_fold["X_train"][:, 0])
    assert np.isclose(float(coxph_fold["X_train"][:, 0].mean()), 0.0)
