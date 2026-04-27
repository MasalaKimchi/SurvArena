from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survarena.data.preprocess import TabularPreprocessor
from survarena.data.profiling import infer_feature_metadata


def test_preprocessor_treats_boolean_and_low_cardinality_numeric_as_categorical() -> None:
    frame = pd.DataFrame(
        {
            "age": np.linspace(40.0, 79.0, 40),
            "stage": np.tile([0.0, 1.0, 2.0, 3.0], 10),
            "marker": np.arange(40, dtype=float),
            "is_male": [True, False] * 20,
        }
    )

    preprocessor = TabularPreprocessor(scale_numeric=False)
    transformed = preprocessor.fit_transform(frame)

    assert preprocessor.numeric_columns == ["age", "marker"]
    assert preprocessor.categorical_columns == ["stage", "is_male"]
    assert "stage_0.0" in transformed.columns
    assert "is_male_True" in transformed.columns
    np.testing.assert_allclose(transformed["age"].to_numpy(), frame["age"].to_numpy())


def test_native_preprocessor_preserves_categorical_frame_for_catboost() -> None:
    frame = pd.DataFrame(
        {
            "age": np.linspace(40.0, 79.0, 40),
            "stage": np.tile([0.0, 1.0, 1.0, 2.0], 10),
            "flag": [True, False] * 20,
        }
    )

    preprocessor = TabularPreprocessor(scale_numeric=False, categorical_encoding="native")
    transformed = preprocessor.fit_transform(frame)

    assert transformed["age"].dtype.kind == "f"
    assert transformed["stage"].head(4).tolist() == ["0.0", "1.0", "1.0", "2.0"]
    assert transformed["flag"].head(4).tolist() == ["True", "False", "True", "False"]


def test_preprocessor_rejects_dense_one_hot_expansion_above_budget() -> None:
    frame = pd.DataFrame({"zip_code": [f"z{i}" for i in range(6)]})

    with pytest.raises(ValueError, match="Dense one-hot preprocessing would create 6 features"):
        TabularPreprocessor(max_dense_features=5).fit(frame)


def test_feature_metadata_marks_low_cardinality_numeric_as_categorical() -> None:
    frame = pd.DataFrame(
        {
            "age": np.linspace(40.0, 79.0, 40),
            "stage": np.tile([0.0, 1.0, 2.0, 3.0], 10),
            "marker": np.arange(40, dtype=float),
            "is_male": [True, False] * 20,
        }
    )

    metadata = {feature.name: feature.inferred_type for feature in infer_feature_metadata(frame)}

    assert metadata == {
        "age": "numerical",
        "stage": "categorical",
        "marker": "numerical",
        "is_male": "boolean",
    }
