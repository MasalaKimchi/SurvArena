from __future__ import annotations

from typing import Any

import pandas as pd

from survarena.data.preprocess import MAX_DENSE_ONE_HOT_FEATURES


_NO_NUMERIC_SCALING_METHOD_IDS = frozenset(
    {
        "rsf",
        "gradient_boosting_survival",
        "extra_survival_trees",
        "xgboost_cox",
        "autogluon_survival",
        "catboost_cox",
        "xgboost_aft",
        "catboost_survival_aft",
    }
)
_NATIVE_CATEGORICAL_METHOD_IDS = frozenset({"autogluon_survival", "catboost_cox", "catboost_survival_aft"})


def method_uses_scaled_numeric_features(method_id: str) -> bool:
    return method_id not in _NO_NUMERIC_SCALING_METHOD_IDS


def method_uses_native_categorical_features(method_id: str) -> bool:
    return method_id in _NATIVE_CATEGORICAL_METHOD_IDS


def method_preprocessor_kwargs(method_id: str) -> dict[str, Any]:
    return {
        "scale_numeric": method_uses_scaled_numeric_features(method_id),
        "categorical_encoding": "native" if method_uses_native_categorical_features(method_id) else "one_hot",
    }


def finalize_preprocessed_features(method_id: str, transformed: pd.DataFrame) -> Any:
    if method_uses_native_categorical_features(method_id):
        return transformed
    return transformed.to_numpy()


def method_preprocessing_summary(method_id: str) -> dict[str, Any]:
    return {
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "numeric_scaling": method_uses_scaled_numeric_features(method_id),
        "categorical_encoding": (
            "native"
            if method_uses_native_categorical_features(method_id)
            else "one_hot"
        ),
        "categorical_detection": "pandas_category_object_string_bool_or_low_cardinality_integer_numeric",
        "dense_one_hot_feature_budget": MAX_DENSE_ONE_HOT_FEATURES,
    }
