from __future__ import annotations

from collections import Counter
import warnings

import numpy as np
import pandas as pd

from survarena.data.schema import DatasetDiagnostics, FeatureMetadata


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return False
        sample = non_null.astype(str).head(25)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return float(parsed.notna().mean()) >= 0.8
    return False


def infer_feature_metadata(frame: pd.DataFrame) -> list[FeatureMetadata]:
    n_rows = max(len(frame), 1)
    feature_metadata: list[FeatureMetadata] = []

    for column in frame.columns:
        series = frame[column]
        n_unique = int(series.nunique(dropna=True))
        missing_fraction = float(series.isna().mean())
        is_constant = int(series.nunique(dropna=False)) <= 1
        uniqueness_ratio = float(n_unique / n_rows)

        if pd.api.types.is_bool_dtype(series):
            inferred_type = "boolean"
        elif pd.api.types.is_numeric_dtype(series):
            inferred_type = "numerical"
        elif _is_datetime_like(series):
            inferred_type = "datetime"
        else:
            avg_length = float(series.dropna().astype(str).str.len().mean()) if series.dropna().size else 0.0
            inferred_type = "text" if avg_length >= 32.0 else "categorical"

        is_id_like = bool(
            not is_constant
            and n_unique > 0
            and uniqueness_ratio >= 0.98
            and inferred_type in {"categorical", "text", "numerical"}
        )

        if inferred_type in {"categorical", "text"}:
            if n_unique >= 100:
                cardinality = "high"
            elif n_unique >= 20:
                cardinality = "medium"
            else:
                cardinality = "low"
        else:
            cardinality = "n/a"

        feature_metadata.append(
            FeatureMetadata(
                name=str(column),
                inferred_type=inferred_type,
                dtype=str(series.dtype),
                n_unique=n_unique,
                missing_fraction=missing_fraction,
                is_constant=is_constant,
                is_id_like=is_id_like,
                cardinality=cardinality,
            )
        )

    return feature_metadata


def summarize_feature_types(feature_metadata: list[FeatureMetadata]) -> list[str]:
    ordered_types = ["numerical", "categorical", "boolean", "datetime", "text"]
    present = {feature.inferred_type for feature in feature_metadata}
    return [feature_type for feature_type in ordered_types if feature_type in present]


def build_dataset_diagnostics(
    frame: pd.DataFrame,
    *,
    event: np.ndarray,
    feature_metadata: list[FeatureMetadata],
) -> DatasetDiagnostics:
    n_rows = int(len(frame))
    n_features = int(frame.shape[1])
    n_events = int(np.asarray(event, dtype=np.int32).sum())
    event_rate = float(n_events / n_rows) if n_rows else 0.0
    censoring_rate = float(1.0 - event_rate) if n_rows else 0.0
    missing_fraction = float(frame.isna().mean().mean()) if n_features > 0 else 0.0

    feature_type_counts = dict(Counter(feature.inferred_type for feature in feature_metadata))
    high_cardinality_features = [
        feature.name for feature in feature_metadata if feature.cardinality == "high" and not feature.is_id_like
    ]
    id_like_features = [feature.name for feature in feature_metadata if feature.is_id_like]

    warnings: list[str] = []
    if n_events < 25:
        warnings.append(
            "Very few observed events were detected (<25); deep or high-capacity survival models may be unstable."
        )
    if event_rate < 0.1:
        warnings.append("Observed event rate is below 10%; expect heavy censoring and noisy model selection.")
    if missing_fraction > 0.3:
        warnings.append("Feature matrix has more than 30% average missingness; imputation quality may dominate results.")
    if id_like_features:
        warnings.append(
            "Potential identifier-like features were detected and retained as inputs; drop them to avoid memorization."
        )
    if any(feature.inferred_type == "datetime" for feature in feature_metadata):
        warnings.append("Datetime-like features were detected, but no specialized datetime featurization is implemented yet.")
    if any(feature.inferred_type == "text" for feature in feature_metadata):
        warnings.append("Text-like features were detected, but only generic categorical handling is available today.")
    if any(feature.is_constant for feature in feature_metadata):
        warnings.append("Constant features were detected; they will be removed during preprocessing.")

    return DatasetDiagnostics(
        n_rows=n_rows,
        n_features=n_features,
        n_events=n_events,
        event_rate=event_rate,
        censoring_rate=censoring_rate,
        missing_fraction=missing_fraction,
        feature_type_counts=feature_type_counts,
        high_cardinality_features=high_cardinality_features,
        id_like_features=id_like_features,
        warnings=warnings,
    )
