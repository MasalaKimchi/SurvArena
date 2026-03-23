from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def remove_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep_cols = [col for col in X.columns if X[col].nunique(dropna=False) > 1]
    return X[keep_cols].copy(), keep_cols


@dataclass(slots=True)
class TabularPreprocessor:
    scale_numeric: bool = True
    keep_columns: list[str] | None = None
    transformer: ColumnTransformer | None = None
    output_columns: list[str] | None = None
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None

    def fit(self, X_train: pd.DataFrame) -> "TabularPreprocessor":
        X_train, keep_cols = remove_constant_columns(X_train)
        if not keep_cols:
            raise ValueError("No non-constant columns remain after preprocessing filter.")
        self.keep_columns = keep_cols
        num_cols, cat_cols = _split_columns(X_train)
        self.numeric_columns = num_cols
        self.categorical_columns = cat_cols

        numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_pipeline = Pipeline(steps=numeric_steps)
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, num_cols),
                ("cat", categorical_pipeline, cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        self.transformer.fit(X_train)
        self.output_columns = self._build_output_columns(num_cols, cat_cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.transformer is None or self.keep_columns is None or self.output_columns is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        data = self.transformer.transform(X[self.keep_columns])
        return pd.DataFrame(data, columns=self.output_columns, index=X.index)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(X_train)
        return self.transform(X_train)

    def _build_output_columns(self, num_cols: list[str], cat_cols: list[str]) -> list[str]:
        if self.transformer is None:
            return []
        columns = list(num_cols)
        if cat_cols:
            cat_transformer = self.transformer.named_transformers_.get("cat")
            if cat_transformer is not None and hasattr(cat_transformer, "named_steps"):
                cat_encoder = cat_transformer.named_steps.get("encoder")
                if cat_encoder is not None:
                    cat_features = cat_encoder.get_feature_names_out(cat_cols).tolist()
                    columns.extend(cat_features)
        return columns
