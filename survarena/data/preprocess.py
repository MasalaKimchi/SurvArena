from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from survarena.data.feature_roles import is_low_cardinality_numeric_categorical


MAX_DENSE_ONE_HOT_FEATURES = 50_000


def _to_object_frame(data: pd.DataFrame) -> pd.DataFrame:
    return data.astype("object")


def _split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols: list[str] = []
    cat_cols: list[str] = []
    for column in X.columns:
        series = X[column]
        if pd.api.types.is_bool_dtype(series):
            cat_cols.append(column)
        elif pd.api.types.is_numeric_dtype(series) and not is_low_cardinality_numeric_categorical(series):
            num_cols.append(column)
        else:
            cat_cols.append(column)
    return num_cols, cat_cols


def remove_constant_columns(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep_cols = [col for col in X.columns if X[col].nunique(dropna=False) > 1]
    return X[keep_cols].copy(), keep_cols


@dataclass(slots=True)
class TabularPreprocessor:
    scale_numeric: bool = True
    categorical_encoding: str = "one_hot"
    keep_columns: list[str] | None = None
    transformer: ColumnTransformer | None = None
    output_columns: list[str] | None = None
    numeric_columns: list[str] | None = None
    categorical_columns: list[str] | None = None
    numeric_imputer: SimpleImputer | None = None
    numeric_scaler: StandardScaler | None = None
    categorical_imputer: SimpleImputer | None = None
    max_dense_features: int = MAX_DENSE_ONE_HOT_FEATURES

    def fit(self, X_train: pd.DataFrame) -> "TabularPreprocessor":
        X_train, keep_cols = remove_constant_columns(X_train)
        if not keep_cols:
            raise ValueError("No non-constant columns remain after preprocessing filter.")
        self.keep_columns = keep_cols
        num_cols, cat_cols = _split_columns(X_train)
        self.numeric_columns = num_cols
        self.categorical_columns = cat_cols

        if self.categorical_encoding == "native":
            self.numeric_imputer = SimpleImputer(strategy="median") if num_cols else None
            self.numeric_scaler = StandardScaler() if self.scale_numeric and num_cols else None
            self.categorical_imputer = SimpleImputer(strategy="most_frequent") if cat_cols else None

            if self.numeric_imputer is not None:
                numeric_values = self.numeric_imputer.fit_transform(X_train[num_cols])
                if self.numeric_scaler is not None:
                    self.numeric_scaler.fit(numeric_values)
            if self.categorical_imputer is not None:
                self.categorical_imputer.fit(X_train[cat_cols].astype("object"))
            self.output_columns = list(keep_cols)
            self.transformer = None
            return self
        if self.categorical_encoding != "one_hot":
            raise ValueError(
                f"Unsupported categorical_encoding '{self.categorical_encoding}'. "
                "Expected 'one_hot' or 'native'."
            )

        numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))

        numeric_pipeline = Pipeline(steps=numeric_steps)
        categorical_pipeline = Pipeline(
            steps=[
                ("to_object", FunctionTransformer(_to_object_frame)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
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
        if len(self.output_columns) > int(self.max_dense_features):
            raise ValueError(
                "Dense one-hot preprocessing would create "
                f"{len(self.output_columns)} features, exceeding max_dense_features={self.max_dense_features}. "
                "Use a native-categorical method or reduce high-cardinality features."
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.keep_columns is None or self.output_columns is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        if self.categorical_encoding == "native":
            return self._transform_native_frame(X)
        if self.transformer is None:
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

    def _transform_native_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.keep_columns is None or self.output_columns is None:
            raise RuntimeError("Preprocessor must be fit before transform.")

        frame = X[self.keep_columns].copy()
        if self.numeric_columns:
            if self.numeric_imputer is None:
                raise RuntimeError("Numeric imputer is unavailable for native preprocessing.")
            numeric_values = self.numeric_imputer.transform(frame[self.numeric_columns])
            if self.numeric_scaler is not None:
                numeric_values = self.numeric_scaler.transform(numeric_values)
            numeric_frame = pd.DataFrame(
                numeric_values,
                columns=self.numeric_columns,
                index=frame.index,
            ).astype(float)
            for column in self.numeric_columns:
                frame[column] = numeric_frame[column]

        if self.categorical_columns:
            if self.categorical_imputer is None:
                raise RuntimeError("Categorical imputer is unavailable for native preprocessing.")
            categorical_values = self.categorical_imputer.transform(frame[self.categorical_columns].astype("object"))
            categorical_frame = pd.DataFrame(
                categorical_values,
                columns=self.categorical_columns,
                index=frame.index,
            )
            for column in self.categorical_columns:
                frame[column] = categorical_frame[column].astype(str)

        return frame.loc[:, self.output_columns]
