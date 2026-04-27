from __future__ import annotations

import numpy as np
import pandas as pd


AUTO_CATEGORICAL_MAX_UNIQUE = 20
AUTO_CATEGORICAL_MAX_RATIO = 0.20
AUTO_CATEGORICAL_MIN_ROWS = 30


def is_integer_like_numeric(series: pd.Series) -> bool:
    non_null = pd.to_numeric(series.dropna(), errors="coerce")
    if non_null.empty or non_null.isna().any():
        return False
    values = non_null.to_numpy(dtype=np.float64, copy=False)
    return bool(np.all(np.isfinite(values)) and np.allclose(values, np.round(values)))


def is_low_cardinality_numeric_categorical(series: pd.Series) -> bool:
    n_unique = int(series.nunique(dropna=True))
    if n_unique <= 1:
        return False
    n_rows = max(int(series.notna().sum()), 1)
    if n_rows < AUTO_CATEGORICAL_MIN_ROWS:
        return False
    return (
        n_unique <= AUTO_CATEGORICAL_MAX_UNIQUE
        and float(n_unique / n_rows) <= AUTO_CATEGORICAL_MAX_RATIO
        and is_integer_like_numeric(series)
    )
