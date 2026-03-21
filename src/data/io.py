from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_tabular_data(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    path = Path(data)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format '{suffix}'. Expected CSV or Parquet.")
