from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def summarize_metric(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "n": 0.0,
        }
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "n": float(arr.size),
    }


def summarize_frame(frame: pd.DataFrame, metric_cols: list[str]) -> dict[str, dict[str, float]]:
    return {metric: summarize_metric(frame[metric].dropna().values) for metric in metric_cols}
