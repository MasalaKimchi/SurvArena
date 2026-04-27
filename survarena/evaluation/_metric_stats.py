from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

MAXIMIZE_METRICS = {
    "validation_score",
    "uno_c",
    "harrell_c",
    "td_auc_25",
    "td_auc_50",
    "td_auc_75",
    "calibration_slope_50",
    "net_benefit_50",
}
MINIMIZE_METRICS = {
    "ibs",
    "brier_25",
    "brier_50",
    "brier_75",
    "runtime_sec",
    "fit_time_sec",
    "infer_time_sec",
    "peak_memory_mb",
}


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


def metric_direction(metric: str) -> str:
    if metric in MINIMIZE_METRICS:
        return "minimize"
    if metric in MAXIMIZE_METRICS:
        return "maximize"
    raise ValueError(f"Unknown metric direction for '{metric}'.")
