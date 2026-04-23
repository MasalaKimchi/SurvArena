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


def add_dataset_ranks(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    ascending = metric_direction(metric) == "minimize"
    ranked = frame.copy()
    ranked[f"{metric}_rank"] = ranked.groupby(["benchmark_id", "dataset_id"])[metric].rank(
        method="average",
        ascending=ascending,
        na_option="bottom",
    )
    return ranked


def aggregate_rank_summary(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    ranked = add_dataset_ranks(frame, metric=metric)
    rank_col = f"{metric}_rank"
    summary = ranked.groupby(["benchmark_id", "method_id"], as_index=False).agg(
        mean_rank=(rank_col, "mean"),
        median_rank=(rank_col, "median"),
        mean_score=(metric, "mean"),
        median_score=(metric, "median"),
        datasets_evaluated=("dataset_id", "nunique"),
    )
    summary.sort_values(["benchmark_id", "mean_rank", "median_rank"], inplace=True)
    return summary.reset_index(drop=True)


def pairwise_win_rate(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    higher_is_better = metric_direction(metric) == "maximize"
    rows: list[dict[str, object]] = []
    for (benchmark_id, dataset_id), sub in frame.groupby(["benchmark_id", "dataset_id"]):
        values = sub[["method_id", metric]].dropna()
        for _, left in values.iterrows():
            for _, right in values.iterrows():
                if left["method_id"] == right["method_id"]:
                    continue
                left_score = float(left[metric])
                right_score = float(right[metric])
                win = left_score > right_score if higher_is_better else left_score < right_score
                tie = left_score == right_score
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "dataset_id": dataset_id,
                        "method_id": left["method_id"],
                        "opponent_method_id": right["method_id"],
                        "win": float(win),
                        "tie": float(tie),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["benchmark_id", "method_id", "opponent_method_id", "win_rate", "tie_rate", "n"])
    pairwise = pd.DataFrame(rows)
    return pairwise.groupby(["benchmark_id", "method_id", "opponent_method_id"], as_index=False).agg(
        win_rate=("win", "mean"),
        tie_rate=("tie", "mean"),
        n=("win", "count"),
    )


def elo_ratings(
    frame: pd.DataFrame,
    *,
    metric: str,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    higher_is_better = metric_direction(metric) == "maximize"
    rows: list[dict[str, object]] = []
    for benchmark_id, benchmark_frame in frame.groupby("benchmark_id"):
        methods = sorted(benchmark_frame["method_id"].dropna().astype(str).unique())
        ratings = {method: float(initial_rating) for method in methods}
        match_count = {method: 0 for method in methods}
        ordered = benchmark_frame.sort_values(["dataset_id", "method_id"])
        for dataset_id, sub in ordered.groupby("dataset_id", sort=True):
            values = sub[["method_id", metric]].dropna().copy()
            values["method_id"] = values["method_id"].astype(str)
            records = values.to_dict(orient="records")
            for left_index, left in enumerate(records):
                for right in records[left_index + 1 :]:
                    left_id = str(left["method_id"])
                    right_id = str(right["method_id"])
                    left_score = float(left[metric])
                    right_score = float(right[metric])
                    if left_score == right_score:
                        outcome = 0.5
                    else:
                        left_wins = left_score > right_score if higher_is_better else left_score < right_score
                        outcome = 1.0 if left_wins else 0.0
                    expected = 1.0 / (1.0 + 10.0 ** ((ratings[right_id] - ratings[left_id]) / 400.0))
                    delta = float(k_factor) * (outcome - expected)
                    ratings[left_id] += delta
                    ratings[right_id] -= delta
                    match_count[left_id] += 1
                    match_count[right_id] += 1
        for method_id, rating in ratings.items():
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "method_id": method_id,
                    "elo_rating": float(rating),
                    "elo_matches": int(match_count[method_id]),
                    "metric": metric,
                    "initial_rating": float(initial_rating),
                    "k_factor": float(k_factor),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["benchmark_id", "method_id", "elo_rating", "elo_matches", "metric", "initial_rating", "k_factor"]
        )
    result = pd.DataFrame(rows)
    result.sort_values(["benchmark_id", "elo_rating", "method_id"], ascending=[True, False, True], inplace=True)
    return result.reset_index(drop=True)


def bootstrap_metric_ci(
    frame: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (benchmark_id, method_id), sub in frame.groupby(["benchmark_id", "method_id"]):
        values = sub[metric].dropna().to_numpy(dtype=float)
        if values.size == 0:
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "method_id": method_id,
                    "metric": metric,
                    "mean": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "n": 0,
                }
            )
            continue
        draws = np.asarray(
            [np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(int(n_bootstrap))],
            dtype=float,
        )
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "method_id": method_id,
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci95_low": float(np.percentile(draws, 2.5)),
                "ci95_high": float(np.percentile(draws, 97.5)),
                "n": int(values.size),
            }
        )
    return pd.DataFrame(rows)


def failure_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if "status" not in frame.columns:
        return pd.DataFrame(columns=["benchmark_id", "dataset_id", "method_id", "n_runs", "n_failed", "failure_rate"])
    grouped = frame.groupby(["benchmark_id", "dataset_id", "method_id"], as_index=False).agg(
        n_runs=("status", "count"),
        n_failed=("status", lambda values: int(np.sum(pd.Series(values) != "success"))),
    )
    grouped["failure_rate"] = grouped["n_failed"] / grouped["n_runs"].replace(0, np.nan)
    return grouped
