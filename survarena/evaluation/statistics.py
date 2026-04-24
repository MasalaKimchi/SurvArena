from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

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
    stratum = ["benchmark_id", "dataset_id"]
    if "hpo_mode" in frame.columns:
        stratum = stratum + ["hpo_mode"]
    ranked[f"{metric}_rank"] = ranked.groupby(stratum)[metric].rank(
        method="average",
        ascending=ascending,
        na_option="bottom",
    )
    return ranked


def aggregate_rank_summary(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    ranked = add_dataset_ranks(frame, metric=metric)
    rank_col = f"{metric}_rank"
    group_keys = ["benchmark_id", "method_id"]
    if "hpo_mode" in ranked.columns:
        group_keys = ["benchmark_id", "method_id", "hpo_mode"]
    summary = ranked.groupby(group_keys, as_index=False).agg(
        mean_rank=(rank_col, "mean"),
        median_rank=(rank_col, "median"),
        mean_score=(metric, "mean"),
        median_score=(metric, "median"),
        datasets_evaluated=("dataset_id", "nunique"),
    )
    summary.sort_values(
        [c for c in ["benchmark_id", "hpo_mode", "mean_rank", "median_rank"] if c in summary.columns],
        inplace=True,
    )
    return summary.reset_index(drop=True)


def pairwise_win_rate(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    higher_is_better = metric_direction(metric) == "maximize"
    rows: list[dict[str, object]] = []
    group_cols = ["benchmark_id", "dataset_id"]
    if "hpo_mode" in frame.columns:
        group_cols = ["benchmark_id", "dataset_id", "hpo_mode"]
    for _key, sub in frame.groupby(group_cols):
        if len(group_cols) == 2:
            benchmark_id, dataset_id = _key
        else:
            benchmark_id, dataset_id, _hpo = _key
        values = sub[["method_id", metric]].dropna()
        for _, left in values.iterrows():
            for _, right in values.iterrows():
                if left["method_id"] == right["method_id"]:
                    continue
                left_score = float(left[metric])
                right_score = float(right[metric])
                win = left_score > right_score if higher_is_better else left_score < right_score
                tie = left_score == right_score
                row: dict[str, object] = {
                    "benchmark_id": benchmark_id,
                    "dataset_id": dataset_id,
                    "method_id": left["method_id"],
                    "opponent_method_id": right["method_id"],
                    "win": float(win),
                    "tie": float(tie),
                }
                if "hpo_mode" in sub.columns and not sub["hpo_mode"].empty:
                    row["hpo_mode"] = sub["hpo_mode"].iloc[0]
                rows.append(row)
    if not rows:
        empty_cols = [
            "benchmark_id",
            "method_id",
            "opponent_method_id",
            "win_rate",
            "tie_rate",
            "n",
        ]
        if "hpo_mode" in frame.columns:
            empty_cols = ["benchmark_id", "hpo_mode", "method_id", "opponent_method_id", "win_rate", "tie_rate", "n"]
        return pd.DataFrame(columns=empty_cols)
    pairwise = pd.DataFrame(rows)
    group_out = ["benchmark_id", "method_id", "opponent_method_id"]
    if "hpo_mode" in pairwise.columns:
        group_out = ["benchmark_id", "hpo_mode", "method_id", "opponent_method_id"]
    return pairwise.groupby(group_out, as_index=False).agg(
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
    if "hpo_mode" in frame.columns:
        bench_iter = frame.groupby(["benchmark_id", "hpo_mode"], sort=True)
    else:
        bench_iter = ((bid, bframe) for bid, bframe in frame.groupby("benchmark_id", sort=True))
    for key, benchmark_frame in bench_iter:
        if "hpo_mode" in frame.columns:
            benchmark_id, hpo_mode = key[0], key[1]
        else:
            benchmark_id, hpo_mode = key, None
        methods = sorted(benchmark_frame["method_id"].dropna().astype(str).unique())
        ratings = {method: float(initial_rating) for method in methods}
        match_count = {method: 0 for method in methods}
        ordered = benchmark_frame.sort_values(["dataset_id", "method_id"])
        for _dataset_id, sub in ordered.groupby("dataset_id", sort=True):
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
            out_row: dict[str, object] = {
                "benchmark_id": benchmark_id,
                "method_id": method_id,
                "elo_rating": float(rating),
                "elo_matches": int(match_count[method_id]),
                "metric": metric,
                "initial_rating": float(initial_rating),
                "k_factor": float(k_factor),
            }
            if hpo_mode is not None:
                out_row["hpo_mode"] = hpo_mode
            rows.append(out_row)
    if not rows:
        cols = [
            "benchmark_id",
            "method_id",
            "elo_rating",
            "elo_matches",
            "metric",
            "initial_rating",
            "k_factor",
        ]
        if "hpo_mode" in frame.columns:
            cols = ["benchmark_id", "hpo_mode", "method_id", "elo_rating", "elo_matches", "metric", "initial_rating", "k_factor"]
        return pd.DataFrame(columns=cols)
    result = pd.DataFrame(rows)
    sort_cols = [c for c in ["benchmark_id", "hpo_mode", "elo_rating", "method_id"] if c in result.columns]
    asc = [c != "elo_rating" for c in sort_cols]
    if "elo_rating" in sort_cols:
        ei = sort_cols.index("elo_rating")
        asc[ei] = False
    result.sort_values(sort_cols, ascending=asc, inplace=True)
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
    group_keys = ["benchmark_id", "method_id"]
    if "hpo_mode" in frame.columns:
        group_keys = ["benchmark_id", "method_id", "hpo_mode"]
    for key, sub in frame.groupby(group_keys):
        values = sub[metric].dropna().to_numpy(dtype=float)
        row_base: dict[str, object] = {
            "metric": metric,
        }
        if "hpo_mode" in frame.columns:
            b_id, m_id, h_mode = key
            row_base["benchmark_id"] = b_id
            row_base["method_id"] = m_id
            row_base["hpo_mode"] = h_mode
        else:
            b_id, m_id = key
            row_base["benchmark_id"] = b_id
            row_base["method_id"] = m_id
        if values.size == 0:
            rows.append(
                {
                    **row_base,
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
                **row_base,
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


def _holm_correction(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * m
    running_max = 0.0
    for rank, (orig_idx, p_value) in enumerate(indexed, start=1):
        adj = (m - rank + 1) * float(p_value)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(running_max, 1.0)
    return adjusted


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda item: item[1], reverse=True)
    adjusted = [1.0] * m
    running_min = 1.0
    for rank, (orig_idx, p_value) in enumerate(indexed, start=1):
        denom = m - rank + 1
        adj = float(p_value) * m / max(denom, 1)
        running_min = min(running_min, adj)
        adjusted[orig_idx] = min(running_min, 1.0)
    return adjusted


def pairwise_significance(
    frame: pd.DataFrame,
    *,
    metric: str,
    correction: str = "holm",
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    higher_is_better = metric_direction(metric) == "maximize"
    rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    group_cols = [col for col in ["benchmark_id", "dataset_id", "split_id", "seed", "hpo_mode"] if col in frame.columns]
    if not group_cols:
        group_cols = [col for col in ["benchmark_id", "dataset_id"] if col in frame.columns]
    for benchmark_id, benchmark_sub in frame.groupby("benchmark_id"):
        methods = sorted(benchmark_sub["method_id"].dropna().astype(str).unique())
        for left_index, left_method in enumerate(methods):
            for right_method in methods[left_index + 1 :]:
                left = benchmark_sub[benchmark_sub["method_id"] == left_method][group_cols + [metric]].rename(
                    columns={metric: "left_metric"}
                )
                right = benchmark_sub[benchmark_sub["method_id"] == right_method][group_cols + [metric]].rename(
                    columns={metric: "right_metric"}
                )
                merged = left.merge(right, on=group_cols, how="inner").dropna()
                if merged.empty:
                    continue
                delta = (
                    merged["left_metric"].to_numpy(dtype=float) - merged["right_metric"].to_numpy(dtype=float)
                    if higher_is_better
                    else merged["right_metric"].to_numpy(dtype=float) - merged["left_metric"].to_numpy(dtype=float)
                )
                if delta.size < 2:
                    p_value = 1.0
                else:
                    try:
                        p_value = float(stats.wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)
                    except ValueError:
                        p_value = 1.0
                effect_size = float(np.mean(delta))
                pair_rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "method_id": left_method,
                        "opponent_method_id": right_method,
                        "metric": metric,
                        "n_pairs": int(delta.size),
                        "effect_size_mean_delta": effect_size,
                        "p_value": p_value,
                        "wins": int(np.sum(delta > 0)),
                        "ties": int(np.sum(delta == 0)),
                    }
                )
                pair_rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "method_id": right_method,
                        "opponent_method_id": left_method,
                        "metric": metric,
                        "n_pairs": int(delta.size),
                        "effect_size_mean_delta": float(-effect_size),
                        "p_value": p_value,
                        "wins": int(np.sum(delta < 0)),
                        "ties": int(np.sum(delta == 0)),
                    }
                )
        benchmark_pairs = [row for row in pair_rows if row["benchmark_id"] == benchmark_id]
        p_values = [float(row["p_value"]) for row in benchmark_pairs]
        if not p_values:
            continue
        if correction == "bh":
            corrected = _benjamini_hochberg(p_values)
        else:
            corrected = _holm_correction(p_values)
        for row, corrected_p in zip(benchmark_pairs, corrected, strict=False):
            rows.append({**row, "p_value_corrected": float(corrected_p), "correction": correction})
    return pd.DataFrame(rows)


def critical_difference_summary(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    ranked = add_dataset_ranks(frame, metric=metric)
    rank_col = f"{metric}_rank"
    rows: list[dict[str, object]] = []
    if "hpo_mode" in ranked.columns:
        for (benchmark_id, hpo_mode), sub in ranked.groupby(["benchmark_id", "hpo_mode"]):
            avg = sub.groupby("method_id", as_index=False)[rank_col].mean()
            n_datasets = int(sub["dataset_id"].nunique()) if "dataset_id" in sub.columns else 1
            n_methods = int(avg["method_id"].nunique())
            q_alpha = 2.569
            cd = float(q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * max(n_datasets, 1))))
            for row in avg.to_dict(orient="records"):
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "hpo_mode": hpo_mode,
                        "method_id": str(row["method_id"]),
                        "average_rank": float(row[rank_col]),
                        "critical_difference": cd,
                        "n_methods": n_methods,
                        "n_datasets": n_datasets,
                        "metric": metric,
                    }
                )
    else:
        for benchmark_id, sub in ranked.groupby("benchmark_id"):
            avg = sub.groupby("method_id", as_index=False)[rank_col].mean()
            n_datasets = int(sub["dataset_id"].nunique()) if "dataset_id" in sub.columns else 1
            n_methods = int(avg["method_id"].nunique())
            q_alpha = 2.569
            cd = float(q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * max(n_datasets, 1))))
            for row in avg.to_dict(orient="records"):
                rows.append(
                    {
                        "benchmark_id": benchmark_id,
                        "method_id": str(row["method_id"]),
                        "average_rank": float(row[rank_col]),
                        "critical_difference": cd,
                        "n_methods": n_methods,
                        "n_datasets": n_datasets,
                        "metric": metric,
                    }
                )
    return pd.DataFrame(rows)
