from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from survarena.evaluation._metric_stats import metric_direction
from survarena.evaluation._ranking import add_dataset_ranks


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
    stratum_cols = ["benchmark_id"]
    if "hpo_mode" in frame.columns:
        stratum_cols.append("hpo_mode")
    pair_cols = [col for col in ["dataset_id", "split_id", "seed"] if col in frame.columns]
    merge_cols = stratum_cols + pair_cols
    for key, stratum_sub in frame.groupby(stratum_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        stratum = dict(zip(stratum_cols, key_tuple, strict=True))
        pair_rows: list[dict[str, object]] = []
        methods = sorted(stratum_sub["method_id"].dropna().astype(str).unique())
        for left_index, left_method in enumerate(methods):
            for right_method in methods[left_index + 1 :]:
                left = stratum_sub[stratum_sub["method_id"] == left_method][merge_cols + [metric]].rename(
                    columns={metric: "left_metric"}
                )
                right = stratum_sub[stratum_sub["method_id"] == right_method][merge_cols + [metric]].rename(
                    columns={metric: "right_metric"}
                )
                merged = left.merge(right, on=merge_cols, how="inner").dropna()
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
                        **stratum,
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
                        **stratum,
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
        p_values = [float(row["p_value"]) for row in pair_rows]
        if not p_values:
            continue
        if correction == "bh":
            corrected = _benjamini_hochberg(p_values)
        else:
            corrected = _holm_correction(p_values)
        for row, corrected_p in zip(pair_rows, corrected, strict=False):
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
