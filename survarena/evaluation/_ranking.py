from __future__ import annotations

import numpy as np
import pandas as pd

from survarena.evaluation._metric_stats import metric_direction


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
        values = sub[["method_id", metric]].dropna().copy()
        values["method_id"] = values["method_id"].astype(str)
        method_values = {
            method_id: group[metric].to_numpy(dtype=float)
            for method_id, group in values.groupby("method_id", sort=True)
            if not group.empty
        }
        for left_id, left_scores in method_values.items():
            for right_id, right_scores in method_values.items():
                if left_id == right_id:
                    continue
                if left_scores.size == 0 or right_scores.size == 0:
                    continue
                left_matrix = left_scores[:, np.newaxis]
                right_matrix = right_scores[np.newaxis, :]
                win = left_matrix > right_matrix if higher_is_better else left_matrix < right_matrix
                tie = left_matrix == right_matrix
                n = int(win.size)
                row: dict[str, object] = {
                    "benchmark_id": benchmark_id,
                    "dataset_id": dataset_id,
                    "method_id": left_id,
                    "opponent_method_id": right_id,
                    "wins": float(np.sum(win)),
                    "ties": float(np.sum(tie)),
                    "n": n,
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
    grouped = pairwise.groupby(group_out, as_index=False).agg(
        wins=("wins", "sum"),
        ties=("ties", "sum"),
        n=("n", "sum"),
    )
    grouped["win_rate"] = grouped["wins"] / grouped["n"].replace(0, np.nan)
    grouped["tie_rate"] = grouped["ties"] / grouped["n"].replace(0, np.nan)
    return grouped[[*group_out, "win_rate", "tie_rate", "n"]]
