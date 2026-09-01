from __future__ import annotations

from typing import Iterable

import pandas as pd

from survarena.evaluation._comparison import build_comparison_population, dataset_method_scores, stratum_columns
from survarena.evaluation._metric_stats import metric_direction


def _require_complete(support_policy: str) -> bool:
    if support_policy not in {"complete", "available"}:
        raise ValueError("support_policy must be 'complete' or 'available'.")
    return support_policy == "complete"


def add_dataset_ranks(
    frame: pd.DataFrame,
    *,
    metric: str,
    required_methods: Iterable[str] | None = None,
    support_policy: str = "complete",
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    ascending = metric_direction(metric) == "minimize"
    ranked, _population = dataset_method_scores(
        frame,
        metric=metric,
        required_methods=required_methods,
        require_complete=_require_complete(support_policy),
    )
    rank_groups = [*stratum_columns(ranked), "dataset_id"]
    ranked[f"{metric}_rank"] = ranked.groupby(rank_groups)[metric].rank(
        method="average",
        ascending=ascending,
        na_option="keep",
    )
    ranked["support_policy"] = support_policy
    return ranked


def aggregate_rank_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    required_methods: Iterable[str] | None = None,
    support_policy: str = "complete",
) -> pd.DataFrame:
    ranked = add_dataset_ranks(
        frame,
        metric=metric,
        required_methods=required_methods,
        support_policy=support_policy,
    )
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
        cells_evaluated=("n_cells", "sum"),
        support_digest=("support_digest", "first"),
    )
    summary["support_policy"] = support_policy
    summary.sort_values(
        [c for c in ["benchmark_id", "hpo_mode", "mean_rank", "median_rank"] if c in summary.columns],
        inplace=True,
    )
    return summary.reset_index(drop=True)


def pairwise_win_rate(
    frame: pd.DataFrame,
    *,
    metric: str,
    required_methods: Iterable[str] | None = None,
    support_policy: str = "complete",
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    higher_is_better = metric_direction(metric) == "maximize"
    population = build_comparison_population(
        frame,
        metric=metric,
        required_methods=required_methods,
        require_complete=_require_complete(support_policy),
    )
    frame = population.frame
    rows: list[dict[str, object]] = []
    strata = stratum_columns(frame)
    grouper: str | list[str] = strata[0] if len(strata) == 1 else strata
    for stratum_key, sub in frame.groupby(grouper, sort=True):
        key_tuple = stratum_key if isinstance(stratum_key, tuple) else (stratum_key,)
        stratum = dict(zip(strata, key_tuple, strict=True))
        methods = sorted(sub["method_id"].astype(str).unique())
        merge_keys = list(population.cell_keys)
        for left_id in methods:
            for right_id in methods:
                if left_id == right_id:
                    continue
                left = sub[sub["method_id"].astype(str) == left_id][[*merge_keys, metric]].rename(
                    columns={metric: "left_metric"}
                )
                right = sub[sub["method_id"].astype(str) == right_id][[*merge_keys, metric]].rename(
                    columns={metric: "right_metric"}
                )
                matched = left.merge(right, on=merge_keys, how="inner", validate="one_to_one")
                if matched.empty:
                    continue
                left_scores = matched["left_metric"].to_numpy(dtype=float)
                right_scores = matched["right_metric"].to_numpy(dtype=float)
                win = left_scores > right_scores if higher_is_better else left_scores < right_scores
                tie = left_scores == right_scores
                matched = matched.assign(_win=win.astype(float), _tie=tie.astype(float))
                dataset_rates = matched.groupby("dataset_id", as_index=False).agg(
                    win_rate=("_win", "mean"),
                    tie_rate=("_tie", "mean"),
                )
                rows.append(
                    {
                        **stratum,
                        "method_id": left_id,
                        "opponent_method_id": right_id,
                        "wins": int(win.sum()),
                        "ties": int(tie.sum()),
                        "losses": int((~win & ~tie).sum()),
                        "win_rate": float(dataset_rates["win_rate"].mean()),
                        "tie_rate": float(dataset_rates["tie_rate"].mean()),
                        "n": int(len(matched)),
                        "n_datasets": int(matched["dataset_id"].nunique()),
                        "support_policy": support_policy,
                        "support_digest": population.support_digest,
                    }
                )
    if not rows:
        empty_cols = [
            *strata,
            "method_id",
            "opponent_method_id",
            "win_rate",
            "tie_rate",
            "n",
            "n_datasets",
            "support_policy",
            "support_digest",
        ]
        return pd.DataFrame(columns=empty_cols)
    return pd.DataFrame(rows)
