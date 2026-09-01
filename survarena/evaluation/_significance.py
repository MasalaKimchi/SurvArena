from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from survarena.evaluation._comparison import (
    build_comparison_population,
    dataset_method_scores,
    stratum_columns,
)
from survarena.evaluation._metric_stats import metric_direction


def _cluster_bootstrap_mean(
    sub: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap the mean metric resampling at the dataset (cluster) level.

    Per-fold values from the same dataset are correlated and datasets differ in
    difficulty, so an i.i.d. bootstrap over pooled folds understates uncertainty
    (M15). When a ``dataset_id`` column is present we resample whole datasets
    with replacement and take every retained fold of each drawn dataset, which
    is a clustered (block) bootstrap that captures between-dataset variance. We
    fall back to an i.i.d. bootstrap only when no cluster key is available.
    """
    if "dataset_id" in sub.columns:
        dataset_means = sub.groupby("dataset_id", sort=True)[metric].mean().dropna().to_numpy(dtype=float)
        if dataset_means.size == 0:
            return np.asarray([], dtype=float)
        n_clusters = int(dataset_means.size)
        draws = np.empty(int(n_bootstrap), dtype=float)
        for i in range(int(n_bootstrap)):
            picked = rng.integers(0, n_clusters, size=n_clusters)
            draws[i] = float(np.mean(dataset_means[picked]))
        return draws
    values = sub[metric].dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    return np.asarray(
        [np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(int(n_bootstrap))],
        dtype=float,
    )


def bootstrap_metric_ci(
    frame: pd.DataFrame,
    *,
    metric: str,
    n_bootstrap: int = 1000,
    seed: int = 0,
    required_methods: Iterable[str] | None = None,
    support_policy: str = "complete",
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    if support_policy not in {"complete", "available"}:
        raise ValueError("support_policy must be 'complete' or 'available'.")
    frame, population = dataset_method_scores(
        frame,
        metric=metric,
        required_methods=required_methods,
        require_complete=support_policy == "complete",
    )
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
                    "median": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "n": 0,
                    "n_datasets": 0,
                    "n_cells": 0,
                    "support_policy": support_policy,
                    "support_digest": population.support_digest,
                    "bootstrap_seed": seed,
                }
            )
            continue
        draws = _cluster_bootstrap_mean(sub, metric=metric, n_bootstrap=n_bootstrap, rng=rng)
        rows.append(
            {
                **row_base,
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "ci95_low": float(np.percentile(draws, 2.5)) if draws.size else float("nan"),
                "ci95_high": float(np.percentile(draws, 97.5)) if draws.size else float("nan"),
                "n": int(values.size),
                "n_datasets": int(values.size),
                "n_cells": int(sub["n_cells"].sum()),
                "support_policy": support_policy,
                "support_digest": population.support_digest,
                "bootstrap_seed": seed,
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
    required_methods: Iterable[str] | None = None,
    support_policy: str = "complete",
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    if correction not in {"holm", "bh"}:
        raise ValueError("correction must be 'holm' or 'bh'.")
    if support_policy not in {"complete", "available"}:
        raise ValueError("support_policy must be 'complete' or 'available'.")
    higher_is_better = metric_direction(metric) == "maximize"
    population = build_comparison_population(
        frame,
        metric=metric,
        required_methods=required_methods,
        require_complete=support_policy == "complete",
    )
    frame = population.frame
    rows: list[dict[str, object]] = []
    stratum_cols = stratum_columns(frame)
    merge_cols = list(population.cell_keys)
    # scipy is imported lazily so this module (and the rest of the stats/report
    # pipeline: eligibility, ranking, Elo, and the Nemenyi CD table fallback) stay
    # importable without scipy. Only the paired Wilcoxon test genuinely needs it.
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scipy is required for pairwise Wilcoxon significance testing. "
            "Install scipy, or generate the report without the significance table."
        ) from exc
    for key, stratum_sub in frame.groupby(stratum_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        stratum = dict(zip(stratum_cols, key_tuple, strict=True))
        pair_rows: list[dict[str, object]] = []
        pair_p_values: list[float] = []
        methods = sorted(stratum_sub["method_id"].dropna().astype(str).unique())
        for left_index, left_method in enumerate(methods):
            for right_method in methods[left_index + 1 :]:
                left = stratum_sub[stratum_sub["method_id"] == left_method][merge_cols + [metric]].rename(
                    columns={metric: "left_metric"}
                )
                right = stratum_sub[stratum_sub["method_id"] == right_method][merge_cols + [metric]].rename(
                    columns={metric: "right_metric"}
                )
                merged = left.merge(right, on=merge_cols, how="inner", validate="one_to_one").dropna()
                if merged.empty:
                    continue
                cell_delta = (
                    merged["left_metric"].to_numpy(dtype=float) - merged["right_metric"].to_numpy(dtype=float)
                    if higher_is_better
                    else merged["right_metric"].to_numpy(dtype=float) - merged["left_metric"].to_numpy(dtype=float)
                )
                dataset_delta = (
                    pd.DataFrame({"dataset_id": merged["dataset_id"].to_numpy(), "delta": cell_delta})
                    .groupby("dataset_id", sort=True)["delta"]
                    .mean()
                    .to_numpy(dtype=float)
                )
                testable = dataset_delta.size >= 2 and not bool(np.allclose(dataset_delta, 0.0))
                not_testable_reason = ""
                if dataset_delta.size < 2:
                    not_testable_reason = "insufficient_datasets"
                elif bool(np.allclose(dataset_delta, 0.0)):
                    not_testable_reason = "all_zero_differences"
                if not testable:
                    p_value = 1.0
                else:
                    try:
                        # Two-sided paired Wilcoxon signed-rank test: the null is
                        # "no difference between the two methods". A two-sided
                        # p-value is symmetric, so the same value legitimately
                        # applies to both directed rows of the pair; the DIRECTION
                        # of any difference is conveyed by effect_size_mean_delta
                        # and the win counts. (Fixes C1: the previous one-sided
                        # "greater" test reused an identical p-value for the
                        # reversed row, which encoded the opposite, wrong-direction
                        # hypothesis.)
                        p_value = float(
                            stats.wilcoxon(dataset_delta, alternative="two-sided", zero_method="wilcox").pvalue
                        )
                    except ValueError:
                        p_value = 1.0
                        testable = False
                        not_testable_reason = "wilcoxon_undefined"
                effect_size = float(np.mean(dataset_delta))
                median_effect = float(np.median(dataset_delta))
                pair_index = len(pair_p_values)
                pair_p_values.append(p_value)
                pair_rows.append(
                    {
                        **stratum,
                        "_pair_index": pair_index,
                        "method_id": left_method,
                        "opponent_method_id": right_method,
                        "metric": metric,
                        "n_pairs": int(dataset_delta.size),
                        "n_datasets": int(dataset_delta.size),
                        "n_matched_cells": int(cell_delta.size),
                        "effect_size_mean_delta": effect_size,
                        "effect_size_median_delta": median_effect,
                        "p_value": p_value,
                        "wins": int(np.sum(dataset_delta > 0)),
                        "ties": int(np.sum(dataset_delta == 0)),
                        "losses": int(np.sum(dataset_delta < 0)),
                        "testable": testable,
                        "not_testable_reason": not_testable_reason,
                        "support_policy": support_policy,
                        "support_digest": population.support_digest,
                    }
                )
                pair_rows.append(
                    {
                        **stratum,
                        "_pair_index": pair_index,
                        "method_id": right_method,
                        "opponent_method_id": left_method,
                        "metric": metric,
                        "n_pairs": int(dataset_delta.size),
                        "n_datasets": int(dataset_delta.size),
                        "n_matched_cells": int(cell_delta.size),
                        "effect_size_mean_delta": float(-effect_size),
                        "effect_size_median_delta": float(-median_effect),
                        "p_value": p_value,
                        "wins": int(np.sum(dataset_delta < 0)),
                        "ties": int(np.sum(dataset_delta == 0)),
                        "losses": int(np.sum(dataset_delta > 0)),
                        "testable": testable,
                        "not_testable_reason": not_testable_reason,
                        "support_policy": support_policy,
                        "support_digest": population.support_digest,
                    }
                )
        if not pair_p_values:
            continue
        if correction == "bh":
            corrected = _benjamini_hochberg(pair_p_values)
        else:
            corrected = _holm_correction(pair_p_values)
        for row in pair_rows:
            pair_index = int(row.pop("_pair_index"))
            rows.append({**row, "p_value_corrected": float(corrected[pair_index]), "correction": correction})
    return pd.DataFrame(rows)


# Nemenyi q_alpha values for alpha=0.05, indexed by number of methods k.
# Each value is the Studentized-range critical value q_{0.05, k, inf} divided by
# sqrt(2), which is the constant used in the Nemenyi critical-difference formula
# CD = q_alpha * sqrt(k(k+1) / (6N)). Used only as a fallback when
# scipy.stats.studentized_range is unavailable.
_NEMENYI_Q05: dict[int, float] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
    10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
    17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
}


def _nemenyi_q_alpha(n_methods: int, *, alpha: float = 0.05) -> float:
    """Nemenyi critical value for ``n_methods`` methods (not a fixed constant).

    The previous implementation hardcoded 2.569, which is correct only for
    exactly four methods (H4). We derive the value exactly from the Studentized
    range distribution when SciPy exposes it, and otherwise fall back to a table.
    """
    k = max(int(n_methods), 2)
    try:
        from scipy.stats import studentized_range

        q = float(studentized_range.ppf(1.0 - alpha, k, np.inf)) / np.sqrt(2.0)
        if np.isfinite(q):
            return q
    except Exception:  # noqa: BLE001 - fall back to the table on any SciPy issue
        pass
    if k in _NEMENYI_Q05:
        return _NEMENYI_Q05[k]
    # Clamp for out-of-table k (only reachable without studentized_range).
    return _NEMENYI_Q05[min(max(k, 2), 20)]


def critical_difference_summary(
    frame: pd.DataFrame,
    *,
    metric: str,
    required_methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    # Friedman/Nemenyi require exactly one observation per method per dataset
    # (block), so we FIRST collapse folds/seeds to a per-(dataset, method) mean
    # metric, THEN rank methods 1..k within each dataset (H5). Ranking pooled
    # fold rows would produce ranks on a 1..(folds*methods) scale that is
    # inconsistent with the CD formula's assumptions. Only eligible, successful
    # cells contribute (M4).
    ascending = metric_direction(metric) == "minimize"
    cell_means, population = dataset_method_scores(
        frame,
        metric=metric,
        required_methods=required_methods,
        require_complete=True,
    )
    stratum_cols = stratum_columns(frame)
    output_columns = [
        *stratum_cols,
        "method_id",
        "average_rank",
        "critical_difference",
        "n_methods",
        "n_datasets",
        "n_datasets_total",
        "n_datasets_excluded",
        "metric",
        "friedman_p_value",
        "posthoc_eligible",
        "assumption_status",
        "support_policy",
        "support_digest",
    ]
    if cell_means.empty:
        return pd.DataFrame(columns=output_columns)

    rank_col = f"{metric}_rank"
    cell_means[rank_col] = cell_means.groupby([*stratum_cols, "dataset_id"])[metric].rank(
        method="average",
        ascending=ascending,
        na_option="keep",
    )

    rows: list[dict[str, object]] = []
    for key, sub in cell_means.groupby(stratum_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        stratum = dict(zip(stratum_cols, key_tuple, strict=True))
        avg = sub.groupby("method_id", as_index=False)[rank_col].mean()
        n_datasets = int(sub["dataset_id"].nunique())
        n_methods = int(avg["method_id"].nunique())
        coverage = population.coverage
        for column, value in stratum.items():
            coverage = coverage[coverage[column] == value]
        n_datasets_total = int(coverage["dataset_id"].nunique())
        n_datasets_excluded = n_datasets_total - n_datasets
        if n_methods < 3:
            assumption_status = "insufficient_methods"
        elif n_datasets < 2:
            assumption_status = "insufficient_datasets"
        else:
            assumption_status = "complete_block"
        friedman_p_value = float("nan")
        if assumption_status == "complete_block":
            matrix = sub.pivot(index="dataset_id", columns="method_id", values=metric)
            try:
                from scipy import stats

                friedman_p_value = float(stats.friedmanchisquare(*(matrix[column] for column in matrix.columns)).pvalue)
            except ValueError:
                assumption_status = "friedman_undefined"
        q_alpha = _nemenyi_q_alpha(n_methods)
        posthoc_eligible = assumption_status == "complete_block" and friedman_p_value < 0.05
        cd = (
            float(q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_datasets)))
            if posthoc_eligible
            else float("nan")
        )
        for row in avg.to_dict(orient="records"):
            rows.append(
                {
                    **stratum,
                    "method_id": str(row["method_id"]),
                    "average_rank": float(row[rank_col]),
                    "critical_difference": cd,
                    "n_methods": n_methods,
                    "n_datasets": n_datasets,
                    "n_datasets_total": n_datasets_total,
                    "n_datasets_excluded": n_datasets_excluded,
                    "metric": metric,
                    "friedman_p_value": friedman_p_value,
                    "posthoc_eligible": posthoc_eligible,
                    "assumption_status": assumption_status,
                    "support_policy": "complete",
                    "support_digest": population.support_digest,
                }
            )
    return pd.DataFrame(rows, columns=output_columns)
