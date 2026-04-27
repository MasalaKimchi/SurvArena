from __future__ import annotations

import numpy as np
import pandas as pd

from survarena.evaluation._metric_stats import metric_direction


def _rating_from_score(score: float, *, initial_rating: float) -> float:
    clipped = float(np.clip(score, 1e-6, 1.0 - 1e-6))
    return float(initial_rating + 400.0 * np.log10(clipped / (1.0 - clipped)))


def _paired_rating_rows(
    frame: pd.DataFrame,
    *,
    metric: str,
    initial_rating: float,
) -> tuple[dict[str, float], dict[str, int]]:
    higher_is_better = metric_direction(metric) == "maximize"
    methods = sorted(frame["method_id"].dropna().astype(str).unique())
    points = {method: 0.0 for method in methods}
    match_count = {method: 0 for method in methods}
    unit_cols = [col for col in ["dataset_id", "split_id", "seed"] if col in frame.columns]
    if not unit_cols:
        unit_cols = ["method_id"]
    for _, sub in frame.groupby(unit_cols, sort=True):
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
                    left_points = 0.5
                else:
                    left_wins = left_score > right_score if higher_is_better else left_score < right_score
                    left_points = 1.0 if left_wins else 0.0
                points[left_id] += left_points
                points[right_id] += 1.0 - left_points
                match_count[left_id] += 1
                match_count[right_id] += 1
    ratings = {
        method: _rating_from_score(points[method] / match_count[method], initial_rating=initial_rating)
        if match_count[method] > 0
        else float(initial_rating)
        for method in methods
    }
    return ratings, match_count


def elo_ratings(
    frame: pd.DataFrame,
    *,
    metric: str,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative.")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    stratum_cols = ["benchmark_id"]
    if "hpo_mode" in frame.columns:
        stratum_cols.append("hpo_mode")
    for key, benchmark_frame in frame.groupby(stratum_cols, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        stratum = dict(zip(stratum_cols, key_tuple, strict=True))
        ratings, match_count = _paired_rating_rows(benchmark_frame, metric=metric, initial_rating=initial_rating)
        bootstrap_draws: dict[str, list[float]] = {method: [] for method in ratings}
        unit_cols = [col for col in ["dataset_id", "split_id", "seed"] if col in benchmark_frame.columns]
        if n_bootstrap > 0 and unit_cols:
            units = list(benchmark_frame.groupby(unit_cols, sort=True).groups)
            for _ in range(int(n_bootstrap)):
                sampled_units = [units[int(rng.integers(0, len(units)))] for _ in units]
                sampled = pd.concat(
                    [
                        benchmark_frame[
                            np.logical_and.reduce(
                                [
                                    benchmark_frame[col].eq(value)
                                    for col, value in zip(
                                        unit_cols,
                                        unit if isinstance(unit, tuple) else (unit,),
                                        strict=True,
                                    )
                                ]
                            )
                        ]
                        for unit in sampled_units
                    ],
                    ignore_index=True,
                )
                sampled_ratings, _sampled_matches = _paired_rating_rows(
                    sampled,
                    metric=metric,
                    initial_rating=initial_rating,
                )
                for method_id, rating in sampled_ratings.items():
                    bootstrap_draws.setdefault(method_id, []).append(rating)
        for method_id, rating in ratings.items():
            draws = np.asarray(bootstrap_draws.get(method_id, []), dtype=float)
            out_row: dict[str, object] = {
                **stratum,
                "method_id": method_id,
                "elo_rating": float(rating),
                "elo_matches": int(match_count[method_id]),
                "elo_rating_ci95_low": float(np.percentile(draws, 2.5)) if draws.size else float("nan"),
                "elo_rating_ci95_high": float(np.percentile(draws, 97.5)) if draws.size else float("nan"),
                "metric": metric,
                "initial_rating": float(initial_rating),
                "k_factor": float(k_factor),
                "rating_method": "paired_logit_winrate",
                "n_bootstrap": int(n_bootstrap),
            }
            rows.append(out_row)
    if not rows:
        cols = [
            "benchmark_id",
            "method_id",
            "elo_rating",
            "elo_matches",
            "elo_rating_ci95_low",
            "elo_rating_ci95_high",
            "metric",
            "initial_rating",
            "k_factor",
            "rating_method",
            "n_bootstrap",
        ]
        if "hpo_mode" in frame.columns:
            cols = [
                "benchmark_id",
                "hpo_mode",
                "method_id",
                "elo_rating",
                "elo_matches",
                "elo_rating_ci95_low",
                "elo_rating_ci95_high",
                "metric",
                "initial_rating",
                "k_factor",
                "rating_method",
                "n_bootstrap",
            ]
        return pd.DataFrame(columns=cols)
    result = pd.DataFrame(rows)
    sort_cols = [c for c in ["benchmark_id", "hpo_mode", "elo_rating", "method_id"] if c in result.columns]
    asc = [c != "elo_rating" for c in sort_cols]
    if "elo_rating" in sort_cols:
        ei = sort_cols.index("elo_rating")
        asc[ei] = False
    result.sort_values(sort_cols, ascending=asc, inplace=True)
    return result.reset_index(drop=True)
