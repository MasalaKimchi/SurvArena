from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from survarena.evaluation._comparison import dataset_method_scores
from survarena.evaluation._metric_stats import metric_direction

# Number of passes over the head-to-head match set when fitting Elo ratings, and
# the fraction of trailing epochs averaged to damp update-order noise.
_ELO_EPOCHS = 100
_ELO_AVG_TAIL_FRACTION = 0.2


def _rating_from_score(score: float, *, initial_rating: float) -> float:
    """Legacy logit-of-win-rate transform (retained for backward compatibility).

    No longer used to produce the exported ratings; see ``_paired_rating_rows``
    for the true iterative Elo implementation.
    """
    clipped = float(np.clip(score, 1e-6, 1.0 - 1e-6))
    return float(initial_rating + 400.0 * np.log10(clipped / (1.0 - clipped)))


def _extract_matches(frame: pd.DataFrame, *, metric: str) -> tuple[list[str], list[tuple[str, str, float]], dict[str, int]]:
    """Build the head-to-head match list from paired per-unit results.

    Each evaluation unit (dataset x split x seed) contributes one match between
    every pair of methods that produced a score for that unit, with outcome 1.0
    (left better), 0.0 (right better), or 0.5 (tie) according to the metric's
    optimisation direction.
    """
    higher_is_better = metric_direction(metric) == "maximize"
    methods = sorted(frame["method_id"].dropna().astype(str).unique())
    match_count = {method: 0 for method in methods}
    matches: list[tuple[str, str, float]] = []
    unit_cols = [col for col in ["dataset_id", "split_id", "seed"] if col in frame.columns]
    if not unit_cols:
        unit_cols = ["method_id"]
    for _, sub in frame.groupby(unit_cols, sort=True):
        values = sub[["method_id", metric]].dropna().copy()
        values["method_id"] = values["method_id"].astype(str)
        # Sort records by method_id so the match list is a deterministic function
        # of the data alone, independent of input row order. Combined with the
        # seeded per-epoch shuffle, this makes the fitted ratings reproducible and
        # order-independent (required by the benchmark's reproducibility contract).
        values = values.sort_values("method_id", kind="stable")
        records = values.to_dict(orient="records")
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                left_id = str(left["method_id"])
                right_id = str(right["method_id"])
                left_score = float(left[metric])
                right_score = float(right[metric])
                if left_score == right_score:
                    s_left = 0.5
                else:
                    left_wins = left_score > right_score if higher_is_better else left_score < right_score
                    s_left = 1.0 if left_wins else 0.0
                matches.append((left_id, right_id, s_left))
                match_count[left_id] += 1
                match_count[right_id] += 1
    return methods, matches, match_count


def _paired_rating_rows(
    frame: pd.DataFrame,
    *,
    metric: str,
    initial_rating: float,
    k_factor: float = 32.0,
    n_epochs: int = _ELO_EPOCHS,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """Fit true iterative Elo ratings from head-to-head match outcomes.

    Unlike a logit transform of the raw win rate, Elo accounts for opponent
    strength: beating a strongly-rated method moves a rating more than beating a
    weak one, because the expected score is computed from the current rating gap
    (M12). Ratings are updated over ``n_epochs`` passes; matches are shuffled
    each epoch (seeded, for reproducibility) and the final ratings are averaged
    over the trailing epochs to damp update-order sensitivity.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    methods, matches, match_count = _extract_matches(frame, metric=metric)
    ratings = {method: float(initial_rating) for method in methods}
    if not matches:
        return ratings, match_count

    tail_epochs = max(1, int(round(n_epochs * _ELO_AVG_TAIL_FRACTION)))
    tail_accum = {method: 0.0 for method in methods}
    tail_seen = 0
    n_matches = len(matches)
    for epoch in range(int(n_epochs)):
        order = rng.permutation(n_matches)
        for idx in order:
            left_id, right_id, s_left = matches[idx]
            r_left = ratings[left_id]
            r_right = ratings[right_id]
            expected_left = 1.0 / (1.0 + 10.0 ** ((r_right - r_left) / 400.0))
            ratings[left_id] = r_left + k_factor * (s_left - expected_left)
            ratings[right_id] = r_right + k_factor * ((1.0 - s_left) - (1.0 - expected_left))
        if epoch >= int(n_epochs) - tail_epochs:
            for method in methods:
                tail_accum[method] += ratings[method]
            tail_seen += 1

    if tail_seen > 0:
        ratings = {method: tail_accum[method] / tail_seen for method in methods}
    return ratings, match_count


def _fit_elo_from_matches(
    matches: list[tuple[str, str, float]],
    *,
    methods: list[str],
    initial_rating: float,
    k_factor: float,
    n_epochs: int = _ELO_EPOCHS,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Fit iterative Elo ratings from a prebuilt match list.

    This is the epoch loop of ``_paired_rating_rows`` factored out so it can be
    driven directly from a caller-supplied match list and an explicit set of
    ``methods`` to initialise/return. The expected-score and update formulas, the
    seeded per-epoch shuffle (``rng.permutation``) and the trailing-epoch
    averaging are identical to ``_paired_rating_rows`` -- fitting the same match
    list with the same ``rng`` state yields the same ratings. The bootstrap uses
    it to refit each resampled draw in O(number of matches) with no DataFrame
    construction.
    """
    ratings = {method: float(initial_rating) for method in methods}
    if not matches:
        return ratings

    tail_epochs = max(1, int(round(n_epochs * _ELO_AVG_TAIL_FRACTION)))
    tail_accum = {method: 0.0 for method in methods}
    tail_seen = 0
    n_matches = len(matches)
    for epoch in range(int(n_epochs)):
        order = rng.permutation(n_matches)
        for idx in order:
            left_id, right_id, s_left = matches[idx]
            r_left = ratings[left_id]
            r_right = ratings[right_id]
            expected_left = 1.0 / (1.0 + 10.0 ** ((r_right - r_left) / 400.0))
            ratings[left_id] = r_left + k_factor * (s_left - expected_left)
            ratings[right_id] = r_right + k_factor * ((1.0 - s_left) - (1.0 - expected_left))
        if epoch >= int(n_epochs) - tail_epochs:
            for method in methods:
                tail_accum[method] += ratings[method]
            tail_seen += 1

    if tail_seen > 0:
        ratings = {method: tail_accum[method] / tail_seen for method in methods}
    return ratings


def _clustered_match_lists(
    frame: pd.DataFrame, *, metric: str
) -> tuple[dict[object, list[tuple[str, str, float]]], list[str]]:
    """Build head-to-head match lists grouped per evaluation unit but keyed by cluster.

    Matches are built exactly as in ``_extract_matches`` (one match per method
    pair per evaluation unit, same tie/win rule and same stable ``method_id``
    ordering so the result is independent of input row order). Instead of a flat
    list they are keyed by resampling cluster: the ``dataset_id`` value when that
    column exists, otherwise the full evaluation-unit key. This lets the Elo
    bootstrap resample whole datasets (dataset-clustered) using matches computed
    once, so each bootstrap iteration is O(number of matches).
    """
    higher_is_better = metric_direction(metric) == "maximize"
    methods = sorted(frame["method_id"].dropna().astype(str).unique())
    unit_cols = [col for col in ["dataset_id", "split_id", "seed"] if col in frame.columns]
    if not unit_cols:
        unit_cols = ["method_id"]
    has_dataset = "dataset_id" in frame.columns
    dataset_pos = unit_cols.index("dataset_id") if has_dataset else 0
    cluster_to_matches: dict[object, list[tuple[str, str, float]]] = {}
    for unit_key, sub in frame.groupby(unit_cols, sort=True):
        key_tuple = unit_key if isinstance(unit_key, tuple) else (unit_key,)
        cluster = key_tuple[dataset_pos] if has_dataset else key_tuple
        values = sub[["method_id", metric]].dropna().copy()
        values["method_id"] = values["method_id"].astype(str)
        # Sort records by method_id (stable) so matches depend only on the data,
        # mirroring _extract_matches for a byte-identical match list.
        values = values.sort_values("method_id", kind="stable")
        records = values.to_dict(orient="records")
        bucket = cluster_to_matches.setdefault(cluster, [])
        for left_index, left in enumerate(records):
            for right in records[left_index + 1 :]:
                left_id = str(left["method_id"])
                right_id = str(right["method_id"])
                left_score = float(left[metric])
                right_score = float(right[metric])
                if left_score == right_score:
                    s_left = 0.5
                else:
                    left_wins = left_score > right_score if higher_is_better else left_score < right_score
                    s_left = 1.0 if left_wins else 0.0
                bucket.append((left_id, right_id, s_left))
    return cluster_to_matches, methods


def elo_ratings(
    frame: pd.DataFrame,
    *,
    metric: str,
    initial_rating: float = 1500.0,
    k_factor: float = 32.0,
    n_bootstrap: int = 200,
    seed: int = 0,
    required_methods: Iterable[str] | None = None,
    support_policy: str = "complete",
) -> pd.DataFrame:
    if metric not in frame.columns:
        raise ValueError(f"Metric '{metric}' not found in frame.")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be non-negative.")
    if support_policy not in {"complete", "available"}:
        raise ValueError("support_policy must be 'complete' or 'available'.")
    # A dataset contributes one match per method pair. Repeated folds/seeds are
    # averaged within dataset first so they reduce measurement noise without
    # changing the dataset's weight in the rating population.
    frame, population = dataset_method_scores(
        frame,
        metric=metric,
        required_methods=required_methods,
        require_complete=support_policy == "complete",
    )
    # Point-estimate ratings are driven by ``rng``; the bootstrap uses a separate,
    # independent generator (``boot_rng``) so the reported ratings are invariant to
    # ``n_bootstrap``. If the bootstrap shared ``rng``, resampling in an earlier
    # stratum would advance the stream and shift a later stratum's point estimate.
    rng = np.random.default_rng(seed)
    boot_rng = np.random.default_rng(seed + 1_000_003)
    rows: list[dict[str, object]] = []
    stratum_cols = ["benchmark_id"]
    if "hpo_mode" in frame.columns:
        stratum_cols.append("hpo_mode")
    for key, benchmark_frame in frame.groupby(stratum_cols, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        stratum = dict(zip(stratum_cols, key_tuple, strict=True))
        ratings, match_count = _paired_rating_rows(
            benchmark_frame, metric=metric, initial_rating=initial_rating, k_factor=k_factor, rng=rng
        )
        n_datasets = int(benchmark_frame["dataset_id"].nunique())
        bootstrap_draws: dict[str, list[float]] = {method: [] for method in ratings}
        if n_bootstrap > 0:
            # Dataset-clustered bootstrap. Matches are built per evaluation unit
            # but keyed by dataset cluster and precomputed ONCE, so each iteration
            # only resamples cluster indices and concatenates their (already
            # built) match lists -- O(number of matches) with no per-iteration
            # DataFrame. The previous implementation rebuilt a frame via pd.concat
            # and re-grouped it every iteration; because a unit drawn c times
            # collapsed back into a single group, the all-pairs regrouping emitted
            # ~c^2 cross-method matches plus spurious within-method self-ties
            # instead of c linear copies, inflating the CI width. Here a cluster
            # drawn c times contributes its matches exactly c times (linear).
            cluster_to_matches, _bootstrap_methods = _clustered_match_lists(
                benchmark_frame, metric=metric
            )
            # Perf: precompute each cluster's method-id set ONCE (from the match
            # lists already built above) so the per-iteration ``present`` is a
            # cheap union over the drawn clusters instead of an O(#matches) rescan
            # of the concatenated draw. Output-preserving: ``boot_matches`` is
            # exactly the concatenation of the drawn clusters' match lists, so the
            # union of those clusters' method sets equals the set of ids appearing
            # in ``boot_matches`` -- ``present`` (and thus every downstream fit) is
            # identical.
            cluster_to_methods = {
                cluster: {m for a, b, _ in matches for m in (a, b)}
                for cluster, matches in cluster_to_matches.items()
            }
            clusters = sorted(cluster_to_matches)
            n_iters = int(n_bootstrap) if clusters else 0
            for _ in range(n_iters):
                idx = boot_rng.integers(0, len(clusters), size=len(clusters))
                boot_matches: list[tuple[str, str, float]] = []
                for j in idx:
                    boot_matches.extend(cluster_to_matches[clusters[int(j)]])
                if not boot_matches:
                    continue
                present = sorted(set().union(*(cluster_to_methods[clusters[int(j)]] for j in idx)))
                boot = _fit_elo_from_matches(
                    boot_matches,
                    methods=present,
                    initial_rating=initial_rating,
                    k_factor=k_factor,
                    rng=boot_rng,
                )
                for method_id, rating in boot.items():
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
                "rating_method": "iterative_elo",
                "n_bootstrap": int(n_bootstrap),
                "n_datasets": n_datasets,
                "support_policy": support_policy,
                "support_digest": population.support_digest,
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
            "n_datasets",
            "support_policy",
            "support_digest",
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
                "n_datasets",
                "support_policy",
                "support_digest",
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
