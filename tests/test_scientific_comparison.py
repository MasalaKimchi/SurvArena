from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survarena.evaluation._comparison import build_comparison_population, coverage_summary
from survarena.evaluation._ranking import add_dataset_ranks, pairwise_win_rate
from survarena.evaluation._ratings import elo_ratings


def _row(
    dataset_id: str,
    split_id: str,
    method_id: str,
    score: float,
    *,
    status: str = "success",
    comparison_ineligible: bool = False,
    ineligible_reason: str = "",
) -> dict[str, object]:
    return {
        "benchmark_id": "bench",
        "hpo_mode": "no_hpo",
        "dataset_id": dataset_id,
        "split_id": split_id,
        "seed": 11,
        "method_id": method_id,
        "uno_c": score,
        "status": status,
        "comparison_ineligible": comparison_ineligible,
        "ineligible_reason": ineligible_reason,
    }


def test_pairwise_win_rate_joins_exact_cells_instead_of_cross_product() -> None:
    frame = pd.DataFrame(
        [
            _row("d1", "s1", "a", 0.90),
            _row("d1", "s2", "a", 0.10),
            _row("d1", "s1", "b", 0.80),
            _row("d1", "s2", "b", 0.20),
        ]
    )

    result = pairwise_win_rate(frame, metric="uno_c")
    a_vs_b = result[(result["method_id"] == "a") & (result["opponent_method_id"] == "b")].iloc[0]

    assert int(a_vs_b["n"]) == 2
    assert int(a_vs_b["n_datasets"]) == 1
    assert float(a_vs_b["win_rate"]) == pytest.approx(0.5)


def test_duplicate_method_cell_identity_fails_closed() -> None:
    frame = pd.DataFrame(
        [
            _row("d1", "s1", "a", 0.8),
            _row("d1", "s1", "a", 0.81),
            _row("d1", "s1", "b", 0.7),
        ]
    )

    with pytest.raises(ValueError, match="Duplicate comparison cell"):
        build_comparison_population(frame, metric="uno_c")


def test_common_support_excludes_incomplete_dataset_and_reports_missing_method() -> None:
    frame = pd.DataFrame(
        [
            _row("d1", "s1", "a", 0.8),
            _row("d1", "s1", "b", 0.7),
            _row("d2", "s1", "a", 0.9),
        ]
    )

    population = build_comparison_population(frame, metric="uno_c")

    assert set(population.frame["dataset_id"]) == {"d1"}
    missing = population.coverage[
        (population.coverage["dataset_id"] == "d2") & (population.coverage["method_id"] == "b")
    ].iloc[0]
    assert int(missing["n_attempted"]) == 0
    assert int(missing["n_missing_cells"]) == 1
    assert not bool(missing["dataset_on_common_support"])


def test_coverage_separates_failure_invalid_fallback_and_timeout() -> None:
    frame = pd.DataFrame(
        [
            _row("d1", "s1", "a", 0.8),
            _row("d1", "s2", "a", np.nan, status="timeout"),
            _row("d1", "s3", "a", 0.5, comparison_ineligible=True, ineligible_reason="invalid_prediction"),
            _row("d1", "s4", "a", 0.5, comparison_ineligible=True, ineligible_reason="degenerate_fallback"),
        ]
    )

    row = coverage_summary(frame, metric="uno_c", required_methods=["a"]).iloc[0]

    assert int(row["n_attempted"]) == 4
    assert int(row["n_successful"]) == 3
    assert int(row["n_eligible"]) == 1
    assert int(row["n_timeout"]) == 1
    assert int(row["n_invalid_prediction"]) == 1
    assert int(row["n_fallback"]) == 1
    assert int(row["n_failed"]) == 1


def test_dataset_ranks_are_method_ranks_not_pooled_fold_ranks() -> None:
    frame = pd.DataFrame(
        [
            _row("d1", "s1", "a", 0.9),
            _row("d1", "s2", "a", 0.8),
            _row("d1", "s1", "b", 0.7),
            _row("d1", "s2", "b", 0.6),
            _row("d1", "s1", "c", 0.5),
            _row("d1", "s2", "c", 0.4),
        ]
    )

    ranked = add_dataset_ranks(frame, metric="uno_c").set_index("method_id")

    assert len(ranked) == 3
    assert ranked["uno_c_rank"].to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_elo_point_rating_is_invariant_to_fold_replication_within_dataset() -> None:
    base = pd.DataFrame(
        [
            _row("d1", "s1", "a", 0.8),
            _row("d1", "s1", "b", 0.7),
            _row("d2", "s1", "a", 0.6),
            _row("d2", "s1", "b", 0.65),
        ]
    )
    replicated_rows = [*base.to_dict(orient="records")]
    for split_id in ("s2", "s3", "s4", "s5"):
        for method_id, score in (("a", 0.8), ("b", 0.7)):
            replicated_rows.append(_row("d1", split_id, method_id, score))
    replicated = pd.DataFrame(replicated_rows)

    expected = elo_ratings(base, metric="uno_c", n_bootstrap=0, seed=9).set_index("method_id")
    actual = elo_ratings(replicated, metric="uno_c", n_bootstrap=0, seed=9).set_index("method_id")

    np.testing.assert_allclose(actual.loc[expected.index, "elo_rating"], expected["elo_rating"])
    assert (actual["n_datasets"] == 2).all()
