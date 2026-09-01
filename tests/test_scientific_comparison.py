from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survarena.evaluation._comparison import build_comparison_population, coverage_summary
from survarena.evaluation._ranking import add_dataset_ranks, pairwise_win_rate
from survarena.evaluation._ratings import elo_ratings
from survarena.evaluation._significance import (
    bootstrap_metric_ci,
    critical_difference_summary,
    pairwise_significance,
)


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


def test_support_digest_identifies_exact_comparison_cells() -> None:
    first = pd.DataFrame([_row("d1", "s1", "a", 0.8), _row("d1", "s1", "b", 0.7)])
    second = pd.DataFrame([_row("d1", "s2", "a", 0.8), _row("d1", "s2", "b", 0.7)])

    first_population = build_comparison_population(first, metric="uno_c")
    second_population = build_comparison_population(second, metric="uno_c")

    assert first_population.support_digest != second_population.support_digest


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


def test_pairwise_significance_uses_dataset_as_inferential_unit() -> None:
    rows: list[dict[str, object]] = []
    for split in range(15):
        rows.append(_row("d1", f"s{split}", "a", 0.8))
        rows.append(_row("d1", f"s{split}", "b", 0.6))

    result = pairwise_significance(pd.DataFrame(rows), metric="uno_c")
    a_vs_b = result[(result["method_id"] == "a") & (result["opponent_method_id"] == "b")].iloc[0]

    assert int(a_vs_b["n_datasets"]) == 1
    assert int(a_vs_b["n_matched_cells"]) == 15
    assert not bool(a_vs_b["testable"])
    assert a_vs_b["not_testable_reason"] == "insufficient_datasets"
    assert float(a_vs_b["p_value"]) == 1.0


def test_dataset_replication_does_not_change_pairwise_effect_or_p_value() -> None:
    rows: list[dict[str, object]] = []
    for dataset, a_score, b_score in (("d1", 0.8, 0.7), ("d2", 0.6, 0.65), ("d3", 0.9, 0.75)):
        rows.append(_row(dataset, "s1", "a", a_score))
        rows.append(_row(dataset, "s1", "b", b_score))
    base = pd.DataFrame(rows)
    replicated = [*rows]
    for split in range(2, 12):
        replicated.append(_row("d1", f"s{split}", "a", 0.8))
        replicated.append(_row("d1", f"s{split}", "b", 0.7))

    expected = pairwise_significance(base, metric="uno_c")
    actual = pairwise_significance(pd.DataFrame(replicated), metric="uno_c")
    expected_ab = expected[(expected["method_id"] == "a") & (expected["opponent_method_id"] == "b")].iloc[0]
    actual_ab = actual[(actual["method_id"] == "a") & (actual["opponent_method_id"] == "b")].iloc[0]

    assert int(actual_ab["n_datasets"]) == 3
    assert float(actual_ab["effect_size_mean_delta"]) == pytest.approx(
        float(expected_ab["effect_size_mean_delta"])
    )
    assert float(actual_ab["p_value"]) == pytest.approx(float(expected_ab["p_value"]))


def test_bootstrap_metric_ci_weights_dataset_means_equally() -> None:
    rows = [_row("d1", "s1", "a", 0.9), _row("d2", "s1", "a", 0.3)]
    for split in range(2, 12):
        rows.append(_row("d1", f"s{split}", "a", 0.9))

    result = bootstrap_metric_ci(pd.DataFrame(rows), metric="uno_c", n_bootstrap=100, seed=3).iloc[0]

    assert float(result["mean"]) == pytest.approx(0.6)
    assert float(result["median"]) == pytest.approx(0.6)
    assert int(result["n_datasets"]) == 2
    assert int(result["n_cells"]) == 12
    assert int(result["bootstrap_seed"]) == 3


def test_critical_difference_uses_only_complete_rectangular_dataset_block() -> None:
    frame = pd.DataFrame(
        [
            _row("d1", "s1", "a", 0.9),
            _row("d1", "s1", "b", 0.8),
            _row("d1", "s1", "c", 0.7),
            _row("d2", "s1", "a", 0.9),
            _row("d2", "s1", "b", 0.8),
        ]
    )

    result = critical_difference_summary(frame, metric="uno_c")

    assert set(result["method_id"]) == {"a", "b", "c"}
    assert (result["n_methods"] == 3).all()
    assert (result["n_datasets"] == 1).all()
    assert (result["n_datasets_total"] == 2).all()
    assert (result["n_datasets_excluded"] == 1).all()
    assert (result["assumption_status"] == "insufficient_datasets").all()
    assert result["critical_difference"].isna().all()


def test_critical_difference_is_emitted_only_after_significant_friedman() -> None:
    rows: list[dict[str, object]] = []
    for dataset_index in range(4):
        dataset_id = f"d{dataset_index}"
        rows.extend(
            [
                _row(dataset_id, "s1", "a", 0.9),
                _row(dataset_id, "s1", "b", 0.7),
                _row(dataset_id, "s1", "c", 0.5),
            ]
        )

    result = critical_difference_summary(pd.DataFrame(rows), metric="uno_c")

    assert (result["assumption_status"] == "complete_block").all()
    assert result["posthoc_eligible"].all()
    assert (result["critical_difference"] > 0.0).all()
