from __future__ import annotations

import math
import numpy as np
import pytest
from survarena.benchmark import tuning
from survarena.evaluation.metrics import (
    MetricBundle,
    _event_status_at_horizons,
    compute_primary_metric_score,
    compute_survival_metrics,
    horizons_from_train_event_times,
)
import pandas as pd
from survarena.evaluation import _significance
from survarena.evaluation.statistics import (
    aggregate_rank_summary,
    critical_difference_summary,
    elo_ratings,
    pairwise_significance,
    pairwise_win_rate,
)


# --- test_metrics_and_tuning.py ---


def test_metric_bundle_to_dict_casts_all_metrics_to_plain_floats() -> None:
    bundle = MetricBundle(
        uno_c=np.float32(0.61),
        harrell_c=np.float64(0.63),
        ibs=np.float32(0.18),
        td_auc_25=np.float32(0.66),
        td_auc_50=np.float32(0.67),
        td_auc_75=np.float32(0.68),
    )

    as_dict = bundle.to_dict()

    assert {key: as_dict[key] for key in ["uno_c", "harrell_c", "ibs", "td_auc_25", "td_auc_50", "td_auc_75"]} == {
        "uno_c": 0.6100000143051147,
        "harrell_c": 0.63,
        "ibs": 0.18000000715255737,
        "td_auc_25": 0.6600000262260437,
        "td_auc_50": 0.6700000166893005,
        "td_auc_75": 0.6800000071525574,
    }
    for key in [
        "brier_25",
        "brier_50",
        "brier_75",
        "calibration_slope_50",
        "calibration_intercept_50",
        "net_benefit_50",
    ]:
        assert math.isnan(as_dict[key])
    assert all(isinstance(value, float) for value in as_dict.values())


def test_compute_primary_metric_score_dispatches_to_harrell(monkeypatch) -> None:
    called: dict[str, np.ndarray] = {}

    def fake_harrell(**kwargs) -> float:
        called.update(kwargs)
        return 0.73

    monkeypatch.setattr("survarena.evaluation.metrics.compute_harrell_c_index", fake_harrell)

    score = compute_primary_metric_score(
        primary_metric="harrell_c",
        train_time=np.asarray([1.0, 2.0]),
        train_event=np.asarray([1, 0]),
        eval_time=np.asarray([1.5, 2.5]),
        eval_event=np.asarray([1, 0]),
        eval_risk_scores=np.asarray([0.2, 0.4]),
    )

    assert score == 0.73
    np.testing.assert_allclose(called["eval_time"], np.asarray([1.5, 2.5]))


def test_compute_primary_metric_score_dispatches_to_uno(monkeypatch) -> None:
    called: dict[str, np.ndarray] = {}

    def fake_uno(**kwargs) -> float:
        called.update(kwargs)
        return 0.69

    monkeypatch.setattr("survarena.evaluation.metrics.compute_uno_c_index", fake_uno)

    score = compute_primary_metric_score(
        primary_metric="uno_c",
        train_time=np.asarray([1.0, 2.0]),
        train_event=np.asarray([1, 0]),
        eval_time=np.asarray([1.5, 2.5]),
        eval_event=np.asarray([1, 0]),
        eval_risk_scores=np.asarray([0.2, 0.4]),
    )

    assert score == 0.69
    np.testing.assert_allclose(called["train_time"], np.asarray([1.0, 2.0]))


def test_compute_primary_metric_score_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="Unsupported primary metric"):
        compute_primary_metric_score(
            primary_metric="ibs",
            train_time=np.asarray([1.0, 2.0]),
            train_event=np.asarray([1, 0]),
            eval_time=np.asarray([1.5, 2.5]),
            eval_event=np.asarray([1, 0]),
            eval_risk_scores=np.asarray([0.2, 0.4]),
        )


def test_horizons_from_train_event_times_uses_defaults_without_events() -> None:
    horizons = horizons_from_train_event_times(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0, 0, 0]),
    )

    assert horizons == (1.0, 2.0, 3.0)


def test_horizons_from_train_event_times_uses_event_quantiles() -> None:
    horizons = horizons_from_train_event_times(
        np.asarray([1.0, 2.0, 4.0, 8.0]),
        np.asarray([0, 1, 1, 1]),
    )

    assert horizons == (3.0, 4.0, 6.0)


def test_horizon_event_status_marks_censored_before_horizon_unknown() -> None:
    observed, known = _event_status_at_horizons(
        np.asarray([1.0, 2.0, 4.0]),
        np.asarray([0, 1, 0]),
        (3.0, 5.0, 6.0),
    )

    assert observed[:, 0].tolist() == [0.0, 1.0, 0.0]
    assert known[:, 0].tolist() == [False, True, True]
    assert known[:, 1].tolist() == [False, True, False]


def test_selection_helpers_strip_runtime_only_defaults() -> None:
    assert tuning.resolve_runtime_method_params({"alpha": 0.1, "seed": 3}, seed=11) == {"alpha": 0.1, "seed": 11}
    assert tuning._searchable_default_params({"default_params": {"alpha": 0.1, "seed": 3}}) == {"alpha": 0.1}


def test_metric_bundle_float_conversion_stays_finite_for_normal_values() -> None:
    values = MetricBundle(0.6, 0.7, 0.2, 0.65, 0.66, 0.67).to_dict()

    assert math.isfinite(values["uno_c"])
    assert math.isfinite(values["harrell_c"])


def test_compute_survival_metrics_trims_survival_grid_to_ipcw_support() -> None:
    metrics = compute_survival_metrics(
        train_time=np.asarray([1.0, 2.0, 3.0, 10.0]),
        train_event=np.asarray([1, 1, 1, 0]),
        test_time=np.asarray([1.5, 2.5]),
        test_event=np.asarray([1, 0]),
        risk_scores=np.asarray([0.8, 0.2]),
        survival_probs=np.asarray(
            [
                [0.95, 0.82, 0.75, 0.65],
                [0.98, 0.92, 0.86, 0.78],
            ]
        ),
        survival_times=np.asarray([1.0, 2.0, 3.0, 9.0]),
        horizons=(1.5, 2.0, 2.5),
    ).to_dict()

    assert math.isfinite(metrics["harrell_c"])
    assert "ibs" in metrics


def test_time_dependent_auc_uses_horizon_specific_event_probabilities() -> None:
    train_time = np.asarray([1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0], dtype=float)
    train_event = np.asarray([1, 1, 1, 1, 1, 1, 0], dtype=int)
    test_time = np.asarray([1.5, 2.5, 3.5, 6.5, 7.5, 8.5], dtype=float)
    test_event = np.asarray([1, 1, 1, 0, 0, 0], dtype=int)
    risk_scores = np.asarray([0.9, 0.8, 0.7, 0.1, 0.2, 0.3], dtype=float)
    survival_times = np.linspace(1.0, 8.0, 8)
    aligned_survival = np.vstack(
        [
            np.linspace(0.9, 0.2, 8),
            np.linspace(0.9, 0.25, 8),
            np.linspace(0.9, 0.3, 8),
            np.linspace(0.98, 0.8, 8),
            np.linspace(0.98, 0.78, 8),
            np.linspace(0.98, 0.76, 8),
        ]
    )
    reversed_survival = aligned_survival[::-1].copy()

    aligned = compute_survival_metrics(
        train_time=train_time,
        train_event=train_event,
        test_time=test_time,
        test_event=test_event,
        risk_scores=risk_scores,
        survival_probs=aligned_survival,
        survival_times=survival_times,
        horizons=(2.0, 3.0, 4.0),
    ).to_dict()
    reversed_metrics = compute_survival_metrics(
        train_time=train_time,
        train_event=train_event,
        test_time=test_time,
        test_event=test_event,
        risk_scores=risk_scores,
        survival_probs=reversed_survival,
        survival_times=survival_times,
        horizons=(2.0, 3.0, 4.0),
    ).to_dict()

    assert aligned["td_auc_50"] > reversed_metrics["td_auc_50"]
    assert aligned["brier_50"] < reversed_metrics["brier_50"]


def test_compute_survival_metrics_handles_duplicate_clipped_horizons() -> None:
    metrics = compute_survival_metrics(
        train_time=np.asarray([1.0, 2.0, 3.0, 10.0]),
        train_event=np.asarray([1, 1, 1, 0]),
        test_time=np.asarray([1.5, 2.5]),
        test_event=np.asarray([1, 0]),
        risk_scores=np.asarray([0.8, 0.2]),
        survival_probs=np.asarray(
            [
                [0.95, 0.82, 0.75, 0.65],
                [0.98, 0.92, 0.86, 0.78],
            ]
        ),
        survival_times=np.asarray([1.0, 2.0, 3.0, 9.0]),
        horizons=(4.0, 5.0, 6.0),
    ).to_dict()

    assert math.isfinite(metrics["td_auc_25"])
    assert metrics["td_auc_25"] == metrics["td_auc_50"] == metrics["td_auc_75"]
    assert metrics["horizon_used_25"] == metrics["horizon_used_50"] == metrics["horizon_used_75"]


# --- test_statistics.py ---


def _leaderboard_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"benchmark_id": "bench", "dataset_id": "d1", "method_id": "a", "uno_c": 0.80, "runtime_sec": 1.0},
            {"benchmark_id": "bench", "dataset_id": "d1", "method_id": "b", "uno_c": 0.70, "runtime_sec": 2.0},
            {"benchmark_id": "bench", "dataset_id": "d2", "method_id": "a", "uno_c": 0.60, "runtime_sec": 1.0},
            {"benchmark_id": "bench", "dataset_id": "d2", "method_id": "b", "uno_c": 0.65, "runtime_sec": 2.0},
        ]
    )


def test_aggregate_rank_summary_and_pairwise_win_rate() -> None:
    frame = _leaderboard_frame()

    ranks = aggregate_rank_summary(frame, metric="uno_c").set_index("method_id")
    wins = pairwise_win_rate(frame, metric="uno_c")
    elo = elo_ratings(frame, metric="uno_c").set_index("method_id")

    assert float(ranks.loc["a", "mean_rank"]) == 1.5
    assert float(ranks.loc["b", "mean_rank"]) == 1.5
    a_vs_b = wins[(wins["method_id"] == "a") & (wins["opponent_method_id"] == "b")].iloc[0]
    assert float(a_vs_b["win_rate"]) == 0.5
    assert set(elo.index) == {"a", "b"}
    assert int(elo.loc["a", "elo_matches"]) == 2
    assert {"elo_rating_ci95_low", "elo_rating_ci95_high", "rating_method"}.issubset(elo.columns)


def test_elo_ratings_are_order_independent_with_bootstrap_ci() -> None:
    frame = pd.DataFrame(
        [
            {"benchmark_id": "bench", "dataset_id": "d1", "method_id": "a", "uno_c": 0.80},
            {"benchmark_id": "bench", "dataset_id": "d1", "method_id": "b", "uno_c": 0.70},
            {"benchmark_id": "bench", "dataset_id": "d1", "method_id": "c", "uno_c": 0.60},
            {"benchmark_id": "bench", "dataset_id": "d2", "method_id": "a", "uno_c": 0.71},
            {"benchmark_id": "bench", "dataset_id": "d2", "method_id": "b", "uno_c": 0.73},
            {"benchmark_id": "bench", "dataset_id": "d2", "method_id": "c", "uno_c": 0.62},
            {"benchmark_id": "bench", "dataset_id": "d3", "method_id": "a", "uno_c": 0.83},
            {"benchmark_id": "bench", "dataset_id": "d3", "method_id": "b", "uno_c": 0.78},
            {"benchmark_id": "bench", "dataset_id": "d3", "method_id": "c", "uno_c": 0.76},
        ]
    )

    first = elo_ratings(frame, metric="uno_c", n_bootstrap=50, seed=7).sort_values("method_id").reset_index(drop=True)
    shuffled = (
        elo_ratings(
            frame.sample(frac=1.0, random_state=0),
            metric="uno_c",
            n_bootstrap=50,
            seed=7,
        )
        .sort_values("method_id")
        .reset_index(drop=True)
    )

    np.testing.assert_allclose(first["elo_rating"], shuffled["elo_rating"])
    np.testing.assert_allclose(first["elo_rating_ci95_low"], shuffled["elo_rating_ci95_low"])
    assert (first["elo_rating_ci95_low"] <= first["elo_rating"]).all()
    assert (first["elo_rating"] <= first["elo_rating_ci95_high"]).all()


def test_elo_ratings_treat_foundation_methods_as_methods() -> None:
    frame = pd.DataFrame(
        [
            {
                "benchmark_id": "manuscript_v1",
                "dataset_id": "d1",
                "split_id": "s1",
                "seed": 11,
                "hpo_mode": "no_hpo",
                "method_id": "tabpfn_survival",
                "uno_c": 0.72,
            },
            {
                "benchmark_id": "manuscript_v1",
                "dataset_id": "d1",
                "split_id": "s1",
                "seed": 11,
                "hpo_mode": "no_hpo",
                "method_id": "rsf",
                "uno_c": 0.70,
            },
            {
                "benchmark_id": "manuscript_v1",
                "dataset_id": "d1",
                "split_id": "s1",
                "seed": 11,
                "hpo_mode": "no_hpo",
                "method_id": "mitra_survival_frozen",
                "uno_c": 0.74,
            },
        ]
    )

    result = elo_ratings(frame, metric="uno_c", n_bootstrap=0)

    assert set(result["hpo_mode"]) == {"no_hpo"}
    no_hpo = result[result["hpo_mode"] == "no_hpo"]
    assert set(no_hpo["method_id"]) == {
        "tabpfn_survival",
        "mitra_survival_frozen",
        "rsf",
    }
    assert int(no_hpo.set_index("method_id").loc["tabpfn_survival", "elo_matches"]) == 2


def test_pairwise_significance_produces_corrected_p_values() -> None:
    rows = []
    rng = np.random.default_rng(0)
    for seed in [11, 22, 33, 44, 55]:
        for split_idx in range(3):
            rows.append(
                {
                    "benchmark_id": "b1",
                    "dataset_id": "d1",
                    "split_id": f"s{split_idx}",
                    "seed": seed,
                    "method_id": "m_strong",
                    "uno_c": float(0.75 + 0.02 * rng.random()),
                }
            )
            rows.append(
                {
                    "benchmark_id": "b1",
                    "dataset_id": "d1",
                    "split_id": f"s{split_idx}",
                    "seed": seed,
                    "method_id": "m_weak",
                    "uno_c": float(0.55 + 0.02 * rng.random()),
                }
            )
    frame = pd.DataFrame(rows)
    result = pairwise_significance(frame, metric="uno_c", correction="holm")
    assert not result.empty
    assert {"p_value", "p_value_corrected", "effect_size_mean_delta"}.issubset(result.columns)
    assert (result["p_value_corrected"] <= 1.0).all()


def test_pairwise_significance_corrects_unique_pairs_before_mirroring(monkeypatch) -> None:
    captured: dict[str, list[float]] = {}

    def fake_holm(p_values: list[float]) -> list[float]:
        captured["p_values"] = list(p_values)
        return [0.11 + index for index, _value in enumerate(p_values)]

    monkeypatch.setattr(_significance, "_holm_correction", fake_holm)
    frame = pd.DataFrame(
        [
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s1", "method_id": "a", "uno_c": 0.8},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s1", "method_id": "b", "uno_c": 0.7},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s1", "method_id": "c", "uno_c": 0.6},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s2", "method_id": "a", "uno_c": 0.81},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s2", "method_id": "b", "uno_c": 0.71},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s2", "method_id": "c", "uno_c": 0.61},
        ]
    )

    result = pairwise_significance(frame, metric="uno_c", correction="holm")

    assert len(captured["p_values"]) == 3
    assert len(result) == 6
    ab = result[(result["method_id"] == "a") & (result["opponent_method_id"] == "b")].iloc[0]
    ba = result[(result["method_id"] == "b") & (result["opponent_method_id"] == "a")].iloc[0]
    assert float(ab["p_value_corrected"]) == float(ba["p_value_corrected"])


def test_pairwise_significance_keeps_hpo_mode_strata_separate() -> None:
    frame = pd.DataFrame(
        [
            {
                "benchmark_id": "b1",
                "dataset_id": "d1",
                "split_id": "s1",
                "method_id": "a",
                "hpo_mode": "none",
                "uno_c": 0.8,
            },
            {
                "benchmark_id": "b1",
                "dataset_id": "d1",
                "split_id": "s1",
                "method_id": "b",
                "hpo_mode": "none",
                "uno_c": 0.7,
            },
            {
                "benchmark_id": "b1",
                "dataset_id": "d1",
                "split_id": "s2",
                "method_id": "a",
                "hpo_mode": "none",
                "uno_c": 0.81,
            },
            {
                "benchmark_id": "b1",
                "dataset_id": "d1",
                "split_id": "s2",
                "method_id": "b",
                "hpo_mode": "none",
                "uno_c": 0.72,
            },
            {
                "benchmark_id": "b1",
                "dataset_id": "d1",
                "split_id": "s1",
                "method_id": "a",
                "hpo_mode": "hpo",
                "uno_c": 0.75,
            },
            {
                "benchmark_id": "b1",
                "dataset_id": "d1",
                "split_id": "s1",
                "method_id": "c",
                "hpo_mode": "hpo",
                "uno_c": 0.77,
            },
            {
                "benchmark_id": "b1",
                "dataset_id": "d1",
                "split_id": "s2",
                "method_id": "a",
                "hpo_mode": "hpo",
                "uno_c": 0.76,
            },
            {
                "benchmark_id": "b1",
                "dataset_id": "d1",
                "split_id": "s2",
                "method_id": "c",
                "hpo_mode": "hpo",
                "uno_c": 0.79,
            },
        ]
    )

    result = pairwise_significance(frame, metric="uno_c")

    assert set(result["hpo_mode"]) == {"none", "hpo"}
    assert not ((result["method_id"] == "b") & (result["opponent_method_id"] == "c")).any()
    assert not ((result["method_id"] == "c") & (result["opponent_method_id"] == "b")).any()
    assert (result.groupby("hpo_mode")["p_value_corrected"].count() == 2).all()


def test_critical_difference_summary_contains_cd() -> None:
    frame = pd.DataFrame(
        [
            {"benchmark_id": "b1", "dataset_id": "d1", "method_id": "a", "uno_c": 0.8},
            {"benchmark_id": "b1", "dataset_id": "d1", "method_id": "b", "uno_c": 0.7},
            {"benchmark_id": "b1", "dataset_id": "d2", "method_id": "a", "uno_c": 0.79},
            {"benchmark_id": "b1", "dataset_id": "d2", "method_id": "b", "uno_c": 0.68},
        ]
    )
    result = critical_difference_summary(frame, metric="uno_c")
    assert not result.empty
    assert (result["critical_difference"] > 0).all()
