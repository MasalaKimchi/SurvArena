from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from survarena.evaluation.statistics import (
    aggregate_rank_summary,
    critical_difference_summary,
    elo_ratings,
    pairwise_significance,
    pairwise_win_rate,
)
from survarena.logging.export import export_experiment_navigator, export_manuscript_comparison, export_run_ledger


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
    shuffled = elo_ratings(
        frame.sample(frac=1.0, random_state=0),
        metric="uno_c",
        n_bootstrap=50,
        seed=7,
    ).sort_values("method_id").reset_index(drop=True)

    np.testing.assert_allclose(first["elo_rating"], shuffled["elo_rating"])
    np.testing.assert_allclose(first["elo_rating_ci95_low"], shuffled["elo_rating_ci95_low"])
    assert (first["elo_rating_ci95_low"] <= first["elo_rating"]).all()
    assert (first["elo_rating"] <= first["elo_rating_ci95_high"]).all()


def test_export_manuscript_comparison_writes_summary_files(tmp_path: Path) -> None:
    frame = _leaderboard_frame()
    fold_results = frame.assign(status=["success", "failed", "success", "success"])

    paths = export_manuscript_comparison(
        tmp_path,
        frame,
        primary_metric="uno_c",
        fold_results=fold_results,
        output_dir=tmp_path,
        file_prefix="bench",
    )

    for path in paths.values():
        assert Path(path).exists()
    assert (tmp_path / "bench_manuscript_summary.json").exists()
    assert (tmp_path / "bench_elo_ratings.csv").exists()


def test_export_manuscript_comparison_compact_writes_report_and_figures(tmp_path: Path) -> None:
    frame = _leaderboard_frame()
    fold_results = frame.assign(status=["success", "success", "success", "success"], parity_eligible=[True] * 4)

    paths = export_manuscript_comparison(
        tmp_path,
        frame,
        primary_metric="uno_c",
        fold_results=fold_results,
        output_dir=tmp_path,
        file_prefix="cmp",
        artifact_layout="compact",
    )

    assert Path(paths["consolidated_report"]).exists()
    assert not (tmp_path / "cmp_pairwise_win_rate.csv").exists()
    assert (tmp_path / "cmp_manuscript_summary.json").exists()
    assert any("fig_pairwise" in k for k in paths)
    report = pd.read_csv(paths["consolidated_report"])
    assert "agg_elo_rating" in report.columns
    assert "agg_ci95_low_uno_c" in report.columns
    assert "agg_mean_rank" in report.columns


def test_export_run_ledger_defaults_to_compact_only(tmp_path: Path) -> None:
    export_run_ledger(
        tmp_path,
        [
            {
                "manifest": {"benchmark_id": "bench", "method_id": "a", "dataset_id": "d1"},
                "metrics": {"status": "success"},
                "failure": None,
            }
        ],
        benchmark_id="bench",
        output_dir=tmp_path,
    )

    assert (tmp_path / "bench_run_records_compact.jsonl.gz").exists()
    assert (tmp_path / "bench_run_records_compact_index.json").exists()
    assert not (tmp_path / "bench_run_records.jsonl.gz").exists()
    assert not (tmp_path / "bench_run_records_index.json").exists()


def test_export_run_ledger_can_write_full_compatibility_ledger(tmp_path: Path) -> None:
    export_run_ledger(
        tmp_path,
        [
            {
                "manifest": {"benchmark_id": "bench", "method_id": "a", "dataset_id": "d1"},
                "metrics": {"status": "success"},
                "failure": None,
            }
        ],
        benchmark_id="bench",
        output_dir=tmp_path,
        write_full_ledger=True,
    )

    assert (tmp_path / "bench_run_records_compact.jsonl.gz").exists()
    assert (tmp_path / "bench_run_records.jsonl.gz").exists()
    assert (tmp_path / "bench_run_records_index.json").exists()


def test_export_experiment_navigator_lists_existing_files_only(tmp_path: Path) -> None:
    leaderboard = pd.DataFrame(
        [{"benchmark_id": "bench", "dataset_id": "d1", "method_id": "a", "uno_c": 0.8}]
    )
    (tmp_path / "bench_leaderboard.csv").write_text("benchmark_id,dataset_id,method_id,uno_c\n", encoding="utf-8")
    (tmp_path / "bench_run_records_compact_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "experiment_manifest.json").write_text("{}", encoding="utf-8")

    export_experiment_navigator(
        tmp_path,
        benchmark_id="bench",
        primary_metric="uno_c",
        split_count=1,
        method_count=1,
        leaderboard=leaderboard,
    )

    navigator = json.loads((tmp_path / "experiment_navigator.json").read_text(encoding="utf-8"))
    assert "bench_leaderboard.csv" in navigator["core_files"]
    assert "bench_run_records_compact_index.json" in navigator["core_files"]
    assert "bench_run_records.jsonl.gz" not in navigator["detailed_files"]


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


def test_pairwise_significance_keeps_hpo_mode_strata_separate() -> None:
    frame = pd.DataFrame(
        [
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s1", "method_id": "a", "hpo_mode": "none", "uno_c": 0.8},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s1", "method_id": "b", "hpo_mode": "none", "uno_c": 0.7},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s2", "method_id": "a", "hpo_mode": "none", "uno_c": 0.81},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s2", "method_id": "b", "hpo_mode": "none", "uno_c": 0.72},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s1", "method_id": "a", "hpo_mode": "hpo", "uno_c": 0.75},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s1", "method_id": "c", "hpo_mode": "hpo", "uno_c": 0.77},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s2", "method_id": "a", "hpo_mode": "hpo", "uno_c": 0.76},
            {"benchmark_id": "b1", "dataset_id": "d1", "split_id": "s2", "method_id": "c", "hpo_mode": "hpo", "uno_c": 0.79},
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
