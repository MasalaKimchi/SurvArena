from __future__ import annotations

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
from survarena.logging.export import export_manuscript_comparison


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
