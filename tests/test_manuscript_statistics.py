from __future__ import annotations

from pathlib import Path

import pandas as pd

from survarena.evaluation.statistics import aggregate_rank_summary, elo_ratings, pairwise_win_rate
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
