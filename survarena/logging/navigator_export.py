from __future__ import annotations

from pathlib import Path

import pandas as pd

from survarena.evaluation.statistics import metric_direction
from survarena.logging.tracker import write_json


def export_experiment_navigator(
    output_dir: Path,
    *,
    benchmark_id: str,
    file_prefix: str | None = None,
    primary_metric: str,
    split_count: int,
    method_count: int,
    leaderboard: pd.DataFrame,
) -> None:
    prefix = file_prefix or benchmark_id
    if primary_metric in leaderboard.columns:
        top_runs = leaderboard.sort_values(by=[primary_metric], ascending=metric_direction(primary_metric) == "minimize").head(5)
    else:
        top_runs = leaderboard.head(5)
    top_records = top_runs.to_dict(orient="records")
    core_candidates = [
        f"{prefix}_leaderboard.csv",
        f"{prefix}_overall_summary.json",
        f"{prefix}_manuscript_summary.json",
        f"{prefix}_dataset_curation.csv",
        f"{prefix}_run_records_compact_index.json",
        "experiment_manifest.json",
    ]
    detailed_candidates = [
        f"{prefix}_fold_results.csv",
        f"{prefix}_seed_summary.csv",
        f"{prefix}_leaderboard.json",
        f"{prefix}_report.csv",
        f"{prefix}_rank_summary.csv",
        f"{prefix}_pairwise_win_rate.csv",
        f"{prefix}_pairwise_significance.csv",
        f"{prefix}_multiple_comparison_summary.csv",
        f"{prefix}_critical_difference.csv",
        f"{prefix}_elo_ratings.csv",
        f"{prefix}_bootstrap_ci.csv",
        f"{prefix}_failure_summary.csv",
        f"{prefix}_missing_metric_summary.csv",
        f"{prefix}_run_records_compact.jsonl.gz",
        f"{prefix}_run_records_compact_index.json",
        f"{prefix}_run_records.jsonl.gz",
        f"{prefix}_run_records_index.json",
        f"{prefix}_hpo_trials.csv",
        f"{prefix}_hpo_summary.json",
    ]
    core_files = [name for name in core_candidates if (output_dir / name).exists()]
    detailed_files = [name for name in detailed_candidates if (output_dir / name).exists()]
    write_json(
        output_dir / "experiment_navigator.json",
        {
            "benchmark_id": benchmark_id,
            "primary_metric": primary_metric,
            "split_count": int(split_count),
            "method_count": int(method_count),
            "core_files": core_files,
            "detailed_files": detailed_files,
            "top_runs": top_records,
        },
    )
    lines = [
        "# Experiment Navigator",
        "",
        f"- benchmark_id: `{benchmark_id}`",
        f"- primary_metric: `{primary_metric}`",
        f"- split_count: `{int(split_count)}`",
        f"- method_count: `{int(method_count)}`",
        "",
        "## Start here (concise)",
        *(f"- `{name}`" for name in core_files),
        "",
        "## Detailed artifacts",
        *(f"- `{name}`" for name in detailed_files),
        "",
        "## Top runs",
    ]
    if top_records:
        for idx, record in enumerate(top_records, start=1):
            method_id = record.get("method_id", "unknown")
            dataset_id = record.get("dataset_id", "unknown")
            metric_value = record.get(primary_metric)
            lines.append(f"{idx}. `{dataset_id}/{method_id}` {primary_metric}={metric_value}")
    else:
        lines.append("- No leaderboard rows available.")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
