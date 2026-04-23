from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from survarena.evaluation.statistics import (
    aggregate_rank_summary,
    bootstrap_metric_ci,
    elo_ratings,
    failure_summary,
    metric_direction,
    pairwise_win_rate,
    summarize_frame,
)
from survarena.logging.tracker import write_json, write_jsonl_gz

RUN_LEDGER_SCHEMA_VERSION = "2.0"
MANUSCRIPT_REPORT_SCHEMA_VERSION = "2.0"

CORE_METRIC_COLUMNS = [
    "validation_score",
    "uno_c",
    "harrell_c",
    "ibs",
    "td_auc_25",
    "td_auc_50",
    "td_auc_75",
]
MANUSCRIPT_METRIC_COLUMNS = [
    "brier_25",
    "brier_50",
    "brier_75",
    "calibration_slope_50",
    "calibration_intercept_50",
    "net_benefit_50",
]
EFFICIENCY_COLUMNS = [
    "tuning_time_sec",
    "runtime_sec",
    "fit_time_sec",
    "infer_time_sec",
    "peak_memory_mb",
]
BENCHMARK_METRIC_COLUMNS = CORE_METRIC_COLUMNS + MANUSCRIPT_METRIC_COLUMNS + EFFICIENCY_COLUMNS


def create_experiment_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "results" / "summary" / f"exp_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _benchmark_label(frame: pd.DataFrame, fallback: str = "benchmark") -> str:
    if "benchmark_id" not in frame.columns or frame.empty:
        return fallback
    unique = frame["benchmark_id"].dropna().astype(str).unique().tolist()
    if len(unique) == 1 and unique[0]:
        return unique[0]
    return fallback


def export_fold_results(
    root: Path,
    records: list[dict],
    *,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    sort_cols = [
        col
        for col in ["benchmark_id", "dataset_id", "method_id", "seed", "split_id"]
        if col in frame.columns
    ]
    if sort_cols:
        frame.sort_values(sort_cols, inplace=True)
    if output_dir is None:
        output = root / "results" / "tables" / "fold_results.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or _benchmark_label(frame)
        output = output_dir / f"{prefix}_fold_results.csv"
    frame.to_csv(output, index=False)
    return frame


def export_seed_summary(
    root: Path,
    frame: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    by_cols = ["benchmark_id", "dataset_id", "method_id", "seed"]
    metric_cols = BENCHMARK_METRIC_COLUMNS
    available_metric_cols = [col for col in metric_cols if col in frame.columns]
    seed_summary = frame.groupby(by_cols, as_index=False)[available_metric_cols].mean(numeric_only=True)
    if output_dir is None:
        output = root / "results" / "summaries" / "seed_summary.csv"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or _benchmark_label(frame)
        output = output_dir / f"{prefix}_seed_summary.csv"
    seed_summary.to_csv(output, index=False)
    return seed_summary


def export_overall_summary(
    root: Path,
    frame: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> dict:
    by_cols = ["benchmark_id", "dataset_id", "method_id"]
    metric_cols = BENCHMARK_METRIC_COLUMNS
    summary: dict[str, dict] = {}

    for key, sub in frame.groupby(by_cols):
        k = "__".join(str(v) for v in key)
        summary[k] = summarize_frame(sub, [col for col in metric_cols if col in sub.columns])
        summary[k]["n_runs"] = int(len(sub))
        summary[k]["n_success"] = int((sub.get("status", pd.Series(dtype=str)) == "success").sum())
        summary[k]["failure_rate"] = float(1.0 - summary[k]["n_success"] / max(len(sub), 1))

    if output_dir is None:
        output = root / "results" / "summaries" / "overall_summary.json"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or _benchmark_label(frame)
        output = output_dir / f"{prefix}_overall_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def export_leaderboard(
    root: Path,
    seed_summary: pd.DataFrame,
    primary_metric: str = "harrell_c",
    *,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    metric_cols = BENCHMARK_METRIC_COLUMNS
    available_metric_cols = [col for col in metric_cols if col in seed_summary.columns]
    leaderboard = seed_summary.groupby(["benchmark_id", "dataset_id", "method_id"], as_index=False)[
        available_metric_cols
    ].mean(numeric_only=True)
    if primary_metric not in leaderboard.columns:
        raise ValueError(f"Primary metric '{primary_metric}' not found in leaderboard columns.")
    primary_metric_ascending = metric_direction(primary_metric) == "minimize"
    leaderboard.sort_values(
        by=["dataset_id", primary_metric, "runtime_sec"],
        ascending=[True, primary_metric_ascending, True],
        inplace=True,
    )

    if output_dir is None:
        csv_path = root / "results" / "tables" / "leaderboard.csv"
        json_path = root / "results" / "summaries" / "leaderboard.json"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or _benchmark_label(seed_summary)
        csv_path = output_dir / f"{prefix}_leaderboard.csv"
        json_path = output_dir / f"{prefix}_leaderboard.json"
    leaderboard.to_csv(csv_path, index=False)
    leaderboard.to_json(json_path, orient="records", indent=2)
    return leaderboard


def export_manuscript_comparison(
    root: Path,
    leaderboard: pd.DataFrame,
    *,
    primary_metric: str,
    fold_results: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> dict[str, str]:
    prefix = file_prefix or _benchmark_label(leaderboard)
    if output_dir is None:
        output_dir = root / "results" / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_summary = aggregate_rank_summary(leaderboard, metric=primary_metric)
    pairwise = pairwise_win_rate(leaderboard, metric=primary_metric)
    ci = bootstrap_metric_ci(leaderboard, metric=primary_metric, n_bootstrap=1000, seed=0)
    elo = elo_ratings(leaderboard, metric=primary_metric)
    failures = failure_summary(fold_results if fold_results is not None else leaderboard)
    missing_metric_rows = []
    metric_cols = CORE_METRIC_COLUMNS[1:] + MANUSCRIPT_METRIC_COLUMNS
    for metric in [col for col in metric_cols if col in leaderboard.columns]:
        missing_metric_rows.append(
            {
                "metric": metric,
                "missing_rate": float(leaderboard[metric].isna().mean()) if len(leaderboard) else float("nan"),
            }
        )
    missing = pd.DataFrame(missing_metric_rows)

    paths = {
        "rank_summary": str(output_dir / f"{prefix}_rank_summary.csv"),
        "pairwise_win_rate": str(output_dir / f"{prefix}_pairwise_win_rate.csv"),
        "bootstrap_ci": str(output_dir / f"{prefix}_bootstrap_ci.csv"),
        "elo_ratings": str(output_dir / f"{prefix}_elo_ratings.csv"),
        "failure_summary": str(output_dir / f"{prefix}_failure_summary.csv"),
        "missing_metric_summary": str(output_dir / f"{prefix}_missing_metric_summary.csv"),
    }
    rank_summary.to_csv(paths["rank_summary"], index=False)
    pairwise.to_csv(paths["pairwise_win_rate"], index=False)
    ci.to_csv(paths["bootstrap_ci"], index=False)
    elo.to_csv(paths["elo_ratings"], index=False)
    failures.to_csv(paths["failure_summary"], index=False)
    missing.to_csv(paths["missing_metric_summary"], index=False)
    write_json(
        output_dir / f"{prefix}_manuscript_summary.json",
        {
            "schema_version": MANUSCRIPT_REPORT_SCHEMA_VERSION,
            "primary_metric": primary_metric,
            "comparison_files": paths,
            "rank_summary_records": rank_summary.to_dict(orient="records"),
            "elo_rating_records": elo.to_dict(orient="records"),
            "missing_metric_summary": missing.to_dict(orient="records"),
        },
    )
    return paths


def export_dataset_curation_table(
    root: Path,
    rows: list[dict],
    *,
    benchmark_id: str,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if output_dir is None:
        output = root / "results" / "summaries" / f"{benchmark_id}_dataset_curation.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{benchmark_id}_dataset_curation.csv"
    frame.to_csv(output, index=False)
    return frame


def export_run_ledger(
    root: Path,
    run_records: list[dict],
    *,
    benchmark_id: str,
    output_dir: Path | None = None,
) -> None:
    created_at = datetime.now().isoformat(timespec="seconds")
    normalized_records = [
        {
            "schema_version": RUN_LEDGER_SCHEMA_VERSION,
            **record,
        }
        for record in run_records
    ]
    if output_dir is None:
        output = root / "results" / "runs" / f"{benchmark_id}_run_records.jsonl.gz"
        index_output = root / "results" / "runs" / f"{benchmark_id}_run_records_index.json"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{benchmark_id}_run_records.jsonl.gz"
        index_output = output_dir / f"{benchmark_id}_run_records_index.json"
    write_jsonl_gz(output, normalized_records)
    write_json(
        index_output,
        {
            "schema_version": RUN_LEDGER_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "record_count": len(normalized_records),
            "format": "jsonl.gz",
            "path": str(output),
            "created_at": created_at,
            "record_sections": ["schema_version", "manifest", "metrics", "backend_metadata", "failure"],
        },
    )
