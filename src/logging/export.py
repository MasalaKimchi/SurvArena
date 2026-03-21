from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.statistics import summarize_frame
from src.logging.tracker import write_json, write_jsonl_gz


def export_fold_results(root: Path, records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    output = root / "results" / "tables" / "fold_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def export_seed_summary(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    by_cols = ["benchmark_id", "dataset_id", "method_id", "seed"]
    metric_cols = [
        "uno_c",
        "harrell_c",
        "ibs",
        "td_auc_25",
        "td_auc_50",
        "td_auc_75",
        "fit_time_sec",
        "infer_time_sec",
        "peak_memory_mb",
    ]
    seed_summary = frame.groupby(by_cols, as_index=False)[metric_cols].mean(numeric_only=True)
    output = root / "results" / "summaries" / "seed_summary.csv"
    seed_summary.to_csv(output, index=False)
    return seed_summary


def export_overall_summary(root: Path, frame: pd.DataFrame) -> dict:
    by_cols = ["benchmark_id", "dataset_id", "method_id"]
    metric_cols = ["uno_c", "harrell_c", "ibs", "td_auc_25", "td_auc_50", "td_auc_75"]
    summary: dict[str, dict] = {}

    for key, sub in frame.groupby(by_cols):
        k = "__".join(str(v) for v in key)
        summary[k] = summarize_frame(sub, metric_cols)
        summary[k]["n_runs"] = int(len(sub))

    output = root / "results" / "summaries" / "overall_summary.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def export_leaderboard(root: Path, seed_summary: pd.DataFrame, primary_metric: str = "harrell_c") -> None:
    metric_cols = ["uno_c", "harrell_c", "ibs"]
    leaderboard = seed_summary.groupby(["benchmark_id", "dataset_id", "method_id"], as_index=False)[
        metric_cols
    ].mean(numeric_only=True)
    if primary_metric not in leaderboard.columns:
        raise ValueError(f"Primary metric '{primary_metric}' not found in leaderboard columns.")
    leaderboard.sort_values(by=["dataset_id", primary_metric], ascending=[True, False], inplace=True)

    csv_path = root / "results" / "tables" / "leaderboard.csv"
    json_path = root / "results" / "summaries" / "leaderboard.json"
    leaderboard.to_csv(csv_path, index=False)
    leaderboard.to_json(json_path, orient="records", indent=2)


def export_run_ledger(
    root: Path,
    run_records: list[dict],
    *,
    benchmark_id: str,
) -> None:
    output = root / "results" / "runs" / f"{benchmark_id}_run_records.jsonl.gz"
    write_jsonl_gz(output, run_records)
    write_json(
        root / "results" / "runs" / f"{benchmark_id}_run_records_index.json",
        {
            "benchmark_id": benchmark_id,
            "record_count": len(run_records),
            "format": "jsonl.gz",
            "path": str(output),
        },
    )
