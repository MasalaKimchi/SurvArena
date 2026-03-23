from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.evaluation.statistics import summarize_frame
from src.logging.tracker import write_json, write_jsonl_gz


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
    metric_cols = [
        "validation_score",
        "uno_c",
        "harrell_c",
        "ibs",
        "td_auc_25",
        "td_auc_50",
        "td_auc_75",
        "tuning_time_sec",
        "runtime_sec",
        "fit_time_sec",
        "infer_time_sec",
        "peak_memory_mb",
    ]
    seed_summary = frame.groupby(by_cols, as_index=False)[metric_cols].mean(numeric_only=True)
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
    metric_cols = [
        "validation_score",
        "uno_c",
        "harrell_c",
        "ibs",
        "td_auc_25",
        "td_auc_50",
        "td_auc_75",
        "tuning_time_sec",
        "runtime_sec",
        "fit_time_sec",
        "infer_time_sec",
        "peak_memory_mb",
    ]
    summary: dict[str, dict] = {}

    for key, sub in frame.groupby(by_cols):
        k = "__".join(str(v) for v in key)
        summary[k] = summarize_frame(sub, metric_cols)
        summary[k]["n_runs"] = int(len(sub))

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
) -> None:
    metric_cols = [
        "validation_score",
        "uno_c",
        "harrell_c",
        "ibs",
        "tuning_time_sec",
        "runtime_sec",
        "fit_time_sec",
        "infer_time_sec",
        "peak_memory_mb",
    ]
    leaderboard = seed_summary.groupby(["benchmark_id", "dataset_id", "method_id"], as_index=False)[
        metric_cols
    ].mean(numeric_only=True)
    if primary_metric not in leaderboard.columns:
        raise ValueError(f"Primary metric '{primary_metric}' not found in leaderboard columns.")
    primary_metric_ascending = primary_metric in {"ibs"}
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


def export_run_ledger(
    root: Path,
    run_records: list[dict],
    *,
    benchmark_id: str,
    output_dir: Path | None = None,
) -> None:
    if output_dir is None:
        output = root / "results" / "runs" / f"{benchmark_id}_run_records.jsonl.gz"
        index_output = root / "results" / "runs" / f"{benchmark_id}_run_records_index.json"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{benchmark_id}_run_records.jsonl.gz"
        index_output = output_dir / f"{benchmark_id}_run_records_index.json"
    write_jsonl_gz(output, run_records)
    write_json(
        index_output,
        {
            "benchmark_id": benchmark_id,
            "record_count": len(run_records),
            "format": "jsonl.gz",
            "path": str(output),
        },
    )
