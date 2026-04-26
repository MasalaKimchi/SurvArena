from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from survarena.evaluation.statistics import failure_summary, metric_direction, summarize_frame
from survarena.logging.export_shared import (
    BENCHMARK_METRIC_COLUMNS,
    CORE_METRIC_COLUMNS,
    EFFICIENCY_COLUMNS,
    GOVERNANCE_COLUMNS,
    MANUSCRIPT_METRIC_COLUMNS,
    MANUSCRIPT_REPORT_SCHEMA_VERSION,
    RUN_LEDGER_COMPACT_SCHEMA_VERSION,
    RUN_LEDGER_SCHEMA_VERSION,
    benchmark_label,
    expand_dynamic_metric_columns,
    group_keys_with_hpo_mode,
    parity_gated_frame,
    unique_in_order,
)
from survarena.logging.ledger_export import export_run_ledger
from survarena.logging.manuscript_export import export_manuscript_comparison
from survarena.logging.navigator_export import export_experiment_navigator
from survarena.logging.tracker import write_json

__all__ = [
    "BENCHMARK_METRIC_COLUMNS",
    "CORE_METRIC_COLUMNS",
    "EFFICIENCY_COLUMNS",
    "GOVERNANCE_COLUMNS",
    "MANUSCRIPT_METRIC_COLUMNS",
    "MANUSCRIPT_REPORT_SCHEMA_VERSION",
    "RUN_LEDGER_COMPACT_SCHEMA_VERSION",
    "RUN_LEDGER_SCHEMA_VERSION",
    "create_experiment_dir",
    "export_dataset_curation_table",
    "export_experiment_navigator",
    "export_fold_results",
    "export_hpo_trials",
    "export_leaderboard",
    "export_manuscript_comparison",
    "export_overall_summary",
    "export_run_diagnostics",
    "export_run_ledger",
    "export_seed_summary",
]

# Backward-compatible private helper aliases for callers/tests that imported them from this module.
_benchmark_label = benchmark_label
_expand_dynamic_metric_columns = expand_dynamic_metric_columns
_group_keys_with_hpo_mode = group_keys_with_hpo_mode
_parity_gated_frame = parity_gated_frame
_unique_in_order = unique_in_order


def create_experiment_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "results" / "summary" / f"exp_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


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
        prefix = file_prefix or benchmark_label(frame)
        output = output_dir / f"{prefix}_fold_results.csv"
    frame.to_csv(output, index=False)
    return frame


def export_seed_summary(
    root: Path,
    frame: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
    write_file: bool = True,
) -> pd.DataFrame:
    by_cols = group_keys_with_hpo_mode(
        frame,
        ["benchmark_id", "dataset_id", "method_id", "seed"],
    )
    metric_cols = unique_in_order(BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + expand_dynamic_metric_columns(frame))
    available_metric_cols = [col for col in metric_cols if col in frame.columns]
    seed_summary = frame.groupby(by_cols, as_index=False)[available_metric_cols].mean(numeric_only=True)
    if not write_file:
        return seed_summary
    if output_dir is None:
        output = root / "results" / "summaries" / "seed_summary.csv"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or benchmark_label(frame)
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
    metric_cols = unique_in_order(BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + expand_dynamic_metric_columns(frame))
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
        prefix = file_prefix or benchmark_label(frame)
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
    write_json_output: bool = True,
) -> pd.DataFrame:
    metric_cols = unique_in_order(
        BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + expand_dynamic_metric_columns(seed_summary)
    )
    available_metric_cols = [col for col in metric_cols if col in seed_summary.columns]
    lb_keys = group_keys_with_hpo_mode(
        seed_summary,
        ["benchmark_id", "dataset_id", "method_id"],
    )
    leaderboard = seed_summary.groupby(lb_keys, as_index=False)[available_metric_cols].mean(numeric_only=True)
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
        prefix = file_prefix or benchmark_label(seed_summary)
        csv_path = output_dir / f"{prefix}_leaderboard.csv"
        json_path = output_dir / f"{prefix}_leaderboard.json"
    leaderboard.to_csv(csv_path, index=False)
    if write_json_output:
        leaderboard.to_json(json_path, orient="records", indent=2)
    return leaderboard


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


def export_hpo_trials(
    root: Path,
    rows: list[dict],
    *,
    benchmark_id: str,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if output_dir is None:
        output = root / "results" / "summaries" / f"{benchmark_id}_hpo_trials.csv"
        summary_output = root / "results" / "summaries" / f"{benchmark_id}_hpo_summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{benchmark_id}_hpo_trials.csv"
        summary_output = output_dir / f"{benchmark_id}_hpo_summary.json"
    frame.to_csv(output, index=False)
    if frame.empty:
        summary = {"benchmark_id": benchmark_id, "trial_count": 0, "methods": []}
    else:
        method_summary = (
            frame.groupby(["benchmark_id", "method_id"], as_index=False)
            .agg(
                trial_count=("trial_number", "count"),
                best_trial_value=("value", "max"),
            )
            .to_dict(orient="records")
        )
        summary = {
            "benchmark_id": benchmark_id,
            "trial_count": int(len(frame)),
            "methods": method_summary,
        }
    write_json(summary_output, summary)
    return frame


def export_run_diagnostics(
    root: Path,
    *,
    benchmark_id: str,
    fold_results: pd.DataFrame,
    dataset_curation_rows: list[dict],
    hpo_trial_rows: list[dict],
    output_dir: Path | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not fold_results.empty:
        failure_keys = group_keys_with_hpo_mode(
            fold_results,
            ["benchmark_id", "dataset_id", "method_id"],
        )
        failure_keys = [key for key in failure_keys if key in fold_results.columns]
        for row in failure_summary(fold_results).to_dict(orient="records"):
            rows.append({"record_type": "failure_summary", **row})
        if failure_keys and "status" in fold_results.columns:
            for row in (
                fold_results.groupby(failure_keys, as_index=False)
                .agg(n_runs=("status", "count"), n_success=("status", lambda values: int((values == "success").sum())))
                .to_dict(orient="records")
            ):
                n_runs = int(row.get("n_runs", 0) or 0)
                n_success = int(row.get("n_success", 0) or 0)
                rows.append(
                    {
                        "record_type": "run_summary",
                        **row,
                        "n_failed": n_runs - n_success,
                        "failure_rate": float((n_runs - n_success) / max(n_runs, 1)),
                    }
                )
    for row in dataset_curation_rows:
        rows.append({"record_type": "dataset_curation", "benchmark_id": benchmark_id, **row})
    for row in hpo_trial_rows:
        rows.append({"record_type": "hpo_trial", **row})

    frame = pd.DataFrame(rows)
    if output_dir is None:
        output = root / "results" / "summaries" / f"{benchmark_id}_run_diagnostics.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{benchmark_id}_run_diagnostics.csv"
    frame.to_csv(output, index=False)
    return frame
