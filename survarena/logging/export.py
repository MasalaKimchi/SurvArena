from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from survarena.evaluation.statistics import failure_summary, metric_direction
from survarena.logging.export_shared import (
    BENCHMARK_METRIC_COLUMNS,
    CORE_METRIC_COLUMNS,
    EFFICIENCY_COLUMNS,
    GOVERNANCE_COLUMNS,
    MANUSCRIPT_METRIC_COLUMNS,
    benchmark_label,
    expand_dynamic_metric_columns,
    group_keys_with_hpo_mode,
    parity_gated_frame,
    unique_in_order,
)

__all__ = [
    "BENCHMARK_METRIC_COLUMNS",
    "CORE_METRIC_COLUMNS",
    "EFFICIENCY_COLUMNS",
    "GOVERNANCE_COLUMNS",
    "MANUSCRIPT_METRIC_COLUMNS",
    "create_experiment_dir",
    "export_fold_results",
    "export_leaderboard",
    "export_run_diagnostics",
]

# Backward-compatible private helper aliases for callers/tests that imported them from this module.
_benchmark_label = benchmark_label
_expand_dynamic_metric_columns = expand_dynamic_metric_columns
_group_keys_with_hpo_mode = group_keys_with_hpo_mode
_parity_gated_frame = parity_gated_frame
_unique_in_order = unique_in_order


def _slugify_component(value: str, *, fallback: str) -> str:
    normalized = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(value).strip())
    normalized = normalized.strip("_")
    return normalized or fallback


def _artifact_prefix(file_prefix: str | None, fallback: str) -> str:
    return _slugify_component(file_prefix or fallback, fallback=fallback)


def create_experiment_dir(
    root: Path,
    *,
    dataset_id: str | None = None,
    benchmark_id: str | None = None,
    model_name: str | None = None,
    run_stamp: str | None = None,
) -> Path:
    stamp = run_stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if dataset_id is None or benchmark_id is None or model_name is None:
        folder_name = f"exp_{stamp}"
        output_dir = root / "results" / "summary" / folder_name
    else:
        dataset_component = _slugify_component(dataset_id, fallback="dataset")
        benchmark_component = _slugify_component(benchmark_id or "benchmark", fallback="benchmark")
        model_component = _slugify_component(model_name, fallback="model")
        model_dir = root / "results" / "summary" / dataset_component / benchmark_component / model_component
        existing_csv = list(model_dir.glob("*.csv")) if model_dir.exists() else []
        output_dir = model_dir if not existing_csv else model_dir.with_name(f"{model_component}_{stamp}")
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
        prefix = _artifact_prefix(file_prefix, fallback="benchmark")
        output = output_dir / f"{prefix}_fold_results.csv"
    frame.to_csv(output, index=False)
    return frame


def export_leaderboard(
    root: Path,
    fold_results: pd.DataFrame,
    primary_metric: str = "harrell_c",
    *,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    requested_seed_keys = group_keys_with_hpo_mode(
        fold_results,
        ["benchmark_id", "dataset_id", "method_id", "seed"],
    )
    seed_keys = [col for col in requested_seed_keys if col in fold_results.columns]
    if not seed_keys:
        seed_keys = [col for col in ["dataset_id", "method_id", "seed"] if col in fold_results.columns]
    metric_cols = unique_in_order(BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + expand_dynamic_metric_columns(fold_results))
    available_metric_cols = [col for col in metric_cols if col in fold_results.columns]
    seed_summary = fold_results.groupby(seed_keys, as_index=False)[available_metric_cols].mean(numeric_only=True)
    metric_cols = unique_in_order(
        BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + expand_dynamic_metric_columns(seed_summary)
    )
    available_metric_cols = [col for col in metric_cols if col in seed_summary.columns]
    requested_lb_keys = group_keys_with_hpo_mode(
        seed_summary,
        ["benchmark_id", "dataset_id", "method_id"],
    )
    lb_keys = [col for col in requested_lb_keys if col in seed_summary.columns]
    if not lb_keys:
        lb_keys = [col for col in ["dataset_id", "method_id"] if col in seed_summary.columns]
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
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = _artifact_prefix(file_prefix, fallback=benchmark_label(seed_summary))
        csv_path = output_dir / f"{prefix}_leaderboard.csv"
    leaderboard.to_csv(csv_path, index=False)
    return leaderboard


def export_run_diagnostics(
    root: Path,
    *,
    benchmark_id: str,
    fold_results: pd.DataFrame,
    dataset_curation_rows: list[dict],
    hpo_trial_rows: list[dict],
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not fold_results.empty:
        requested_failure_keys = group_keys_with_hpo_mode(
            fold_results,
            ["benchmark_id", "dataset_id", "method_id"],
        )
        failure_keys = [key for key in requested_failure_keys if key in fold_results.columns]
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
        prefix = _artifact_prefix(file_prefix, fallback=benchmark_id)
        output = output_dir / f"{prefix}_run_diagnostics.csv"
    frame.to_csv(output, index=False)
    return frame
