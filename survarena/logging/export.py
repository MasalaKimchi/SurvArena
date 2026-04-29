from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

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
    unique_in_order,
)

__all__ = [
    "BENCHMARK_METRIC_COLUMNS",
    "CORE_METRIC_COLUMNS",
    "EFFICIENCY_COLUMNS",
    "GOVERNANCE_COLUMNS",
    "MANUSCRIPT_METRIC_COLUMNS",
    "create_experiment_dir",
    "export_coverage_matrix",
    "export_fold_results",
    "export_leaderboard",
    "export_run_diagnostics",
    "export_runtime_failure_summary",
]

_RUNTIME_FAILURE_METRIC_COLUMNS = [
    "validation_score",
    "uno_c",
    "harrell_c",
    "ibs",
    "td_auc_25",
    "td_auc_50",
    "td_auc_75",
]


def _slugify_component(value: str, *, fallback: str) -> str:
    normalized = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in str(value).strip())
    normalized = normalized.strip("_")
    return normalized or fallback


def _artifact_prefix(file_prefix: str | None, fallback: str) -> str:
    return _slugify_component(file_prefix or fallback, fallback=fallback)


def _artifact_paths(
    root: Path,
    *,
    output_dir: Path | None,
    file_prefix: str | None,
    fallback: str,
    stem: str,
) -> tuple[Path, Path]:
    if output_dir is None:
        base_dir = root / "results" / "tables"
        prefix = fallback
    else:
        base_dir = output_dir
        prefix = _artifact_prefix(file_prefix, fallback=fallback)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / f"{prefix}_{stem}.csv", base_dir / f"{prefix}_{stem}.md"


def _parse_split_geometry(split_id: object) -> tuple[int | None, int | None]:
    base_split = str(split_id).split("__", 1)[0]
    repeated = re.search(r"repeat_(?P<repeat>\d+)_fold_(?P<fold>\d+)", base_split)
    if repeated:
        return int(repeated.group("repeat")), int(repeated.group("fold"))
    fixed = re.search(r"fixed_split_(?P<fold>\d+)", base_split)
    if fixed:
        return 0, int(fixed.group("fold"))
    return None, None


def _write_markdown_table(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        if frame.empty:
            handle.write("No rows were exported.\n")
            return
        markdown_frame = frame.astype(object).where(pd.notna(frame), "")
        columns = [str(col) for col in markdown_frame.columns]
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in markdown_frame.itertuples(index=False, name=None):
            values = [str(value).replace("|", "\\|") for value in row]
            handle.write("| " + " | ".join(values) + " |\n")
        handle.write("\n")


def _markdown_table_lines(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["No rows were exported."]
    markdown_frame = frame.astype(object).where(pd.notna(frame), "")
    columns = [str(col) for col in markdown_frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in markdown_frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def create_experiment_dir(
    root: Path,
    *,
    dataset_id: str | None = None,
    benchmark_id: str | None = None,
    model_name: str | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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


def export_coverage_matrix(
    root: Path,
    fold_results: pd.DataFrame,
    primary_metric: str,
    *,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    """Export one row per evaluated dataset-method-mode-seed-fold unit."""
    csv_path, md_path = _artifact_paths(
        root,
        output_dir=output_dir,
        file_prefix=file_prefix,
        fallback=benchmark_label(fold_results),
        stem="coverage_matrix",
    )
    if fold_results.empty:
        coverage = pd.DataFrame()
        coverage.to_csv(csv_path, index=False)
        _write_markdown_table(coverage, md_path, title="Coverage Matrix")
        return coverage

    frame = fold_results.copy()
    if "hpo_mode" not in frame.columns:
        frame["hpo_mode"] = "no_hpo"
    if "status" not in frame.columns:
        frame["status"] = "unknown"
    if primary_metric not in frame.columns:
        frame[primary_metric] = pd.NA

    split_geometry = frame.get("split_id", pd.Series([""] * len(frame))).map(_parse_split_geometry)
    frame["repeat"] = [repeat for repeat, _fold in split_geometry]
    frame["fold"] = [fold for _repeat, fold in split_geometry]
    frame["metric_available"] = frame[primary_metric].notna()
    frame["coverage_status"] = frame["status"].astype(str)
    frame.loc[(frame["coverage_status"] == "success") & (~frame["metric_available"]), "coverage_status"] = "missing_metric"
    if "failure_type" in frame.columns:
        failure_type = frame["failure_type"].fillna("").astype(str)
    else:
        failure_type = pd.Series([""] * len(frame), index=frame.index)
    if "exception_message" in frame.columns:
        exception_message = frame["exception_message"].fillna("").astype(str)
    else:
        exception_message = pd.Series([""] * len(frame), index=frame.index)
    frame["failure_reason"] = failure_type.where(failure_type != "", exception_message)
    frame.loc[frame["status"].astype(str) == "success", "failure_reason"] = ""
    frame.loc[frame["coverage_status"] == "missing_metric", "failure_reason"] = "missing_primary_metric"
    frame["artifact_path"] = csv_path.with_name(f"{_artifact_prefix(file_prefix, fallback=benchmark_label(frame))}_fold_results.csv").name

    metric_cols = unique_in_order(
        [primary_metric]
        + [col for col in CORE_METRIC_COLUMNS + MANUSCRIPT_METRIC_COLUMNS if col != primary_metric]
        + expand_dynamic_metric_columns(frame)
    )
    available_metric_cols = [col for col in metric_cols if col in frame.columns]
    base_cols = [
        "benchmark_id",
        "dataset_id",
        "method_id",
        "hpo_mode",
        "seed",
        "split_id",
        "repeat",
        "fold",
        "coverage_status",
        "status",
        "runtime_sec",
        "failure_reason",
        "artifact_path",
    ]
    optional_cols = [
        "robustness_track_id",
        "retry_attempt",
        "requested_max_trials",
        "requested_timeout_seconds",
        "realized_trial_count",
    ]
    coverage_cols = [col for col in base_cols + optional_cols + available_metric_cols if col in frame.columns]
    coverage = frame[coverage_cols].sort_values(
        [col for col in ["dataset_id", "method_id", "hpo_mode", "seed", "repeat", "fold", "split_id"] if col in frame.columns]
    )
    coverage.to_csv(csv_path, index=False)
    _write_markdown_table(coverage, md_path, title="Coverage Matrix")
    return coverage


def _runtime_failure_detail_by_key(run_records: list[dict[str, Any]] | None) -> dict[tuple[str, str, str, int, str], dict[str, str]]:
    details: dict[tuple[str, str, str, int, str], dict[str, str]] = {}
    for run_record in run_records or []:
        metrics = run_record.get("metrics") or {}
        manifest = run_record.get("manifest") or {}
        dataset_id = str(run_record.get("dataset_id") or metrics.get("dataset_id") or manifest.get("dataset_id") or "")
        method_id = str(run_record.get("method_id") or metrics.get("method_id") or manifest.get("method_id") or "")
        hpo_mode = str(metrics.get("hpo_mode") or run_record.get("hpo_mode") or "unspecified")
        seed = int(metrics.get("seed") or manifest.get("seed") or run_record.get("seed") or 0)
        split_id = str(metrics.get("split_id") or manifest.get("split_id") or run_record.get("split_id") or "")
        failure = run_record.get("failure") or {}
        details[(dataset_id, method_id, hpo_mode, seed, split_id)] = {
            "failure_type": str(metrics.get("failure_type") or failure.get("type") or ""),
            "failure_message": str(metrics.get("exception_message") or failure.get("message") or manifest.get("notes") or ""),
            "traceback": str(failure.get("traceback") or ""),
        }
    return details


def _failure_category(*, status: str, missing_metric_count: int, detail: dict[str, str]) -> str:
    if status == "success" and missing_metric_count == 0:
        return "success"
    if status == "success" and missing_metric_count > 0:
        return "missing_metrics"

    failure_type = detail.get("failure_type", "")
    text = " ".join([failure_type, detail.get("failure_message", ""), detail.get("traceback", "")]).lower()
    if failure_type in {"ImportError", "ModuleNotFoundError"} or "no module named" in text:
        return "dependency_missing"
    if failure_type == "TimeoutError" or "timed out" in text or "timeout" in text or "time limit" in text:
        return "timeout"
    if failure_type == "MemoryError" or "out of memory" in text or "oom" in text:
        return "memory"
    if failure_type in {"FloatingPointError", "LinAlgError"} or "singular" in text or "convergence" in text:
        return "numerical"
    if failure_type == "ValueError" or "invalid" in text or "no event" in text or "shape" in text:
        return "data_validation"
    if failure_type or text.strip():
        return "runtime_error"
    return "unknown_failure"


def _runtime_failure_rows(
    fold_results: pd.DataFrame,
    *,
    run_records: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    output_columns = [
        "benchmark_id",
        "dataset_id",
        "method_id",
        "hpo_mode",
        "seed",
        "repeat",
        "fold",
        "split_id",
        "n_runs",
        "n_success",
        "n_crashed",
        "n_missing_metrics",
        "runtime_sec_mean",
        "runtime_sec_total",
        "failure_category",
        "failure_type",
        "failure_message",
        "missing_metric_columns",
    ]
    if fold_results.empty:
        return pd.DataFrame(columns=output_columns)

    details_by_key = _runtime_failure_detail_by_key(run_records)
    rows: list[dict[str, object]] = []
    metric_cols = [col for col in _RUNTIME_FAILURE_METRIC_COLUMNS if col in fold_results.columns]
    for _, row in fold_results.iterrows():
        dataset_id = str(row.get("dataset_id", ""))
        method_id = str(row.get("method_id", ""))
        hpo_mode = str(row.get("hpo_mode", "unspecified") or "unspecified")
        seed = int(row.get("seed", 0) or 0)
        split_id = str(row.get("split_id", ""))
        repeat, fold = _parse_split_geometry(split_id)
        detail = details_by_key.get((dataset_id, method_id, hpo_mode, seed, split_id), {})
        status = str(row.get("status", "") or "").lower()
        missing_metric_columns = [col for col in metric_cols if pd.isna(row.get(col))]
        category = _failure_category(
            status=status,
            missing_metric_count=len(missing_metric_columns),
            detail=detail,
        )
        rows.append(
            {
                "benchmark_id": row.get("benchmark_id", ""),
                "dataset_id": dataset_id,
                "method_id": method_id,
                "hpo_mode": hpo_mode,
                "seed": seed,
                "repeat": repeat,
                "fold": fold,
                "split_id": split_id,
                "status": status or "unknown",
                "runtime_sec": row.get("runtime_sec"),
                "failure_category": category,
                "failure_type": detail.get("failure_type", ""),
                "failure_message": detail.get("failure_message", ""),
                "missing_metric_columns": ", ".join(missing_metric_columns),
            }
        )

    detail_frame = pd.DataFrame(rows)
    group_keys = ["benchmark_id", "dataset_id", "method_id", "hpo_mode", "seed", "repeat", "fold", "split_id"]
    summary = (
        detail_frame.groupby(group_keys, as_index=False, dropna=False)
        .agg(
            n_runs=("status", "count"),
            n_success=("status", lambda values: int((values == "success").sum())),
            n_crashed=("status", lambda values: int((values != "success").sum())),
            n_missing_metrics=("failure_category", lambda values: int((values == "missing_metrics").sum())),
            runtime_sec_mean=("runtime_sec", "mean"),
            runtime_sec_total=("runtime_sec", "sum"),
            failure_category=("failure_category", lambda values: ", ".join(unique_in_order([str(v) for v in values]))),
            failure_type=("failure_type", lambda values: ", ".join(unique_in_order([str(v) for v in values if str(v)]))),
            failure_message=("failure_message", lambda values: " | ".join(unique_in_order([str(v) for v in values if str(v)]))),
            missing_metric_columns=(
                "missing_metric_columns",
                lambda values: ", ".join(unique_in_order([str(v) for v in values if str(v)])),
            ),
        )
        .sort_values(["dataset_id", "method_id", "hpo_mode", "seed", "repeat", "fold", "split_id"])
    )
    return summary[output_columns]


def _write_runtime_failure_markdown(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        path.write_text("# Runtime and Failure Summary\n\nNo runs were available to summarize.\n", encoding="utf-8")
        return

    total_runs = int(frame["n_runs"].sum())
    total_crashed = int(frame["n_crashed"].sum())
    total_missing = int(frame["n_missing_metrics"].sum())
    category_counts = frame.groupby("failure_category", as_index=False)["n_runs"].sum().sort_values(
        ["n_runs", "failure_category"],
        ascending=[False, True],
    )
    display_cols = [
        "dataset_id",
        "method_id",
        "hpo_mode",
        "seed",
        "fold",
        "n_runs",
        "n_crashed",
        "n_missing_metrics",
        "runtime_sec_mean",
        "failure_category",
    ]
    lines = [
        "# Runtime and Failure Summary",
        "",
        f"- Total rows summarized: {total_runs}",
        f"- Crashed run rows: {total_crashed}",
        f"- Successful rows with missing metrics: {total_missing}",
        "",
        "## Failure Categories",
        "",
    ]
    category_md = category_counts.rename(columns={"failure_category": "category", "n_runs": "rows"})
    detail_md = frame[display_cols]
    lines.extend(_markdown_table_lines(category_md))
    lines.extend(["", "## By Dataset, Method, Mode, Seed, And Fold", ""])
    lines.extend(_markdown_table_lines(detail_md))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_runtime_failure_summary(
    root: Path,
    *,
    benchmark_id: str,
    fold_results: pd.DataFrame,
    run_records: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    frame = _runtime_failure_rows(fold_results, run_records=run_records)
    if output_dir is None:
        output_base = root / "results" / "summaries"
        csv_path = output_base / f"{benchmark_id}_runtime_failure_summary.csv"
        md_path = output_base / f"{benchmark_id}_runtime_failure_summary.md"
    else:
        output_base = output_dir
        prefix = _artifact_prefix(file_prefix, fallback=benchmark_id)
        csv_path = output_base / f"{prefix}_runtime_failure_summary.csv"
        md_path = output_base / f"{prefix}_runtime_failure_summary.md"
    output_base.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    _write_runtime_failure_markdown(md_path, frame)
    return frame


def export_run_diagnostics(
    root: Path,
    *,
    benchmark_id: str,
    fold_results: pd.DataFrame,
    dataset_curation_rows: list[dict],
    hpo_trial_rows: list[dict],
    run_records: list[dict[str, Any]] | None = None,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    export_runtime_failure_summary(
        root,
        benchmark_id=benchmark_id,
        fold_results=fold_results,
        run_records=run_records,
        output_dir=output_dir,
        file_prefix=file_prefix,
    )
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
