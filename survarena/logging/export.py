from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from survarena.evaluation.statistics import (
    aggregate_rank_summary,
    bootstrap_metric_ci,
    critical_difference_summary,
    elo_ratings,
    failure_summary,
    metric_direction,
    pairwise_win_rate,
    pairwise_significance,
    summarize_frame,
)
from survarena.logging.tracker import write_json, write_jsonl_gz

RUN_LEDGER_SCHEMA_VERSION = "2.0"
RUN_LEDGER_COMPACT_SCHEMA_VERSION = "1.0"
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
GOVERNANCE_COLUMNS = [
    "requested_max_trials",
    "requested_timeout_seconds",
    "realized_trial_count",
]


def _expand_dynamic_metric_columns(frame: pd.DataFrame) -> list[str]:
    dynamic_prefixes = (
        "calibration_slope_",
        "calibration_intercept_",
        "net_benefit_",
        "decision_curve_aunb_",
        "brier_",
        "td_auc_",
    )
    dynamic = [col for col in frame.columns if any(col.startswith(prefix) for prefix in dynamic_prefixes)]
    return sorted(set(dynamic))


def _unique_in_order(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def _parity_gated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "parity_eligible" not in frame.columns:
        return frame
    mask = frame["parity_eligible"].fillna(False).astype(bool)
    return frame.loc[mask].copy()


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


def _group_keys_with_hpo_mode(frame: pd.DataFrame, base: list[str]) -> list[str]:
    """Stratify aggregation keys with ``hpo_mode`` when the column is present (dual-mode benchmarks)."""
    if "hpo_mode" not in frame.columns or "method_id" not in base:
        return base
    if "hpo_mode" in base:
        return base
    out: list[str] = []
    for col in base:
        out.append(col)
        if col == "method_id":
            out.append("hpo_mode")
    return out


def export_seed_summary(
    root: Path,
    frame: pd.DataFrame,
    *,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
) -> pd.DataFrame:
    by_cols = _group_keys_with_hpo_mode(
        frame,
        ["benchmark_id", "dataset_id", "method_id", "seed"],
    )
    metric_cols = _unique_in_order(BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + _expand_dynamic_metric_columns(frame))
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
    metric_cols = _unique_in_order(BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + _expand_dynamic_metric_columns(frame))
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
    metric_cols = _unique_in_order(BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + _expand_dynamic_metric_columns(seed_summary))
    available_metric_cols = [col for col in metric_cols if col in seed_summary.columns]
    lb_keys = _group_keys_with_hpo_mode(
        seed_summary,
        ["benchmark_id", "dataset_id", "method_id"],
    )
    leaderboard = seed_summary.groupby(lb_keys, as_index=False)[
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

    significance_source = fold_results if fold_results is not None else leaderboard
    significance_source = _parity_gated_frame(significance_source)
    if fold_results is not None:
        claim_metric_cols = _unique_in_order(BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + _expand_dynamic_metric_columns(significance_source))
        available_claim_metric_cols = [col for col in claim_metric_cols if col in significance_source.columns]
        if significance_source.empty:
            comparative_leaderboard = leaderboard.iloc[0:0].copy()
        else:
            cl_keys = _group_keys_with_hpo_mode(
                significance_source,
                ["benchmark_id", "dataset_id", "method_id"],
            )
            comparative_leaderboard = significance_source.groupby(
                cl_keys, as_index=False
            )[available_claim_metric_cols].mean(numeric_only=True)
    else:
        comparative_leaderboard = _parity_gated_frame(leaderboard)

    rank_summary = aggregate_rank_summary(comparative_leaderboard, metric=primary_metric)
    pairwise = pairwise_win_rate(comparative_leaderboard, metric=primary_metric)
    pairwise_sig = pairwise_significance(significance_source, metric=primary_metric, correction="holm")
    cd_summary = critical_difference_summary(comparative_leaderboard, metric=primary_metric)
    ci = bootstrap_metric_ci(comparative_leaderboard, metric=primary_metric, n_bootstrap=1000, seed=0)
    elo = elo_ratings(comparative_leaderboard, metric=primary_metric)
    failures = failure_summary(fold_results if fold_results is not None else leaderboard)
    missing_metric_rows = []
    metric_cols = CORE_METRIC_COLUMNS[1:] + MANUSCRIPT_METRIC_COLUMNS
    for metric in [col for col in metric_cols if col in comparative_leaderboard.columns]:
        missing_metric_rows.append(
            {
                "metric": metric,
                "missing_rate": float(comparative_leaderboard[metric].isna().mean())
                if len(comparative_leaderboard)
                else float("nan"),
            }
        )
    missing = pd.DataFrame(missing_metric_rows)

    paths = {
        "rank_summary": str(output_dir / f"{prefix}_rank_summary.csv"),
        "pairwise_win_rate": str(output_dir / f"{prefix}_pairwise_win_rate.csv"),
        "pairwise_significance": str(output_dir / f"{prefix}_pairwise_significance.csv"),
        "multiple_comparison_summary": str(output_dir / f"{prefix}_multiple_comparison_summary.csv"),
        "critical_difference": str(output_dir / f"{prefix}_critical_difference.csv"),
        "bootstrap_ci": str(output_dir / f"{prefix}_bootstrap_ci.csv"),
        "elo_ratings": str(output_dir / f"{prefix}_elo_ratings.csv"),
        "failure_summary": str(output_dir / f"{prefix}_failure_summary.csv"),
        "missing_metric_summary": str(output_dir / f"{prefix}_missing_metric_summary.csv"),
    }
    rank_summary.to_csv(paths["rank_summary"], index=False)
    pairwise.to_csv(paths["pairwise_win_rate"], index=False)
    pairwise_sig.to_csv(paths["pairwise_significance"], index=False)
    if pairwise_sig.empty:
        mcs_cols = ["benchmark_id", "method_id", "n_significant_wins", "n_significant_losses", "correction"]
        if "hpo_mode" in significance_source.columns:
            mcs_cols = [
                "benchmark_id",
                "hpo_mode",
                "method_id",
                "n_significant_wins",
                "n_significant_losses",
                "correction",
            ]
        multiple_summary = pd.DataFrame(columns=mcs_cols)
    else:
        pairwise_sig_flags = pairwise_sig.assign(
            significant=lambda df: df["p_value_corrected"] < 0.05,
            positive_effect=lambda df: df["effect_size_mean_delta"] > 0,
        )
        pairwise_sig_flags["significant_win"] = pairwise_sig_flags["significant"] & pairwise_sig_flags["positive_effect"]
        pairwise_sig_flags["significant_loss"] = pairwise_sig_flags["significant"] & (~pairwise_sig_flags["positive_effect"])
        mc_keys = ["benchmark_id", "method_id", "correction"]
        if "hpo_mode" in pairwise_sig_flags.columns:
            mc_keys = ["benchmark_id", "hpo_mode", "method_id", "correction"]
        multiple_summary = pairwise_sig_flags.groupby(mc_keys, as_index=False).agg(
            n_significant_wins=("significant_win", lambda values: int(np.sum(values))),
            n_significant_losses=("significant_loss", lambda values: int(np.sum(values))),
        )
    multiple_summary.to_csv(paths["multiple_comparison_summary"], index=False)
    cd_summary.to_csv(paths["critical_difference"], index=False)
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
            "pairwise_significance_records": pairwise_sig.to_dict(orient="records"),
            "missing_metric_summary": missing.to_dict(orient="records"),
            "parity_gate": {
                "applied": "parity_eligible" in significance_source.columns,
                "comparison_rows": int(len(significance_source)),
            },
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
    normalized_records: list[dict[str, object]] = []
    for record in run_records:
        normalized = {
            "schema_version": RUN_LEDGER_SCHEMA_VERSION,
            **record,
        }
        metrics = normalized.get("metrics")
        status = normalized.get("status")
        retry_attempt = normalized.get("retry_attempt")
        if isinstance(metrics, dict):
            if status is None:
                status = metrics.get("status")
            if retry_attempt is None:
                retry_attempt = metrics.get("retry_attempt")
        normalized["status"] = status if status is not None else "unknown"
        try:
            normalized["retry_attempt"] = int(retry_attempt) if retry_attempt is not None else 0
        except (TypeError, ValueError):
            normalized["retry_attempt"] = 0
        normalized["failure"] = normalized.get("failure")
        normalized_records.append(normalized)
    if output_dir is None:
        output = root / "results" / "runs" / f"{benchmark_id}_run_records.jsonl.gz"
        index_output = root / "results" / "runs" / f"{benchmark_id}_run_records_index.json"
        compact_output = root / "results" / "runs" / f"{benchmark_id}_run_records_compact.jsonl.gz"
        compact_index_output = root / "results" / "runs" / f"{benchmark_id}_run_records_compact_index.json"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"{benchmark_id}_run_records.jsonl.gz"
        index_output = output_dir / f"{benchmark_id}_run_records_index.json"
        compact_output = output_dir / f"{benchmark_id}_run_records_compact.jsonl.gz"
        compact_index_output = output_dir / f"{benchmark_id}_run_records_compact_index.json"
    write_jsonl_gz(output, normalized_records)
    shared_manifest: dict[str, object] = {}
    if normalized_records:
        first_manifest = normalized_records[0].get("manifest")
        if isinstance(first_manifest, dict):
            for key, value in first_manifest.items():
                if all(
                    isinstance(record.get("manifest"), dict) and record["manifest"].get(key) == value
                    for record in normalized_records
                ):
                    shared_manifest[key] = value
    compact_records: list[dict[str, object]] = []
    for record in normalized_records:
        compact: dict[str, object] = {"schema_version": RUN_LEDGER_COMPACT_SCHEMA_VERSION}
        for key, value in record.items():
            if key == "schema_version":
                continue
            if key == "manifest" and isinstance(value, dict):
                unique_manifest = {mk: mv for mk, mv in value.items() if mk not in shared_manifest}
                if unique_manifest:
                    compact["manifest"] = unique_manifest
                continue
            compact[key] = value
        compact_records.append(compact)
    write_jsonl_gz(compact_output, compact_records)
    write_json(
        index_output,
        {
            "schema_version": RUN_LEDGER_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "record_count": len(normalized_records),
            "format": "jsonl.gz",
            "path": str(output),
            "created_at": created_at,
            "record_sections": [
                "schema_version",
                "manifest",
                "metrics",
                "backend_metadata",
                "hpo_metadata",
                "hpo_trials",
                "failure",
            ],
        },
    )
    write_json(
        compact_index_output,
        {
            "schema_version": RUN_LEDGER_COMPACT_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "record_count": len(compact_records),
            "format": "jsonl.gz",
            "path": str(compact_output),
            "created_at": created_at,
            "manifest_shared": shared_manifest,
            "record_sections": [
                "schema_version",
                "manifest_shared",
                "manifest",
                "metrics",
                "backend_metadata",
                "hpo_metadata",
                "hpo_trials",
                "failure",
            ],
        },
    )


def export_experiment_navigator(
    output_dir: Path,
    *,
    benchmark_id: str,
    primary_metric: str,
    split_count: int,
    method_count: int,
    leaderboard: pd.DataFrame,
) -> None:
    top_runs = leaderboard.sort_values(by=[primary_metric], ascending=metric_direction(primary_metric) == "minimize").head(5)
    top_records = top_runs.to_dict(orient="records")
    core_files = [
        f"{benchmark_id}_leaderboard.csv",
        f"{benchmark_id}_overall_summary.json",
        f"{benchmark_id}_manuscript_summary.json",
        f"{benchmark_id}_dataset_curation.csv",
        "experiment_manifest.json",
    ]
    detailed_files = [
        f"{benchmark_id}_fold_results.csv",
        f"{benchmark_id}_seed_summary.csv",
        f"{benchmark_id}_leaderboard.json",
        f"{benchmark_id}_rank_summary.csv",
        f"{benchmark_id}_pairwise_win_rate.csv",
        f"{benchmark_id}_pairwise_significance.csv",
        f"{benchmark_id}_multiple_comparison_summary.csv",
        f"{benchmark_id}_critical_difference.csv",
        f"{benchmark_id}_elo_ratings.csv",
        f"{benchmark_id}_bootstrap_ci.csv",
        f"{benchmark_id}_failure_summary.csv",
        f"{benchmark_id}_missing_metric_summary.csv",
        f"{benchmark_id}_run_records.jsonl.gz",
        f"{benchmark_id}_run_records_index.json",
        f"{benchmark_id}_run_records_compact.jsonl.gz",
        f"{benchmark_id}_run_records_compact_index.json",
    ]
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
