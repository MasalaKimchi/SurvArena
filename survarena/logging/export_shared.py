from __future__ import annotations

from pathlib import Path

import pandas as pd

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


def expand_dynamic_metric_columns(frame: pd.DataFrame) -> list[str]:
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


def unique_in_order(columns: list[str]) -> list[str]:
    return list(dict.fromkeys(columns))


def parity_gated_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "parity_eligible" not in frame.columns:
        return frame
    mask = frame["parity_eligible"].fillna(False).astype(bool)
    return frame.loc[mask].copy()


def benchmark_label(frame: pd.DataFrame, fallback: str = "benchmark") -> str:
    if "benchmark_id" not in frame.columns or frame.empty:
        return fallback
    unique = frame["benchmark_id"].dropna().astype(str).unique().tolist()
    if len(unique) == 1 and unique[0]:
        return unique[0]
    return fallback


def group_keys_with_hpo_mode(frame: pd.DataFrame, base: list[str]) -> list[str]:
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


def output_path(root: Path, output_dir: Path | None, default_parts: tuple[str, ...], filename: str) -> Path:
    if output_dir is None:
        output = root.joinpath(*default_parts, filename)
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename
