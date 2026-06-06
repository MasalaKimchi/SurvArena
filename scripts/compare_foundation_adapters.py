#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ADAPTER_PAIRS = {
    "tabpfn_survival": "tabpfn_discrete_hazard_survival",
    "tabicl_survival": "tabicl_discrete_hazard_survival",
    "tabm_survival": "tabm_discrete_hazard_survival",
    "realtabpfn_survival": "realtabpfn_discrete_hazard_survival",
}
MAXIMIZE_METRICS = {"uno_c", "harrell_c", "td_auc_25", "td_auc_50", "td_auc_75"}
DEFAULT_METRICS = ("uno_c", "ibs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare legacy horizon and discrete-hazard foundation adapters.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing benchmark fold-result CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/foundation_adapter_comparison"),
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metrics to compare. Defaults to uno_c and ibs.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap draws for paired mean-delta CIs.")
    parser.add_argument("--seed", type=int, default=33, help="Random seed for bootstrap draws.")
    return parser.parse_args()


def _load_fold_results(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.rglob("*fold_results.csv"))
    if not paths:
        raise FileNotFoundError(f"No *fold_results.csv files found under {input_dir}.")
    frames = [pd.read_csv(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    if "method_id" not in frame.columns:
        raise ValueError("Fold results must include a method_id column.")
    methods = set(ADAPTER_PAIRS) | set(ADAPTER_PAIRS.values())
    return frame[frame["method_id"].astype(str).isin(methods)].copy()


def _pair_keys(frame: pd.DataFrame) -> list[str]:
    candidates = ["benchmark_id", "dataset_id", "split_id", "seed", "hpo_mode"]
    return [column for column in candidates if column in frame.columns]


def _metric_direction(metric: str) -> str:
    return "maximize" if metric in MAXIMIZE_METRICS else "minimize"


def _bootstrap_ci(values: np.ndarray, *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if int(n_bootstrap) <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [np.mean(rng.choice(values, size=values.size, replace=True)) for _ in range(int(n_bootstrap))],
        dtype=float,
    )
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _paired_delta_rows(frame: pd.DataFrame, metrics: list[str], *, n_bootstrap: int, seed: int) -> pd.DataFrame:
    keys = _pair_keys(frame)
    rows: list[dict[str, Any]] = []
    for legacy_id, hazard_id in ADAPTER_PAIRS.items():
        left = frame[frame["method_id"] == legacy_id]
        right = frame[frame["method_id"] == hazard_id]
        if left.empty or right.empty:
            continue
        merged = left.merge(right, on=keys, suffixes=("_legacy", "_hazard"), how="inner")
        for metric in metrics:
            legacy_col = f"{metric}_legacy"
            hazard_col = f"{metric}_hazard"
            if legacy_col not in merged.columns or hazard_col not in merged.columns:
                continue
            raw_delta = merged[hazard_col].to_numpy(dtype=float) - merged[legacy_col].to_numpy(dtype=float)
            improvement_delta = raw_delta if _metric_direction(metric) == "maximize" else -raw_delta
            finite = improvement_delta[np.isfinite(improvement_delta)]
            if finite.size >= 2:
                try:
                    p_value = float(stats.wilcoxon(finite, alternative="greater", zero_method="wilcox").pvalue)
                except ValueError:
                    p_value = 1.0
            else:
                p_value = 1.0
            ci_low, ci_high = _bootstrap_ci(finite, n_bootstrap=n_bootstrap, seed=seed)
            rows.append(
                {
                    "legacy_method_id": legacy_id,
                    "hazard_method_id": hazard_id,
                    "metric": metric,
                    "metric_direction": _metric_direction(metric),
                    "n_pairs": int(finite.size),
                    "mean_raw_delta_hazard_minus_legacy": float(np.mean(raw_delta)) if raw_delta.size else float("nan"),
                    "mean_improvement_delta": float(np.mean(finite)) if finite.size else float("nan"),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "wilcoxon_greater_p_value": p_value,
                    "wins": int(np.sum(finite > 0.0)),
                    "losses": int(np.sum(finite < 0.0)),
                    "ties": int(np.sum(finite == 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _leaderboard(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    group_cols = [column for column in ["benchmark_id", "dataset_id", "method_id", "hpo_mode"] if column in frame.columns]
    metric_cols = [metric for metric in metrics if metric in frame.columns]
    extra_cols = [column for column in ["runtime_sec", "fit_time_sec", "infer_time_sec", "peak_memory_mb"] if column in frame.columns]
    if not group_cols:
        group_cols = ["method_id"]
    return frame.groupby(group_cols, as_index=False)[metric_cols + extra_cols].mean(numeric_only=True)


def _dataset_summary(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    keys = _pair_keys(frame)
    rows: list[dict[str, Any]] = []
    for legacy_id, hazard_id in ADAPTER_PAIRS.items():
        left = frame[frame["method_id"] == legacy_id]
        right = frame[frame["method_id"] == hazard_id]
        if left.empty or right.empty:
            continue
        merged = left.merge(right, on=keys, suffixes=("_legacy", "_hazard"), how="inner")
        if "dataset_id" not in merged.columns:
            continue
        for dataset_id, sub in merged.groupby("dataset_id"):
            row: dict[str, Any] = {
                "dataset_id": dataset_id,
                "legacy_method_id": legacy_id,
                "hazard_method_id": hazard_id,
                "n_pairs": int(len(sub)),
            }
            for metric in metrics:
                legacy_col = f"{metric}_legacy"
                hazard_col = f"{metric}_hazard"
                if legacy_col not in sub.columns or hazard_col not in sub.columns:
                    continue
                delta = sub[hazard_col].to_numpy(dtype=float) - sub[legacy_col].to_numpy(dtype=float)
                improvement = delta if _metric_direction(metric) == "maximize" else -delta
                row[f"{metric}_mean_delta"] = float(np.nanmean(delta))
                row[f"{metric}_wins"] = int(np.sum(improvement > 0.0))
                row[f"{metric}_losses"] = int(np.sum(improvement < 0.0))
                row[f"{metric}_ties"] = int(np.sum(improvement == 0.0))
            rows.append(row)
    return pd.DataFrame(rows)


def _metadata_summary(frame: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [column for column in frame.columns if column.startswith("foundation_")]
    if not metadata_cols:
        return pd.DataFrame(columns=["method_id", "runs"])
    rows: list[dict[str, Any]] = []
    for method_id, sub in frame.groupby("method_id"):
        row: dict[str, Any] = {"method_id": method_id, "runs": int(len(sub))}
        for column in metadata_cols:
            values = sub[column].dropna()
            if values.empty:
                continue
            if pd.api.types.is_numeric_dtype(values):
                row[f"{column}_mean"] = float(values.astype(float).mean())
            else:
                row[column] = ";".join(sorted(values.astype(str).unique())[:5])
        rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown_summary(output_dir: Path, paired: pd.DataFrame, leaderboard: pd.DataFrame) -> None:
    lines = [
        "# Foundation Adapter Comparison",
        "",
        "This summary compares legacy cumulative-horizon foundation adapters against pooled discrete-time hazard adapters.",
        "",
        "## Paired Deltas",
        "",
    ]
    if paired.empty:
        lines.append("No paired adapter rows were available.")
    else:
        for row in paired.to_dict(orient="records"):
            lines.append(
                "- {hazard} vs {legacy} on {metric}: mean improvement {delta:.4f} "
                "over {n_pairs} pairs (wins/losses/ties {wins}/{losses}/{ties}).".format(
                    hazard=row["hazard_method_id"],
                    legacy=row["legacy_method_id"],
                    metric=row["metric"],
                    delta=float(row["mean_improvement_delta"]),
                    n_pairs=int(row["n_pairs"]),
                    wins=int(row["wins"]),
                    losses=int(row["losses"]),
                    ties=int(row["ties"]),
                )
            )
    lines.extend(["", "## Coverage", ""])
    if leaderboard.empty:
        lines.append("No leaderboard rows were available.")
    else:
        methods = ", ".join(sorted(leaderboard["method_id"].astype(str).unique()))
        lines.append(f"Methods represented: {methods}.")
    (output_dir / "manuscript_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    frame = _load_fold_results(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [metric for metric in args.metrics if metric in frame.columns]
    if not metrics:
        raise ValueError("None of the requested metrics were found in the fold results.")

    leaderboard = _leaderboard(frame, metrics)
    paired = _paired_delta_rows(frame, metrics, n_bootstrap=int(args.bootstrap), seed=int(args.seed))
    dataset = _dataset_summary(frame, metrics)
    metadata = _metadata_summary(frame)

    leaderboard.to_csv(args.output_dir / "leaderboard.csv", index=False)
    paired.to_csv(args.output_dir / "paired_deltas.csv", index=False)
    dataset.to_csv(args.output_dir / "dataset_summary.csv", index=False)
    metadata.to_csv(args.output_dir / "adapter_metadata_summary.csv", index=False)
    _write_markdown_summary(args.output_dir, paired, leaderboard)


if __name__ == "__main__":
    main()
