from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survarena.config import read_yaml
from survarena.evaluation.statistics import aggregate_rank_summary, elo_ratings, metric_direction, pairwise_win_rate


DEFAULT_INPUT = Path("results/manuscript_grade/clinical_no_hpo/dataset_model")
DEFAULT_OUTPUT = Path("results/manuscript_grade/clinical_no_hpo/elo")
DEFAULT_ASSET_DIR = Path("docs/assets")
DEFAULT_METRIC_SUITE = (
    "uno_c",
    "harrell_c",
    "ibs",
    "td_auc_25",
    "td_auc_50",
    "td_auc_75",
    "brier_25",
    "brier_50",
    "brier_75",
    "calibration_slope_abs_error_25",
    "calibration_slope_abs_error_50",
    "calibration_slope_abs_error_75",
    "calibration_intercept_abs_error_25",
    "calibration_intercept_abs_error_50",
    "calibration_intercept_abs_error_75",
    "net_benefit_25",
    "net_benefit_50",
    "net_benefit_75",
)

DISPLAY_NAMES = {
    "aalen_additive": "Aalen Additive",
    "catboost_cox": "CatBoost Cox",
    "catboost_survival_aft": "CatBoost AFT",
    "componentwise_gradient_boosting": "Component GB",
    "cox_time": "Cox-Time",
    "coxnet": "CoxNet",
    "coxph": "CoxPH",
    "deephit_single": "DeepHit",
    "deepsurv": "DeepSurv",
    "deepsurv_moco": "DeepSurv MoCo",
    "extra_survival_trees": "Extra Trees",
    "fast_survival_svm": "Fast Survival SVM",
    "gradient_boosting_survival": "Gradient Boosting",
    "logistic_hazard": "Logistic Hazard",
    "loglogistic_aft": "Log-Logistic AFT",
    "lognormal_aft": "Log-Normal AFT",
    "mtlr": "MTLR",
    "pchazard": "PC-Hazard",
    "pmf": "PMF",
    "rsf": "Random Survival Forest",
    "weibull_aft": "Weibull AFT",
    "xgboost_aft": "XGBoost AFT",
    "xgboost_cox": "XGBoost Cox",
}

FAMILY_COLORS = {
    "classical": "#2563eb",
    "tree": "#059669",
    "boosting": "#d97706",
    "deep": "#dc2626",
    "foundation": "#7b2cbf",
    "automl": "#0891b2",
    "unknown": "#6b7280",
}


METRIC_LABELS = {
    "uno_c": "Uno C",
    "harrell_c": "Harrell C",
    "ibs": "Integrated Brier Score",
    "td_auc_25": "Time-dependent AUC (25%)",
    "td_auc_50": "Time-dependent AUC (50%)",
    "td_auc_75": "Time-dependent AUC (75%)",
    "brier_25": "Brier Score (25%)",
    "brier_50": "Brier Score (50%)",
    "brier_75": "Brier Score (75%)",
    "calibration_slope_abs_error_25": "Calibration Slope Absolute Error (25%)",
    "calibration_slope_abs_error_50": "Calibration Slope Absolute Error (50%)",
    "calibration_slope_abs_error_75": "Calibration Slope Absolute Error (75%)",
    "calibration_intercept_abs_error_25": "Calibration Intercept Absolute Error (25%)",
    "calibration_intercept_abs_error_50": "Calibration Intercept Absolute Error (50%)",
    "calibration_intercept_abs_error_75": "Calibration Intercept Absolute Error (75%)",
    "net_benefit_25": "Net Benefit (25%)",
    "net_benefit_50": "Net Benefit (50%)",
    "net_benefit_75": "Net Benefit (75%)",
}


def _base_id(value: Any) -> str:
    return str(value).split("__", 1)[0]


def _display_name(method_id: str) -> str:
    return DISPLAY_NAMES.get(method_id, method_id.replace("_", " ").title())


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _metric_stem(metric: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in metric).strip("_")


def _load_method_metadata(repo_root: Path, method_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method_id in method_ids:
        path = repo_root / "configs" / "methods" / f"{method_id}.yaml"
        cfg = read_yaml(path) if path.exists() else {}
        rows.append(
            {
                "method_id": method_id,
                "family": cfg.get("family", "unknown"),
                "display_name": cfg.get("display_name", _display_name(method_id)),
                "library": cfg.get("library", ""),
            }
        )
    return pd.DataFrame(rows)


def _available_metrics(frame: pd.DataFrame) -> list[str]:
    frame = _with_derived_comparable_metrics(frame)
    metrics: list[str] = []
    for column in frame.columns:
        try:
            metric_direction(str(column))
        except ValueError:
            continue
        if pd.to_numeric(frame[column], errors="coerce").notna().any():
            metrics.append(str(column))
    return metrics


def _with_derived_comparable_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in list(out.columns):
        if column.startswith("calibration_slope_") and not column.startswith("calibration_slope_abs_error_"):
            suffix = column.removeprefix("calibration_slope_")
            target = f"calibration_slope_abs_error_{suffix}"
            if target not in out.columns:
                out[target] = (pd.to_numeric(out[column], errors="coerce") - 1.0).abs()
        if column.startswith("calibration_intercept_") and not column.startswith("calibration_intercept_abs_error_"):
            suffix = column.removeprefix("calibration_intercept_")
            target = f"calibration_intercept_abs_error_{suffix}"
            if target not in out.columns:
                out[target] = pd.to_numeric(out[column], errors="coerce").abs()
    return out


def _default_metrics_for_frame(frame: pd.DataFrame) -> list[str]:
    available = set(_available_metrics(frame))
    return [metric for metric in DEFAULT_METRIC_SUITE if metric in available]


def _input_dirs(input_dir: Path | Sequence[Path]) -> list[Path]:
    if isinstance(input_dir, Path):
        return [input_dir]
    return list(input_dir)


def _load_raw_fold_results(input_dir: Path | Sequence[Path]) -> pd.DataFrame:
    paths: list[Path] = []
    for root in _input_dirs(input_dir):
        paths.extend(sorted(root.glob("*/*/*_fold_results.csv")))
    if not paths:
        roots = ", ".join(str(root) for root in _input_dirs(input_dir))
        raise ValueError(f"No fold result CSVs found under {roots}.")
    return _with_derived_comparable_metrics(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False))


def _load_fold_results(input_dir: Path | Sequence[Path], *, metric: str) -> pd.DataFrame:
    return _prepare_fold_results(_load_raw_fold_results(input_dir), metric=metric)


def _prepare_fold_results(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    required = {"benchmark_id", "dataset_id", "method_id", "split_id", "seed", "hpo_mode", "status", metric}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fold results are missing required columns: {sorted(missing)}")
    metric_direction(metric)

    successful = frame[frame["status"].eq("success") & frame[metric].notna()].copy()
    successful["dataset_id"] = successful["dataset_id"].map(_base_id)
    successful["split_id"] = successful["split_id"].map(_base_id)
    successful["benchmark_id"] = "manuscript_v1"
    successful["hpo_mode"] = "no_hpo"
    return successful


def _prepare_metric_suite_fold_results(frame: pd.DataFrame, *, metrics: list[str]) -> pd.DataFrame:
    required = {"benchmark_id", "dataset_id", "method_id", "split_id", "seed", "hpo_mode", "status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fold results are missing required columns: {sorted(missing)}")
    for metric in metrics:
        if metric not in frame.columns:
            raise ValueError(f"Fold results are missing required metric column: {metric}")
        metric_direction(metric)

    metric_frame = frame[metrics].apply(pd.to_numeric, errors="coerce")
    successful = frame[frame["status"].eq("success") & metric_frame.notna().any(axis=1)].copy()
    successful["dataset_id"] = successful["dataset_id"].map(_base_id)
    successful["split_id"] = successful["split_id"].map(_base_id)
    successful["benchmark_id"] = "manuscript_v1"
    successful["hpo_mode"] = "no_hpo"
    return successful


def _complete_eligible_subset(
    frame: pd.DataFrame,
    *,
    metrics: list[str],
    expected_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if expected_splits < 1:
        raise ValueError(f"expected_splits must be >= 1. Received: {expected_splits}.")
    required = {"dataset_id", "method_id", "split_id", "status", *metrics}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fold results are missing required columns for eligibility filtering: {sorted(missing)}")

    normalized = frame.copy()
    normalized["dataset_id_base"] = normalized["dataset_id"].map(_base_id)
    normalized["split_id_base"] = normalized["split_id"].map(_base_id)
    metric_complete = normalized[metrics].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    success_complete = normalized["status"].eq("success") & metric_complete

    total = (
        normalized.groupby(["dataset_id_base", "method_id"], as_index=False)
        .agg(total_rows=("split_id_base", "size"), observed_splits=("split_id_base", "nunique"))
        .rename(columns={"dataset_id_base": "dataset_id"})
    )
    complete = (
        normalized[success_complete]
        .groupby(["dataset_id_base", "method_id"], as_index=False)
        .agg(metric_complete_success_rows=("split_id_base", "size"), metric_complete_success_splits=("split_id_base", "nunique"))
        .rename(columns={"dataset_id_base": "dataset_id"})
    )
    summary = total.merge(complete, on=["dataset_id", "method_id"], how="left")
    summary[["metric_complete_success_rows", "metric_complete_success_splits"]] = summary[
        ["metric_complete_success_rows", "metric_complete_success_splits"]
    ].fillna(0).astype(int)
    summary["expected_splits"] = int(expected_splits)
    summary["eligible"] = (summary["metric_complete_success_rows"] == expected_splits) & (
        summary["metric_complete_success_splits"] == expected_splits
    )
    summary["ineligibility_reason"] = ""
    summary.loc[summary["metric_complete_success_rows"].eq(0), "ineligibility_reason"] = "no_metric_complete_success_rows"
    summary.loc[
        summary["metric_complete_success_rows"].gt(0) & summary["metric_complete_success_splits"].lt(expected_splits),
        "ineligibility_reason",
    ] = "incomplete_successful_split_coverage"

    eligible_pairs = set(
        summary.loc[summary["eligible"], ["dataset_id", "method_id"]].itertuples(index=False, name=None)
    )
    pair_keys = list(zip(normalized["dataset_id_base"], normalized["method_id"], strict=False))
    filtered = frame[[key in eligible_pairs for key in pair_keys]].copy()
    return filtered, summary.sort_values(["eligible", "dataset_id", "method_id"], ascending=[True, True, True])


def _validate_coverage(frame: pd.DataFrame, *, metric: str, strict: bool) -> pd.DataFrame:
    coverage = (
        frame.groupby(["dataset_id", "method_id"], as_index=False)
        .agg(
            rows=("split_id", "size"),
            splits=("split_id", "nunique"),
            seeds=("seed", "nunique"),
            mean_score=(metric, "mean"),
        )
        .sort_values(["dataset_id", "method_id"])
    )
    bad = coverage[(coverage["rows"] != 15) | (coverage["splits"] != 15)]
    if strict and not bad.empty:
        raise ValueError("Manuscript fold coverage is incomplete:\n" + bad.to_string(index=False))
    return coverage


def _write_plot(elo: pd.DataFrame, output_dir: Path, asset_path: Path | None, *, metric: str) -> dict[str, Path]:
    plot_frame = elo.sort_values("elo_rating", ascending=True).copy()
    plot_frame["display_name"] = plot_frame["display_name"].fillna(plot_frame["method_id"].map(_display_name))
    plot_frame["family"] = plot_frame["family"].fillna("unknown")

    fig, ax = plt.subplots(figsize=(11.5, max(8.0, 0.34 * len(plot_frame))))
    y = range(len(plot_frame))
    colors = [FAMILY_COLORS.get(str(family), FAMILY_COLORS["unknown"]) for family in plot_frame["family"]]
    low = (plot_frame["elo_rating"] - plot_frame["elo_rating_ci95_low"]).clip(lower=0)
    high = (plot_frame["elo_rating_ci95_high"] - plot_frame["elo_rating"]).clip(lower=0)

    ax.barh(y, plot_frame["elo_rating"], color=colors, alpha=0.88, edgecolor="white", linewidth=0.8)
    ax.errorbar(
        plot_frame["elo_rating"],
        list(y),
        xerr=[low.to_numpy(), high.to_numpy()],
        fmt="none",
        ecolor="#111827",
        elinewidth=0.85,
        capsize=2,
    )
    ax.axvline(1500, color="#111827", linestyle="--", linewidth=1.1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(plot_frame["display_name"], fontsize=9)
    metric_label = _metric_label(metric)
    ax.set_xlabel(f"Elo rating from paired {metric_label} win rate")
    ax.set_title(f"Manuscript Benchmark Elo (No-HPO, {metric_label})", fontsize=14, fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    families = list(dict.fromkeys(plot_frame["family"].astype(str)))
    handles = [Patch(facecolor=FAMILY_COLORS.get(family, FAMILY_COLORS["unknown"]), label=family.title()) for family in families]
    ax.legend(handles=handles, ncol=min(4, len(handles)), loc="lower right", frameon=True)

    fig.tight_layout()
    stem = _metric_stem(metric)
    png = output_dir / f"elo_manuscript_no_hpo_{stem}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    if asset_path is not None:
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(asset_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs = {"figure_png": png}
    if asset_path is not None:
        outputs["readme_asset"] = asset_path
    return outputs


def build_outputs(
    *,
    repo_root: Path,
    input_dir: Path | Sequence[Path],
    output_dir: Path,
    asset_path: Path | None,
    metric: str,
    n_bootstrap: int = 1000,
    strict_coverage: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = _load_fold_results(input_dir, metric=metric)
    return _build_outputs_from_fold_results(
        repo_root=repo_root,
        output_dir=output_dir,
        asset_path=asset_path,
        metric=metric,
        n_bootstrap=n_bootstrap,
        strict_coverage=strict_coverage,
        fold_results=fold_results,
    )


def _build_outputs_from_fold_results(
    *,
    repo_root: Path,
    output_dir: Path,
    asset_path: Path | None,
    metric: str,
    n_bootstrap: int,
    strict_coverage: bool,
    fold_results: pd.DataFrame,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _metric_stem(metric)
    tables = _metric_output_tables(
        repo_root=repo_root,
        metric=metric,
        n_bootstrap=n_bootstrap,
        strict_coverage=strict_coverage,
        fold_results=fold_results,
        include_metric_column=False,
    )
    fold_results = tables["fold_results"]

    combined_path = output_dir / f"manuscript_fold_results_success_{stem}.csv"
    fold_results.to_csv(combined_path, index=False)

    elo = tables["elo"]
    elo_path = output_dir / f"elo_ratings_{stem}.csv"
    elo.to_csv(elo_path, index=False)

    wins = tables["pairwise"]
    wins_path = output_dir / f"pairwise_win_rate_{stem}.csv"
    wins.to_csv(wins_path, index=False)

    ranks = tables["ranks"]
    ranks_path = output_dir / f"rank_summary_{stem}.csv"
    ranks.to_csv(ranks_path, index=False)

    coverage = tables["coverage"]
    coverage_path = output_dir / f"coverage_summary_{stem}.csv"
    coverage.to_csv(coverage_path, index=False)

    summary = tables["summary"]
    summary_path = output_dir / f"method_summary_{stem}.csv"
    summary.to_csv(summary_path, index=False)

    outputs = {
        "combined": combined_path,
        "elo": elo_path,
        "pairwise": wins_path,
        "ranks": ranks_path,
        "coverage": coverage_path,
        "summary": summary_path,
    }
    outputs.update(_write_plot(elo, output_dir, asset_path, metric=metric))
    return outputs


def _metric_output_tables(
    *,
    repo_root: Path,
    metric: str,
    n_bootstrap: int,
    strict_coverage: bool,
    fold_results: pd.DataFrame,
    include_metric_column: bool,
) -> dict[str, pd.DataFrame]:
    coverage = _validate_coverage(fold_results, metric=metric, strict=strict_coverage)
    method_ids = sorted(fold_results["method_id"].dropna().astype(str).unique())
    metadata = _load_method_metadata(repo_root, method_ids)

    elo = elo_ratings(fold_results, metric=metric, n_bootstrap=n_bootstrap, seed=33).merge(
        metadata,
        on="method_id",
        how="left",
    )
    wins = pairwise_win_rate(fold_results, metric=metric)
    ranks = aggregate_rank_summary(fold_results, metric=metric)
    summary = (
        fold_results.groupby(["benchmark_id", "hpo_mode", "method_id"], as_index=False)
        .agg(
            successful_folds=("status", "size"),
            mean_score=(metric, "mean"),
            median_score=(metric, "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
            total_runtime_sec=("runtime_sec", "sum"),
        )
        .merge(metadata, on="method_id", how="left")
        .sort_values("mean_score", ascending=metric_direction(metric) == "minimize")
    )

    tables = {
        "fold_results": fold_results.copy(),
        "elo": elo,
        "pairwise": wins,
        "ranks": ranks,
        "coverage": coverage,
        "summary": summary,
    }
    if include_metric_column:
        for name, table in tables.items():
            if name != "fold_results":
                table["metric"] = metric
    else:
        summary["metric"] = metric
    return tables


def build_metric_suite_outputs(
    *,
    repo_root: Path,
    input_dir: Path | Sequence[Path],
    output_dir: Path,
    asset_dir: Path | None,
    metrics: list[str],
    n_bootstrap: int = 1000,
    strict_coverage: bool = False,
    complete_eligible_only: bool = False,
    expected_splits: int = 15,
) -> dict[str, Path]:
    if not metrics:
        raise ValueError("No comparable metrics were requested or found in fold results.")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_fold_results = _load_raw_fold_results(input_dir)
    if complete_eligible_only:
        raw_fold_results, eligibility = _complete_eligible_subset(
            raw_fold_results,
            metrics=metrics,
            expected_splits=expected_splits,
        )
        eligibility_path = output_dir / "eligibility_summary.csv"
        eligibility.to_csv(eligibility_path, index=False)
        ineligible_path = output_dir / "ineligible_dataset_method_pairs.csv"
        eligibility[~eligibility["eligible"]].to_csv(ineligible_path, index=False)
    else:
        eligibility_path = None
        ineligible_path = None
    rows: list[dict[str, str]] = []
    outputs: dict[str, Path] = {}
    aggregate_tables: dict[str, list[pd.DataFrame]] = {
        "elo": [],
        "pairwise": [],
        "ranks": [],
        "coverage": [],
        "summary": [],
    }
    for metric in metrics:
        asset_path = None if asset_dir is None else asset_dir / f"elo_manuscript_no_hpo_{_metric_stem(metric)}.png"
        metric_tables = _metric_output_tables(
            repo_root=repo_root,
            metric=metric,
            n_bootstrap=n_bootstrap,
            strict_coverage=strict_coverage,
            fold_results=_prepare_fold_results(raw_fold_results, metric=metric),
            include_metric_column=True,
        )
        for name, table in metric_tables.items():
            if name == "fold_results":
                continue
            aggregate_tables[name].append(table)
        plot_outputs = _write_plot(metric_tables["elo"], output_dir, asset_path, metric=metric)
        for name, path in plot_outputs.items():
            outputs[f"{metric}.{name}"] = path
        rows.append(
            {
                "metric": metric,
                "direction": metric_direction(metric),
                "label": _metric_label(metric),
                "elo": str(output_dir / "elo_ratings.csv"),
                "pairwise": str(output_dir / "pairwise_win_rates.csv"),
                "ranks": str(output_dir / "rank_summary.csv"),
                "coverage": str(output_dir / "coverage_summary.csv"),
                "summary": str(output_dir / "method_summary.csv"),
                "fold_results": str(output_dir / "manuscript_fold_results_success.csv"),
                "figure_png": str(plot_outputs["figure_png"]),
            }
        )
    aggregate_paths = {
        "elo": output_dir / "elo_ratings.csv",
        "pairwise": output_dir / "pairwise_win_rates.csv",
        "ranks": output_dir / "rank_summary.csv",
        "coverage": output_dir / "coverage_summary.csv",
        "summary": output_dir / "method_summary.csv",
    }
    fold_results_path = output_dir / "manuscript_fold_results_success.csv"
    _prepare_metric_suite_fold_results(raw_fold_results, metrics=metrics).to_csv(fold_results_path, index=False)
    outputs["fold_results"] = fold_results_path
    for name, path in aggregate_paths.items():
        pd.concat(aggregate_tables[name], ignore_index=True, sort=False).to_csv(path, index=False)
        outputs[name] = path
    index_path = output_dir / "metric_suite_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    outputs["metric_suite_index"] = index_path
    if eligibility_path is not None:
        outputs["eligibility"] = eligibility_path
    if ineligible_path is not None:
        outputs["ineligible"] = ineligible_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript no-HPO Elo outputs from matrix fold results.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help="Dataset/model result root. Can be repeated. Defaults to the clinical no-HPO root.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Comparable metric to build. Can be repeated. Defaults to the manuscript metric suite.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000, help="Number of bootstrap draws for Elo confidence intervals.")
    parser.add_argument("--asset-path", type=Path, default=None)
    parser.add_argument("--no-asset", action="store_true", help="Do not copy the PNG into docs/assets.")
    parser.add_argument("--strict-coverage", action="store_true", help="Fail if any dataset/method pair is incomplete.")
    parser.add_argument(
        "--complete-eligible-only",
        action="store_true",
        help="Drop dataset/method pairs without complete successful metric coverage before building Elo.",
    )
    parser.add_argument("--expected-splits", type=int, default=15, help="Expected successful splits per dataset/method pair.")
    parser.add_argument("--list-metrics", action="store_true", help="List comparable metrics found in fold results and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dirs = args.input_dir or [DEFAULT_INPUT]
    input_dirs = [path if path.is_absolute() else repo_root / path for path in input_dirs]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    if args.list_metrics:
        metrics = _available_metrics(_load_raw_fold_results(input_dirs))
        for metric in metrics:
            print(f"{metric}\t{metric_direction(metric)}\t{_metric_label(metric)}")
        return
    raw_fold_results = _load_raw_fold_results(input_dirs)
    metrics = list(args.metric) if args.metric else _default_metrics_for_frame(raw_fold_results)
    if not metrics:
        raise ValueError("No comparable metrics found in fold results.")
    if args.no_asset:
        asset_dir = None
    elif args.asset_path is None:
        asset_dir = repo_root / DEFAULT_ASSET_DIR
    else:
        if len(metrics) > 1:
            raise ValueError("--asset-path can only be used with exactly one --metric.")
        asset_path = args.asset_path if args.asset_path.is_absolute() else repo_root / args.asset_path
        asset_dir = None
    if args.asset_path is not None or args.metric:
        if len(metrics) != 1:
            raise ValueError("Single-metric output requires exactly one metric.")
        if args.asset_path is None:
            asset_path = None if asset_dir is None else asset_dir / f"elo_manuscript_no_hpo_{_metric_stem(metrics[0])}.png"
        outputs = build_outputs(
            repo_root=repo_root,
            input_dir=input_dirs,
            output_dir=output_dir,
            asset_path=asset_path,
            metric=metrics[0],
            n_bootstrap=args.bootstrap,
            strict_coverage=args.strict_coverage,
        )
    else:
        outputs = build_metric_suite_outputs(
            repo_root=repo_root,
            input_dir=input_dirs,
            output_dir=output_dir,
            asset_dir=asset_dir,
            metrics=metrics,
            n_bootstrap=args.bootstrap,
            strict_coverage=args.strict_coverage,
            complete_eligible_only=args.complete_eligible_only,
            expected_splits=args.expected_splits,
        )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
