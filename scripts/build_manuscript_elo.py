from __future__ import annotations

import argparse
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


def _load_raw_fold_results(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*/*/*_fold_results.csv"))
    if not paths:
        raise ValueError(f"No fold result CSVs found under {input_dir}.")
    return _with_derived_comparable_metrics(pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False))


def _load_fold_results(input_dir: Path, *, metric: str) -> pd.DataFrame:
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
    input_dir: Path,
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
    coverage = _validate_coverage(fold_results, metric=metric, strict=strict_coverage)
    stem = _metric_stem(metric)

    combined_path = output_dir / f"manuscript_fold_results_success_{stem}.csv"
    fold_results.to_csv(combined_path, index=False)

    method_ids = sorted(fold_results["method_id"].dropna().astype(str).unique())
    metadata = _load_method_metadata(repo_root, method_ids)

    elo = elo_ratings(fold_results, metric=metric, n_bootstrap=n_bootstrap, seed=33).merge(
        metadata,
        on="method_id",
        how="left",
    )
    elo_path = output_dir / f"elo_ratings_{stem}.csv"
    elo.to_csv(elo_path, index=False)

    wins = pairwise_win_rate(fold_results, metric=metric)
    wins_path = output_dir / f"pairwise_win_rate_{stem}.csv"
    wins.to_csv(wins_path, index=False)

    ranks = aggregate_rank_summary(fold_results, metric=metric)
    ranks_path = output_dir / f"rank_summary_{stem}.csv"
    ranks.to_csv(ranks_path, index=False)

    coverage_path = output_dir / f"coverage_summary_{stem}.csv"
    coverage.to_csv(coverage_path, index=False)

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
    summary["metric"] = metric
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


def build_metric_suite_outputs(
    *,
    repo_root: Path,
    input_dir: Path,
    output_dir: Path,
    asset_dir: Path | None,
    metrics: list[str],
    n_bootstrap: int = 1000,
    strict_coverage: bool = False,
) -> dict[str, Path]:
    if not metrics:
        raise ValueError("No comparable metrics were requested or found in fold results.")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_fold_results = _load_raw_fold_results(input_dir)
    rows: list[dict[str, str]] = []
    outputs: dict[str, Path] = {}
    for metric in metrics:
        asset_path = None if asset_dir is None else asset_dir / f"elo_manuscript_no_hpo_{_metric_stem(metric)}.png"
        metric_outputs = _build_outputs_from_fold_results(
            repo_root=repo_root,
            output_dir=output_dir,
            asset_path=asset_path,
            metric=metric,
            n_bootstrap=n_bootstrap,
            strict_coverage=strict_coverage,
            fold_results=_prepare_fold_results(raw_fold_results, metric=metric),
        )
        for name, path in metric_outputs.items():
            outputs[f"{metric}.{name}"] = path
        rows.append(
            {
                "metric": metric,
                "direction": metric_direction(metric),
                "label": _metric_label(metric),
                "elo": str(metric_outputs["elo"]),
                "pairwise": str(metric_outputs["pairwise"]),
                "ranks": str(metric_outputs["ranks"]),
                "figure_png": str(metric_outputs["figure_png"]),
            }
        )
    index_path = output_dir / "metric_suite_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    outputs["metric_suite_index"] = index_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript no-HPO Elo outputs from matrix fold results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
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
    parser.add_argument("--list-metrics", action="store_true", help="List comparable metrics found in fold results and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else repo_root / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    if args.list_metrics:
        metrics = _available_metrics(_load_raw_fold_results(input_dir))
        for metric in metrics:
            print(f"{metric}\t{metric_direction(metric)}\t{_metric_label(metric)}")
        return
    raw_fold_results = _load_raw_fold_results(input_dir)
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
            input_dir=input_dir,
            output_dir=output_dir,
            asset_path=asset_path,
            metric=metrics[0],
            n_bootstrap=args.bootstrap,
            strict_coverage=args.strict_coverage,
        )
    else:
        outputs = build_metric_suite_outputs(
            repo_root=repo_root,
            input_dir=input_dir,
            output_dir=output_dir,
            asset_dir=asset_dir,
            metrics=metrics,
            n_bootstrap=args.bootstrap,
            strict_coverage=args.strict_coverage,
        )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
