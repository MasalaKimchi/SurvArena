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
from survarena.evaluation.statistics import aggregate_rank_summary, elo_ratings, pairwise_win_rate


DEFAULT_INPUT = Path("results/cloud/manuscript_dataset_model")
DEFAULT_OUTPUT = Path("results/manuscript_elo")
DEFAULT_ASSET = Path("docs/assets/elo_manuscript_no_hpo_uno_c.png")
METRIC = "uno_c"

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


def _base_id(value: Any) -> str:
    return str(value).split("__", 1)[0]


def _display_name(method_id: str) -> str:
    return DISPLAY_NAMES.get(method_id, method_id.replace("_", " ").title())


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


def _load_fold_results(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*/*/*_fold_results.csv"))
    if not paths:
        raise ValueError(f"No fold result CSVs found under {input_dir}.")

    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)
    required = {"benchmark_id", "dataset_id", "method_id", "split_id", "seed", "hpo_mode", "status", METRIC}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fold results are missing required columns: {sorted(missing)}")

    successful = frame[frame["status"].eq("success") & frame[METRIC].notna()].copy()
    successful["dataset_id"] = successful["dataset_id"].map(_base_id)
    successful["split_id"] = successful["split_id"].map(_base_id)
    successful["benchmark_id"] = "manuscript_v1"
    successful["hpo_mode"] = "no_hpo"
    return successful


def _validate_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    coverage = (
        frame.groupby(["dataset_id", "method_id"], as_index=False)
        .agg(
            rows=("split_id", "size"),
            splits=("split_id", "nunique"),
            seeds=("seed", "nunique"),
            mean_uno_c=(METRIC, "mean"),
        )
        .sort_values(["dataset_id", "method_id"])
    )
    bad = coverage[(coverage["rows"] != 15) | (coverage["splits"] != 15)]
    if not bad.empty:
        raise ValueError("Manuscript fold coverage is incomplete:\n" + bad.to_string(index=False))
    return coverage


def _write_plot(elo: pd.DataFrame, output_dir: Path, asset_path: Path | None) -> dict[str, Path]:
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
    ax.set_xlabel("Elo rating from paired Uno C win rate")
    ax.set_title("Manuscript Benchmark Elo (No-HPO, Uno C)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    families = list(dict.fromkeys(plot_frame["family"].astype(str)))
    handles = [Patch(facecolor=FAMILY_COLORS.get(family, FAMILY_COLORS["unknown"]), label=family.title()) for family in families]
    ax.legend(handles=handles, ncol=min(4, len(handles)), loc="lower right", frameon=True)

    fig.tight_layout()
    png = output_dir / "elo_manuscript_no_hpo_uno_c.png"
    pdf = output_dir / "elo_manuscript_no_hpo_uno_c.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if asset_path is not None:
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(asset_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    outputs = {"figure_png": png, "figure_pdf": pdf}
    if asset_path is not None:
        outputs["readme_asset"] = asset_path
    return outputs


def build_outputs(*, repo_root: Path, input_dir: Path, output_dir: Path, asset_path: Path | None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = _load_fold_results(input_dir)
    coverage = _validate_coverage(fold_results)

    combined_path = output_dir / "manuscript_fold_results_success.csv"
    fold_results.to_csv(combined_path, index=False)

    method_ids = sorted(fold_results["method_id"].dropna().astype(str).unique())
    metadata = _load_method_metadata(repo_root, method_ids)

    elo = elo_ratings(fold_results, metric=METRIC, n_bootstrap=1000, seed=33).merge(metadata, on="method_id", how="left")
    elo_path = output_dir / "elo_ratings_uno_c.csv"
    elo.to_csv(elo_path, index=False)

    wins = pairwise_win_rate(fold_results, metric=METRIC)
    wins_path = output_dir / "pairwise_win_rate_uno_c.csv"
    wins.to_csv(wins_path, index=False)

    ranks = aggregate_rank_summary(fold_results, metric=METRIC)
    ranks_path = output_dir / "rank_summary_uno_c.csv"
    ranks.to_csv(ranks_path, index=False)

    coverage_path = output_dir / "coverage_summary.csv"
    coverage.to_csv(coverage_path, index=False)

    summary = (
        fold_results.groupby(["benchmark_id", "hpo_mode", "method_id"], as_index=False)
        .agg(
            successful_folds=("status", "size"),
            mean_uno_c=(METRIC, "mean"),
            median_uno_c=(METRIC, "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
            total_runtime_sec=("runtime_sec", "sum"),
        )
        .merge(metadata, on="method_id", how="left")
        .sort_values("mean_uno_c", ascending=False)
    )
    summary_path = output_dir / "method_summary.csv"
    summary.to_csv(summary_path, index=False)

    outputs = {
        "combined": combined_path,
        "elo": elo_path,
        "pairwise": wins_path,
        "ranks": ranks_path,
        "coverage": coverage_path,
        "summary": summary_path,
    }
    outputs.update(_write_plot(elo, output_dir, asset_path))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manuscript no-HPO Elo outputs from matrix fold results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-path", type=Path, default=DEFAULT_ASSET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    input_dir = args.input_dir if args.input_dir.is_absolute() else repo_root / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    asset_path = args.asset_path if args.asset_path.is_absolute() else repo_root / args.asset_path
    outputs = build_outputs(repo_root=repo_root, input_dir=input_dir, output_dir=output_dir, asset_path=asset_path)
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
