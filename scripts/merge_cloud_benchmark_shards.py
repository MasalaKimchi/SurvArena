from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from survarena.evaluation.statistics import elo_ratings, pairwise_win_rate
from survarena.logging.export import export_hpo_budget_summary, export_leaderboard


DISPLAY_NAMES = {
    "coxph": "CoxPH",
    "coxnet": "CoxNet",
    "rsf": "Random Survival Forest",
    "extra_survival_trees": "Extra Survival Trees",
    "fast_survival_svm": "Fast Survival SVM",
    "deepsurv": "DeepSurv",
    "deepsurv_moco": "DeepSurv MoCo",
    "deephit_single": "DeepHit",
    "gradient_boosting_survival": "Gradient Boosting",
    "componentwise_gradient_boosting": "Component GB",
    "xgboost_cox": "XGBoost Cox",
    "xgboost_aft": "XGBoost AFT",
    "catboost_cox": "CatBoost Cox",
    "catboost_survival_aft": "CatBoost AFT",
}


def _display_name(method_id: str) -> str:
    return DISPLAY_NAMES.get(method_id, method_id.replace("_", " ").title())


def _read_fold_results(shards_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(shards_dir.glob("*/*/*_fold_results.csv")):
        if path.name.startswith("combined_"):
            continue
        frame = pd.read_csv(path)
        if not frame.empty and "method_id" in frame.columns:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No shard fold-results CSVs found under {shards_dir}.")
    return pd.concat(frames, ignore_index=True, sort=False)


def _write_paired_elo_figure(elo: pd.DataFrame, output_dir: Path) -> Path:
    no_hpo = elo[elo["hpo_mode"].eq("no_hpo")].copy()
    hpo = elo[elo["hpo_mode"].eq("hpo")].copy()
    paired = hpo.merge(no_hpo, on="method_id", suffixes=("_hpo", "_no_hpo"), how="inner")
    paired = paired.sort_values("elo_rating_hpo", ascending=True).reset_index(drop=True)
    if paired.empty:
        raise ValueError("No paired no-HPO/HPO Elo rows available for plotting.")

    x = range(len(paired))
    fig, ax = plt.subplots(figsize=(max(14, 0.58 * len(paired)), 7.0))
    ax.bar(x, paired["elo_rating_hpo"], width=0.74, color="#334155", alpha=0.86, label="HPO", zorder=2)
    ax.bar(x, paired["elo_rating_no_hpo"], width=0.48, color="#93c5fd", alpha=0.90, label="No HPO", zorder=3)
    ax.axhline(1500, color="#111827", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Elo")
    ax.set_title("Cloud HPO Elo: No-HPO vs HPO", fontsize=15, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_display_name(str(method)) for method in paired["method_id"]], rotation=48, ha="right")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.9)
    ax.legend(ncol=2)
    fig.tight_layout()
    path = output_dir / "paired_hpo_no_hpo_elo_uno_c.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "paired_hpo_no_hpo_elo_uno_c.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def merge_shards(*, shards_dir: Path, output_dir: Path, primary_metric: str = "uno_c") -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = _read_fold_results(shards_dir)
    combined_path = output_dir / "combined_fold_results_all.csv"
    combined.to_csv(combined_path, index=False)

    success = combined[combined["status"].eq("success") & combined[primary_metric].notna()].copy()
    success_path = output_dir / "combined_fold_results_success.csv"
    success.to_csv(success_path, index=False)

    export_leaderboard(Path("."), success, primary_metric=primary_metric, output_dir=output_dir, file_prefix="combined")
    leaderboard_path = output_dir / "combined_leaderboard_all.csv"
    (output_dir / "combined_leaderboard.csv").replace(leaderboard_path)

    export_hpo_budget_summary(Path("."), combined, output_dir=output_dir, file_prefix="combined")
    hpo_summary_path = output_dir / "hpo_budget_summary.csv"
    (output_dir / "combined_hpo_budget_summary.csv").replace(hpo_summary_path)

    elo = elo_ratings(success, metric=primary_metric, n_bootstrap=1000, seed=33)
    elo_path = output_dir / "elo_ratings_uno_c.csv"
    elo.to_csv(elo_path, index=False)

    pairwise_path = output_dir / "pairwise_win_rate_uno_c.csv"
    pairwise_win_rate(success, metric=primary_metric).to_csv(pairwise_path, index=False)
    figure_path = _write_paired_elo_figure(elo, output_dir)
    return {
        "combined_all": combined_path,
        "combined_success": success_path,
        "leaderboard": leaderboard_path,
        "hpo_budget_summary": hpo_summary_path,
        "elo": elo_path,
        "pairwise": pairwise_path,
        "figure": figure_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge per-method cloud benchmark shards.")
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-metric", default="uno_c")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = merge_shards(shards_dir=args.shards_dir, output_dir=args.output_dir, primary_metric=args.primary_metric)
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
