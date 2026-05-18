from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from survarena.config import read_yaml
from survarena.evaluation.statistics import elo_ratings, pairwise_win_rate


DEFAULT_CONVENTIONAL = Path("results/local_feasible_hpo_v1_all/combined_fold_results_success.csv")
DEFAULT_FOUNDATION = Path("results/foundation_elo_v1/combined_fold_results.csv")
DEFAULT_OUTPUT = Path("results/foundation_elo")
FOUNDATION_METHODS = ("tabpfn_survival",)
BENCHMARK_ID = "foundation_vs_conventional"
MODE_NO_HPO = "no_hpo"
MODE_HPO_REFERENCE = "hpo_reference"


def _base_id(value: Any) -> str:
    return str(value).split("__", 1)[0]


def _load_method_metadata(repo_root: Path, method_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method_id in method_ids:
        path = repo_root / "configs" / "methods" / f"{method_id}.yaml"
        cfg = read_yaml(path) if path.exists() else {}
        rows.append(
            {
                "method_id": method_id,
                "family": cfg.get("family", "unknown"),
                "display_name": cfg.get("display_name", method_id),
                "library": cfg.get("library", ""),
            }
        )
    return pd.DataFrame(rows)


def _prepare_frame(frame: pd.DataFrame, *, source_run: str) -> pd.DataFrame:
    out = frame.copy()
    out["source_run"] = source_run
    out["base_dataset_id"] = out["dataset_id"].map(_base_id)
    out["base_split_id"] = out["split_id"].map(_base_id)
    out["unit_id"] = (
        out["base_dataset_id"].astype(str)
        + "|"
        + out["base_split_id"].astype(str)
        + "|"
        + out["seed"].astype(int).astype(str)
    )
    return out


def _build_mode_frame(
    conventional: pd.DataFrame,
    foundation: pd.DataFrame,
    *,
    conventional_mode: str,
    output_mode: str,
) -> pd.DataFrame:
    conv = conventional[
        conventional["status"].eq("success")
        & conventional["hpo_mode"].eq(conventional_mode)
        & conventional["uno_c"].notna()
    ].copy()
    found = foundation[
        foundation["status"].eq("success")
        & foundation["method_id"].isin(FOUNDATION_METHODS)
        & foundation["uno_c"].notna()
    ].copy()

    shared_units = sorted(set(conv["unit_id"]).intersection(set(found["unit_id"])))
    if not shared_units:
        raise ValueError(f"No shared units for conventional mode '{conventional_mode}'.")

    conv = conv[conv["unit_id"].isin(shared_units)].copy()
    found = found[found["unit_id"].isin(shared_units)].copy()
    conv["benchmark_id"] = BENCHMARK_ID
    found["benchmark_id"] = BENCHMARK_ID
    conv["hpo_mode"] = output_mode
    found["hpo_mode"] = output_mode
    return pd.concat([conv, found], ignore_index=True, sort=False)


def _write_plot(elo: pd.DataFrame, output_dir: Path) -> None:
    plot_frame = elo.sort_values(["hpo_mode", "elo_rating"], ascending=[True, True]).copy()
    modes = list(plot_frame["hpo_mode"].drop_duplicates())
    fig, axes = plt.subplots(1, len(modes), figsize=(15, max(7, 0.32 * plot_frame["method_id"].nunique())), sharex=True)
    if len(modes) == 1:
        axes = [axes]

    family_colors = {
        "foundation": "#7b2cbf",
        "classical": "#2563eb",
        "tree": "#059669",
        "boosting": "#d97706",
        "deep": "#dc2626",
        "automl": "#0891b2",
        "unknown": "#6b7280",
    }

    for axis, mode in zip(axes, modes, strict=True):
        sub = plot_frame[plot_frame["hpo_mode"].eq(mode)].copy()
        y = range(len(sub))
        colors = [family_colors.get(str(family), family_colors["unknown"]) for family in sub["family"]]
        low = (sub["elo_rating"] - sub["elo_rating_ci95_low"]).clip(lower=0)
        high = (sub["elo_rating_ci95_high"] - sub["elo_rating"]).clip(lower=0)
        axis.barh(y, sub["elo_rating"], color=colors, alpha=0.88)
        axis.errorbar(
            sub["elo_rating"],
            list(y),
            xerr=[low.to_numpy(), high.to_numpy()],
            fmt="none",
            ecolor="#111827",
            elinewidth=0.8,
            capsize=2,
        )
        axis.axvline(1500, color="#374151", linestyle="--", linewidth=0.9)
        axis.set_yticks(list(y))
        axis.set_yticklabels(sub["method_id"])
        axis.set_title(mode.replace("_", " ").title())
        axis.set_xlabel("Elo rating from paired Uno C win rate")
        axis.grid(axis="x", color="#e5e7eb", linewidth=0.8)

    fig.suptitle("Foundation vs Conventional Elo", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "elo_by_mode_uno_c.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "elo_by_mode_uno_c.pdf", bbox_inches="tight")
    plt.close(fig)


def build_unified_outputs(
    *,
    repo_root: Path,
    conventional_path: Path,
    foundation_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    conventional = _prepare_frame(pd.read_csv(conventional_path), source_run="local_feasible_hpo_v1_all")
    foundation = _prepare_frame(pd.read_csv(foundation_path), source_run="foundation_elo_v1")

    no_hpo = _build_mode_frame(conventional, foundation, conventional_mode="no_hpo", output_mode=MODE_NO_HPO)
    hpo_reference = _build_mode_frame(
        conventional,
        foundation,
        conventional_mode="hpo",
        output_mode=MODE_HPO_REFERENCE,
    )
    combined = pd.concat([no_hpo, hpo_reference], ignore_index=True, sort=False)
    combined_path = output_dir / "fold_results.csv"
    combined.to_csv(combined_path, index=False)

    method_ids = sorted(combined["method_id"].dropna().astype(str).unique())
    metadata = _load_method_metadata(repo_root, method_ids)

    elo = elo_ratings(combined, metric="uno_c", n_bootstrap=1000, seed=33)
    elo = elo.merge(metadata, on="method_id", how="left")
    elo_path = output_dir / "elo_ratings_uno_c.csv"
    elo.to_csv(elo_path, index=False)

    wins = pairwise_win_rate(combined, metric="uno_c")
    wins_path = output_dir / "pairwise_win_rate_uno_c.csv"
    wins.to_csv(wins_path, index=False)

    summary = (
        combined.groupby(["hpo_mode", "method_id"], as_index=False)
        .agg(
            successful_folds=("status", "size"),
            mean_uno_c=("uno_c", "mean"),
            median_uno_c=("uno_c", "median"),
            mean_runtime_sec=("runtime_sec", "mean"),
            total_runtime_sec=("runtime_sec", "sum"),
        )
        .merge(metadata, on="method_id", how="left")
    )
    summary_path = output_dir / "method_summary.csv"
    summary.to_csv(summary_path, index=False)

    coverage = (
        combined.groupby(["hpo_mode", "source_run", "base_dataset_id"], as_index=False)
        .agg(
            rows=("method_id", "size"),
            methods=("method_id", "nunique"),
            units=("unit_id", "nunique"),
        )
        .sort_values(["hpo_mode", "source_run", "base_dataset_id"])
    )
    coverage_path = output_dir / "coverage_summary.csv"
    coverage.to_csv(coverage_path, index=False)

    _write_plot(elo, output_dir)

    return {
        "combined": combined_path,
        "elo": elo_path,
        "pairwise": wins_path,
        "summary": summary_path,
        "coverage": coverage_path,
        "figure_png": output_dir / "elo_by_mode_uno_c.png",
        "figure_pdf": output_dir / "elo_by_mode_uno_c.pdf",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified foundation/conventional Elo outputs.")
    parser.add_argument("--conventional", type=Path, default=DEFAULT_CONVENTIONAL)
    parser.add_argument("--foundation", type=Path, default=DEFAULT_FOUNDATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    conventional = args.conventional if args.conventional.is_absolute() else repo_root / args.conventional
    foundation = args.foundation if args.foundation.is_absolute() else repo_root / args.foundation
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    outputs = build_unified_outputs(
        repo_root=repo_root,
        conventional_path=conventional,
        foundation_path=foundation,
        output_dir=output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
