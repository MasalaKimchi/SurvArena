from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


DEFAULT_ELO = Path("results/unified_foundation_elo/unified_elo_ratings_uno_c.csv")
DEFAULT_OUTPUT_DIR = Path("results/unified_foundation_elo")
NO_HPO_MODE = "no_hpo"
HPO_MODE = "hpo_reference"

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
    "tabpfn_survival_classifier": "TabPFN Classifier",
    "tabpfn_survival_regressor": "TabPFN Regressor",
    "weibull_aft": "Weibull AFT",
    "xgboost_aft": "XGBoost AFT",
    "xgboost_cox": "XGBoost Cox",
}

FAMILY_COLORS = {
    "foundation": ("#eadcf8", "#8f5cc7"),
    "classical": ("#dbeafe", "#2563eb"),
    "tree": ("#dcfce7", "#059669"),
    "boosting": ("#ffedd5", "#d97706"),
    "deep": ("#fee2e2", "#dc2626"),
    "automl": ("#cffafe", "#0891b2"),
    "unknown": ("#e5e7eb", "#6b7280"),
}


def _display_name(method_id: str) -> str:
    if method_id in DISPLAY_NAMES:
        return DISPLAY_NAMES[method_id]
    return method_id.replace("_", " ").title()


def _paired_elo_frame(elo: pd.DataFrame) -> pd.DataFrame:
    required = {"hpo_mode", "method_id", "elo_rating", "elo_rating_ci95_low", "elo_rating_ci95_high", "family"}
    missing = required.difference(elo.columns)
    if missing:
        raise ValueError(f"Elo file is missing required columns: {sorted(missing)}")

    no_hpo = elo[elo["hpo_mode"].eq(NO_HPO_MODE)].copy()
    hpo = elo[elo["hpo_mode"].eq(HPO_MODE)].copy()
    paired = hpo.merge(
        no_hpo,
        on="method_id",
        suffixes=("_hpo", "_no_hpo"),
        how="inner",
    )
    if paired.empty:
        raise ValueError("No methods have both no-HPO and HPO-reference Elo rows.")

    paired["family"] = paired["family_hpo"].fillna(paired["family_no_hpo"]).fillna("unknown")
    paired["display_name"] = paired["method_id"].map(_display_name)
    paired = paired.sort_values("elo_rating_hpo", ascending=True).reset_index(drop=True)
    return paired


def build_figure(elo_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = _paired_elo_frame(pd.read_csv(elo_path))

    fig_width = max(16.0, 0.58 * len(paired))
    fig, ax = plt.subplots(figsize=(fig_width, 7.2))
    x = range(len(paired))

    no_hpo_colors = [FAMILY_COLORS.get(str(family), FAMILY_COLORS["unknown"])[0] for family in paired["family"]]
    hpo_colors = [FAMILY_COLORS.get(str(family), FAMILY_COLORS["unknown"])[1] for family in paired["family"]]

    no_hpo_err = [
        (paired["elo_rating_no_hpo"] - paired["elo_rating_ci95_low_no_hpo"]).clip(lower=0).to_numpy(),
        (paired["elo_rating_ci95_high_no_hpo"] - paired["elo_rating_no_hpo"]).clip(lower=0).to_numpy(),
    ]
    hpo_err = [
        (paired["elo_rating_hpo"] - paired["elo_rating_ci95_low_hpo"]).clip(lower=0).to_numpy(),
        (paired["elo_rating_ci95_high_hpo"] - paired["elo_rating_hpo"]).clip(lower=0).to_numpy(),
    ]

    ax.bar(
        x,
        paired["elo_rating_hpo"],
        width=0.74,
        color=hpo_colors,
        alpha=0.86,
        edgecolor="white",
        linewidth=1.0,
        label="HPO",
        zorder=2,
    )
    ax.errorbar(
        list(x),
        paired["elo_rating_hpo"],
        yerr=hpo_err,
        fmt="none",
        ecolor="#111827",
        elinewidth=0.9,
        capsize=2,
        alpha=0.75,
        zorder=4,
    )
    ax.bar(
        x,
        paired["elo_rating_no_hpo"],
        width=0.48,
        color=no_hpo_colors,
        alpha=0.92,
        edgecolor="white",
        linewidth=1.0,
        label="No HPO",
        zorder=3,
    )
    ax.errorbar(
        list(x),
        paired["elo_rating_no_hpo"],
        yerr=no_hpo_err,
        fmt="none",
        ecolor="#64748b",
        elinewidth=0.75,
        capsize=2,
        alpha=0.65,
        zorder=5,
    )

    ax.axhline(1500, color="#111827", linestyle="--", linewidth=1.4)
    ax.text(0.005, 1500 + 10, "Elo 1500", transform=ax.get_yaxis_transform(), fontsize=11, color="#111827")

    ax.set_ylabel("Elo")
    ax.set_xticks(list(x))
    ax.set_xticklabels(paired["display_name"], rotation=48, ha="right", fontsize=9)
    ax.set_title("Unified Survival Elo: No-HPO vs HPO", fontsize=15, fontweight="bold")
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.9)
    ax.set_axisbelow(True)

    mode_handles = [
        Patch(facecolor="#94a3b8", alpha=0.92, label="No HPO (narrow, light)"),
        Patch(facecolor="#334155", alpha=0.86, label="HPO (wide, dark)"),
    ]
    family_handles = [
        Patch(facecolor=dark, alpha=0.86, label=family.title())
        for family, (_, dark) in FAMILY_COLORS.items()
        if family in set(paired["family"].astype(str))
    ]
    first_legend = ax.legend(handles=mode_handles, ncol=2, loc="upper left", frameon=True)
    ax.add_artist(first_legend)
    ax.legend(handles=family_handles, ncol=min(6, len(family_handles)), loc="upper center", frameon=True)

    fig.tight_layout()
    png = output_dir / "unified_elo_paired_hpo_no_hpo_uno_c.png"
    pdf = output_dir / "unified_elo_paired_hpo_no_hpo_uno_c.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"figure_png": png, "figure_pdf": pdf}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired no-HPO/HPO unified Elo figure.")
    parser.add_argument("--elo", type=Path, default=DEFAULT_ELO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_figure(args.elo, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
