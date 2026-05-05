from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("results/local_feasible_hpo_v1_all/figures/elo_by_mode_uno_c.csv")
DEFAULT_OUTPUT = Path("docs/assets/local_feasible_hpo_elo_uno_c.png")
MODE_ORDER = ("no_hpo", "hpo")
MODE_LABELS = {"no_hpo": "No HPO", "hpo": "HPO"}
MODE_COLORS = {"no_hpo": "#4F8A8B", "hpo": "#D76F30"}
MODE_WIDTHS = {"no_hpo": 0.78, "hpo": 0.42}
Y_FLOOR = 1000.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the README paired no-HPO/HPO Elo figure.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to elo_by_mode_uno_c.csv.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=320, help="PNG resolution.")
    return parser.parse_args()


def _prepare_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "display_method",
        "hpo_mode",
        "elo_rating",
        "elo_rating_ci95_low",
        "elo_rating_ci95_high",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

    frame = frame[frame["hpo_mode"].isin(MODE_ORDER)].copy()
    if frame.empty:
        raise ValueError(f"No paired HPO/no-HPO rows found in {path}.")

    sort_values = frame.groupby("display_method", as_index=True)["elo_rating"].max().sort_values(ascending=False)
    frame["display_method"] = pd.Categorical(frame["display_method"], categories=list(sort_values.index), ordered=True)
    frame["hpo_mode"] = pd.Categorical(frame["hpo_mode"], categories=list(MODE_ORDER), ordered=True)
    return frame.sort_values(["display_method", "hpo_mode"]).reset_index(drop=True)


def _wrapped_labels(methods: list[str]) -> list[str]:
    return [
        method.replace("Gradient boosting survival", "Gradient boosting")
        .replace("Fast survival SVM", "Fast SVM")
        .replace("Extra survival trees", "Extra trees")
        for method in methods
    ]


def _plot(frame: pd.DataFrame, output_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    methods = list(frame["display_method"].cat.categories)
    x = np.arange(len(methods))
    y_max = math.ceil(float(frame["elo_rating_ci95_high"].max()) / 50.0) * 50.0

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2E3440",
            "axes.labelcolor": "#20242A",
            "xtick.color": "#20242A",
            "ytick.color": "#20242A",
            "font.size": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(19, 8.8))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.3)

    for mode in MODE_ORDER:
        mode_frame = frame[frame["hpo_mode"] == mode].set_index("display_method").reindex(methods)
        ratings = mode_frame["elo_rating"].astype(float).to_numpy()
        lows = mode_frame["elo_rating_ci95_low"].astype(float).to_numpy()
        highs = mode_frame["elo_rating_ci95_high"].astype(float).to_numpy()
        visible_ratings = np.clip(ratings, Y_FLOOR, None)
        visible_lows = np.clip(lows, Y_FLOOR, None)
        visible_highs = np.clip(highs, Y_FLOOR, None)
        lower_err = np.maximum(visible_ratings - visible_lows, 0.0)
        upper_err = np.maximum(visible_highs - visible_ratings, 0.0)

        ax.bar(
            x,
            visible_ratings - Y_FLOOR,
            bottom=Y_FLOOR,
            width=MODE_WIDTHS[mode],
            color=MODE_COLORS[mode],
            alpha=0.72 if mode == "no_hpo" else 0.92,
            edgecolor="white",
            linewidth=1.0,
            label=f"{MODE_LABELS[mode]} ({'wide' if mode == 'no_hpo' else 'narrow'})",
            zorder=2 if mode == "no_hpo" else 3,
        )
        ax.errorbar(
            x,
            visible_ratings,
            yerr=np.vstack([lower_err, upper_err]),
            fmt="none",
            ecolor="#27313A" if mode == "hpo" else "#4A5560",
            elinewidth=0.9 if mode == "hpo" else 0.75,
            capsize=2.5,
            alpha=0.75 if mode == "hpo" else 0.5,
            zorder=4,
        )

        clipped = ratings < Y_FLOOR
        if clipped.any():
            ax.scatter(
                x[clipped],
                np.full(int(clipped.sum()), Y_FLOOR + 14.0),
                marker="v",
                s=46,
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
            )

    ax.axhline(1500, color="#1F2933", linewidth=1.0, linestyle=(0, (4, 4)), alpha=0.7, zorder=1)
    ax.text(len(methods) - 0.4, 1508, "1500 baseline", ha="right", va="bottom", color="#39424E", fontsize=9)
    ax.set_ylim(Y_FLOOR, y_max + 35)
    ax.set_xlim(-0.65, len(methods) - 0.35)
    ax.set_ylabel("Elo rating")
    ax.set_title("Local feasible HPO benchmark Elo by model and tuning mode", fontsize=15, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(_wrapped_labels(methods), rotation=48, ha="right", rotation_mode="anchor", fontsize=8.6)
    ax.grid(axis="y", color="#D9DEE5", linewidth=0.85, alpha=0.85)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncols=2)
    fig.text(
        0.055,
        0.045,
        "Metric: Uno C-index. Bars share each model tick; no-HPO is wider, HPO is narrower. "
        "Whiskers show bootstrap 95% CIs. Downward marker indicates a value below the displayed axis floor.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4A5560",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    frame = _prepare_frame(args.input)
    _plot(frame, args.output, args.dpi)


if __name__ == "__main__":
    main()
