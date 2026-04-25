from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from survarena.evaluation.statistics import (
    aggregate_rank_summary,
    bootstrap_metric_ci,
    critical_difference_summary,
    elo_ratings,
    failure_summary,
    pairwise_significance,
    pairwise_win_rate,
)
from survarena.logging.export_shared import (
    BENCHMARK_METRIC_COLUMNS,
    CORE_METRIC_COLUMNS,
    GOVERNANCE_COLUMNS,
    MANUSCRIPT_METRIC_COLUMNS,
    MANUSCRIPT_REPORT_SCHEMA_VERSION,
    benchmark_label,
    expand_dynamic_metric_columns,
    group_keys_with_hpo_mode,
    parity_gated_frame,
    unique_in_order,
)
from survarena.logging.tracker import write_json


def _build_consolidated_benchmark_report(
    leaderboard: pd.DataFrame,
    *,
    primary_metric: str,
    rank_summary: pd.DataFrame,
    elo: pd.DataFrame,
    ci: pd.DataFrame,
    cd_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Merge leaderboard rows with cross-dataset rank, ELO, bootstrap CI, and CD summaries."""
    keys = group_keys_with_hpo_mode(leaderboard, ["benchmark_id", "method_id"])
    rs = rank_summary.rename(
        columns={
            "mean_rank": "agg_mean_rank",
            "median_rank": "agg_median_rank",
            "mean_score": f"agg_mean_score_{primary_metric}",
            "median_score": f"agg_median_score_{primary_metric}",
            "datasets_evaluated": "agg_datasets_in_ranking",
        }
    )
    drop_elo = [c for c in ("metric", "initial_rating", "k_factor") if c in elo.columns]
    elo_sub = elo.drop(columns=drop_elo, errors="ignore").rename(
        columns={"elo_rating": "agg_elo_rating", "elo_matches": "agg_elo_matches"}
    )
    ci_renames = {
        "mean": f"agg_bootstrap_mean_{primary_metric}",
        "ci95_low": f"agg_ci95_low_{primary_metric}",
        "ci95_high": f"agg_ci95_high_{primary_metric}",
        "n": "agg_bootstrap_n",
    }
    ci_sub = ci.drop(columns=[c for c in ("metric",) if c in ci.columns], errors="ignore")
    ci_sub = ci_sub.rename(columns={k: v for k, v in ci_renames.items() if k in ci_sub.columns})
    cd_renames = {
        "average_rank": "agg_cd_average_rank",
        "critical_difference": "agg_critical_difference",
        "n_methods": "agg_cd_n_methods",
        "n_datasets": "agg_cd_n_datasets",
    }
    cd_sub = cd_summary.drop(columns=[c for c in ("metric",) if c in cd_summary.columns], errors="ignore")
    cd_sub = cd_sub.rename(columns={k: v for k, v in cd_renames.items() if k in cd_sub.columns})

    out = leaderboard.merge(rs, on=keys, how="left")
    elo_cols = [c for c in ("agg_elo_rating", "agg_elo_matches") if c in elo_sub.columns]
    if elo_cols:
        out = out.merge(elo_sub[keys + elo_cols], on=keys, how="left")
    ci_merge = [c for c in ci_sub.columns if c not in keys]
    if ci_merge:
        out = out.merge(ci_sub[keys + ci_merge], on=keys, how="left")
    cd_merge = [c for c in cd_sub.columns if c not in keys]
    if cd_merge:
        out = out.merge(cd_sub[keys + cd_merge], on=keys, how="left")
    return out


def _plot_pairwise_matrix(
    sub: pd.DataFrame,
    *,
    value_col: str,
    path: Path,
    title: str,
    cmap: str = "viridis",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if sub.empty or value_col not in sub.columns:
        return
    methods = sorted(set(sub["method_id"].astype(str)) | set(sub["opponent_method_id"].astype(str)))
    n = len(methods)
    mat = np.full((n, n), np.nan, dtype=float)
    pos = {m: i for i, m in enumerate(methods)}
    for _, row in sub.iterrows():
        i, j = pos[str(row["method_id"])], pos[str(row["opponent_method_id"])]
        mat[i, j] = float(row[value_col])
    max_dim = 16
    fig, ax = plt.subplots(figsize=(min(max_dim, 0.4 * n + 4), min(max_dim, 0.38 * n + 3)))
    finite = mat[np.isfinite(mat)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=90, fontsize=7)
    ax.set_yticklabels(methods, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _export_pairwise_figure_bundle(
    pairwise: pd.DataFrame,
    pairwise_sig: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    primary_metric: str,
) -> dict[str, str]:
    out_paths: dict[str, str] = {}
    fig_dir = output_dir / "figures"

    def _plots_for_frame(
        frame: pd.DataFrame,
        *,
        value_col: str,
        stem: str,
        title_fn: Callable[[object, str, str], str],
        cmap: str = "viridis",
    ) -> None:
        if frame.empty or value_col not in frame.columns:
            return
        gcols = ["benchmark_id"] + (["hpo_mode"] if "hpo_mode" in frame.columns else [])
        for gkey, sub in frame.groupby(gcols):
            if isinstance(gkey, tuple) and len(gkey) > 1:
                b_id, h_mode = gkey[0], gkey[1]
                suffix = f"_{h_mode}"
            else:
                b_id = gkey[0] if isinstance(gkey, tuple) else gkey
                suffix = ""
            title = title_fn(b_id, suffix, primary_metric)
            pth = fig_dir / f"{prefix}_fig_{stem}{suffix}.png"
            _plot_pairwise_matrix(sub, value_col=value_col, path=pth, title=title, cmap=cmap)
            out_paths[f"fig_{stem}{suffix}"] = str(pth)

    if not pairwise.empty:
        _plots_for_frame(
            pairwise,
            value_col="win_rate",
            stem="pairwise_win_rate",
            title_fn=lambda b, s, m: f"{b} pairwise win rate ({m})" + (f" [{s.strip('_')}]" if s else ""),
        )
    if not pairwise_sig.empty:
        sig = pairwise_sig.copy()
        pcol = "p_value_corrected" if "p_value_corrected" in sig.columns else "p_value"
        sig = sig.assign(neg_log10_p=-np.log10(np.clip(sig[pcol].astype(float), 1e-300, 1.0)))
        _plots_for_frame(
            sig,
            value_col="neg_log10_p",
            stem="pairwise_significance",
            title_fn=lambda b, s, m: f"{b} -log10 p (pairwise) {m}" + (f" [{s.strip('_')}]" if s else ""),
            cmap="magma_r",
        )
    return out_paths


def export_manuscript_comparison(
    root: Path,
    leaderboard: pd.DataFrame,
    *,
    primary_metric: str,
    fold_results: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    file_prefix: str | None = None,
    artifact_layout: str = "full",
) -> dict[str, str]:
    prefix = file_prefix or benchmark_label(leaderboard)
    if output_dir is None:
        output_dir = root / "results" / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    significance_source = fold_results if fold_results is not None else leaderboard
    significance_source = parity_gated_frame(significance_source)
    if fold_results is not None:
        claim_metric_cols = unique_in_order(
            BENCHMARK_METRIC_COLUMNS + GOVERNANCE_COLUMNS + expand_dynamic_metric_columns(significance_source)
        )
        available_claim_metric_cols = [col for col in claim_metric_cols if col in significance_source.columns]
        if significance_source.empty:
            comparative_leaderboard = leaderboard.iloc[0:0].copy()
        else:
            cl_keys = group_keys_with_hpo_mode(
                significance_source,
                ["benchmark_id", "dataset_id", "method_id"],
            )
            comparative_leaderboard = significance_source.groupby(cl_keys, as_index=False)[
                available_claim_metric_cols
            ].mean(numeric_only=True)
    else:
        comparative_leaderboard = parity_gated_frame(leaderboard)

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

    layout = str(artifact_layout).lower()
    if layout == "compact":
        consolidated = _build_consolidated_benchmark_report(
            leaderboard,
            primary_metric=primary_metric,
            rank_summary=rank_summary,
            elo=elo,
            ci=ci,
            cd_summary=cd_summary,
        )
        report_path = output_dir / f"{prefix}_report.csv"
        consolidated.to_csv(report_path, index=False)
        figure_paths = _export_pairwise_figure_bundle(
            pairwise,
            pairwise_sig,
            output_dir,
            prefix,
            primary_metric=primary_metric,
        )
        paths: dict[str, str] = {
            "consolidated_report": str(report_path),
            **figure_paths,
        }
        write_json(
            output_dir / f"{prefix}_manuscript_summary.json",
            {
                "schema_version": MANUSCRIPT_REPORT_SCHEMA_VERSION,
                "primary_metric": primary_metric,
                "artifact_layout": "compact",
                "comparison_files": paths,
                "rank_summary_records": rank_summary.to_dict(orient="records"),
                "elo_rating_records": elo.to_dict(orient="records"),
                "pairwise_significance_records": pairwise_sig.to_dict(orient="records"),
                "missing_metric_summary": missing.to_dict(orient="records"),
                "failure_summary_records": failures.to_dict(orient="records"),
                "multiple_comparison_summary_records": multiple_summary.to_dict(orient="records"),
                "parity_gate": {
                    "applied": "parity_eligible" in significance_source.columns,
                    "comparison_rows": int(len(significance_source)),
                },
            },
        )
        return paths

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
            "artifact_layout": "full",
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
