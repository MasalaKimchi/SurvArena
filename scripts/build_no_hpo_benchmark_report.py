from __future__ import annotations

import argparse
from collections import Counter
from html import escape
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

from survarena.config import read_yaml


STATUS_ORDER = ["missing", "attempted_incomplete", "failed", "partial", "complete"]
STATUS_COLORS = {
    "missing": "#F4F5F7",
    "attempted_incomplete": "#FFEA8F",
    "failed": "#F5BACC",
    "partial": "#FFBDA1",
    "complete": "#A3D576",
}
STATUS_LABELS = {
    "missing": "Missing",
    "attempted_incomplete": "Attempted, incomplete",
    "failed": "Attempted, no successful folds",
    "partial": "Partially successful",
    "complete": "15/15 successful folds",
}


def classify_cell(*, success_splits: int, attempted_splits: int, expected_splits: int) -> str:
    if success_splits == expected_splits:
        return "complete"
    if success_splits > 0:
        return "partial"
    if attempted_splits >= expected_splits:
        return "failed"
    if attempted_splits > 0:
        return "attempted_incomplete"
    return "missing"


def inventory_matrix(config_path: Path, result_root: Path, *, track: str) -> pd.DataFrame:
    config = read_yaml(config_path)
    expected_splits = int(config["outer_folds"]) * int(config["outer_repeats"])
    rows: list[dict[str, Any]] = []
    for dataset_id in config["datasets"]:
        for method_id in config["methods"]:
            cell_dir = result_root / str(dataset_id) / str(method_id)
            fold_path = cell_dir / f"{method_id}_fold_results.csv"
            frame = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
            if frame.empty:
                success_splits = 0
                attempted_splits = 0
            else:
                success = frame[frame["status"].astype(str).eq("success")] if "status" in frame else frame
                success_splits = int(success["split_id"].nunique()) if "split_id" in success else int(success.shape[0])
                attempted_splits = int(frame["split_id"].nunique()) if "split_id" in frame else int(frame.shape[0])
            rows.append(
                {
                    "track": track,
                    "dataset_id": str(dataset_id),
                    "method_id": str(method_id),
                    "expected_splits": expected_splits,
                    "attempted_splits": attempted_splits,
                    "success_splits": success_splits,
                    "status": classify_cell(
                        success_splits=success_splits,
                        attempted_splits=attempted_splits,
                        expected_splits=expected_splits,
                    ),
                    "fold_results": str(fold_path),
                }
            )
    return pd.DataFrame(rows)


def failure_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    failures = inventory[inventory["status"].isin({"failed", "partial", "attempted_incomplete"})].copy()
    if failures.empty:
        return pd.DataFrame(columns=["track", "method_id", "status", "dataset_count"])
    return (
        failures.groupby(["track", "method_id", "status"], as_index=False)
        .agg(dataset_count=("dataset_id", "nunique"))
        .sort_values(["track", "dataset_count", "method_id"], ascending=[True, False, True])
    )


def render_coverage_heatmap(inventory: pd.DataFrame, output_path: Path, *, title: str, subtitle: str) -> None:
    datasets = list(dict.fromkeys(inventory["dataset_id"].astype(str)))
    methods = list(dict.fromkeys(inventory["method_id"].astype(str)))
    status_codes = {status: index for index, status in enumerate(STATUS_ORDER)}
    matrix = (
        inventory.assign(status_code=inventory["status"].map(status_codes))
        .pivot(index="dataset_id", columns="method_id", values="status_code")
        .reindex(index=datasets, columns=methods)
    )
    palette = [STATUS_COLORS[status] for status in STATUS_ORDER]
    sns.set_theme(style="white", font_scale=0.8)
    width = max(14.0, len(methods) * 0.46)
    height = max(4.5, len(datasets) * 0.58 + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), facecolor="#FCFCFD")
    sns.heatmap(
        matrix,
        cmap=sns.color_palette(palette, as_cmap=True),
        vmin=-0.5,
        vmax=len(STATUS_ORDER) - 0.5,
        cbar=False,
        linewidths=0.8,
        linecolor="#FFFFFF",
        ax=ax,
    )
    fig.text(0.125, 0.98, title, ha="left", va="top", fontsize=14, fontweight="semibold", color="#1F2430")
    fig.text(0.125, 0.945, subtitle, ha="left", va="top", fontsize=9, color="#6F768A")
    ax.set_xlabel("Method", color="#1F2430")
    ax.set_ylabel("Dataset", color="#1F2430")
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    handles = [
        Patch(facecolor=STATUS_COLORS[status], edgecolor="#D7DBE7", label=STATUS_LABELS[status])
        for status in STATUS_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _uno_elo_rows(elo_path: Path, *, limit: int = 10) -> pd.DataFrame:
    if not elo_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(elo_path)
    if "metric" in frame:
        frame = frame[frame["metric"].eq("uno_c")]
    return frame.sort_values("elo_rating", ascending=False).head(limit)


def _table_html(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if frame.empty:
        return "<p>No eligible rows were available.</p>"
    header = "".join(f"<th>{escape(label)}</th>" for _, label in columns)
    body = []
    for row in frame.to_dict(orient="records"):
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:,.1f}"
            cells.append(f"<td>{escape(str(value))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def build_html(
    *,
    inventory: pd.DataFrame,
    failures: pd.DataFrame,
    clinical_elo: pd.DataFrame,
    genomics_elo: pd.DataFrame,
    clinical_eligible_pairs: int,
    genomics_eligible_pairs: int,
    output_path: Path,
) -> None:
    counts = Counter(inventory["status"].astype(str))
    total_pairs = int(inventory.shape[0])
    attempted_pairs = total_pairs - counts["missing"]
    complete_pairs = counts["complete"]
    clinical = inventory[inventory["track"].eq("Clinical")]
    genomics = inventory[inventory["track"].eq("Genomics")]
    clinical_complete = int(clinical["status"].eq("complete").sum())
    genomics_complete = int(genomics["status"].eq("complete").sum())
    top_clinical = clinical_elo.iloc[0]["display_name"] if not clinical_elo.empty else "not available"
    top_genomics = genomics_elo.iloc[0]["display_name"] if not genomics_elo.empty else "not available"
    failure_table = _table_html(
        failures,
        [("track", "Track"), ("method_id", "Method"), ("status", "Status"), ("dataset_count", "Datasets")],
    )
    clinical_table = _table_html(
        clinical_elo,
        [("display_name", "Method"), ("family", "Family"), ("elo_rating", "Uno C Elo")],
    )
    genomics_table = _table_html(
        genomics_elo,
        [("display_name", "Method"), ("family", "Family"), ("elo_rating", "Uno C Elo")],
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SurvArena no-HPO benchmark report</title>
  <style>
    :root {{ --ink:#1F2430; --muted:#6F768A; --line:#E6E8F0; --panel:#FFFFFF; --surface:#FCFCFD; }}
    body {{ margin:0; background:var(--surface); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ max-width:1080px; margin:0 auto; padding:44px 24px 72px; }}
    header, section {{ margin-bottom:38px; }}
    h1 {{ font-size:38px; letter-spacing:-0.03em; margin:0; }}
    h2 {{ font-size:24px; margin:0 0 12px; }}
    p, li {{ line-height:1.65; }}
    .summary {{ border-left:5px solid #5477C4; background:#FFFFFF; padding:20px 24px; border-radius:0 14px 14px 0; }}
    .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
    .metric {{ background:#FFFFFF; border:1px solid var(--line); border-radius:12px; padding:16px; }}
    .metric strong {{ display:block; font-size:25px; }}
    .metric span {{ color:var(--muted); font-size:13px; }}
    figure {{ margin:22px 0 30px; background:#FFFFFF; border:1px solid var(--line); border-radius:14px; padding:16px; }}
    figure img {{ width:100%; height:auto; display:block; }}
    figcaption {{ color:var(--muted); font-size:13px; margin-top:10px; line-height:1.45; }}
    .table-wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; background:#FFFFFF; }}
    table {{ border-collapse:collapse; width:100%; font-size:14px; }}
    th, td {{ text-align:left; padding:10px 12px; border-bottom:1px solid var(--line); }}
    th {{ color:#464C55; background:#F4F5F7; }}
    code {{ background:#F4F5F7; padding:2px 5px; border-radius:5px; }}
    .note {{ color:var(--muted); }}
    @media (max-width:760px) {{ .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} h1 {{ font-size:30px; }} }}
  </style>
</head>
<body>
<main data-report-audience="technical">
  <header data-contract-section="title">
    <h1>SurvArena no-HPO benchmark report</h1>
  </header>

  <section data-contract-section="technical-summary">
    <h2>Technical summary</h2>
    <div class="summary">
      <p><strong>The canonical no-HPO run has attempted {attempted_pairs}/{total_pairs} dataset-method cells, with {complete_pairs} cells reaching all 15 successful outer splits.</strong>
      Clinical coverage is {clinical_complete}/{len(clinical)} complete cells and genomics coverage is {genomics_complete}/{len(genomics)}.
      Failed and partially successful cells remain in the evidence bundle and are excluded only from comparisons that require complete paired metric coverage.</p>
      <p>Among eligibility-complete Uno C comparisons, the current leaders are <strong>{escape(str(top_clinical))}</strong> on clinical datasets and
      <strong>{escape(str(top_genomics))}</strong> on genomics datasets. Complete coverage across the full requested metric suite is
      available for {clinical_eligible_pairs}/189 clinical pairs and {genomics_eligible_pairs}/135 genomics pairs.</p>
    </div>
    <div class="metrics">
      <div class="metric"><strong>{attempted_pairs}/{total_pairs}</strong><span>Cells attempted</span></div>
      <div class="metric"><strong>{complete_pairs}</strong><span>Cells with 15/15 successful folds</span></div>
      <div class="metric"><strong>{counts['partial']}</strong><span>Partially successful cells</span></div>
      <div class="metric"><strong>{counts['failed']}</strong><span>Attempted cells with no successful folds</span></div>
    </div>
  </section>

  <section data-contract-section="key-findings">
    <h2>Clinical coverage is complete; genomics exposes practical failure modes</h2>
    <p>The clinical matrix is broadly comparable across model families. The genomics matrix is more selective: high-dimensional
    numerical failures and foundation runtime failures reduce the set of methods eligible for paired ranking. The coverage
    maps show exactly where those exclusions enter.</p>
    <figure>
      <img src="assets/clinical_coverage.png" alt="Clinical dataset by method coverage matrix">
      <figcaption>Clinical matrix status for 7 datasets × 27 methods; each complete cell contains 15 successful outer splits.</figcaption>
    </figure>
    <figure>
      <img src="assets/genomics_coverage.png" alt="Genomics dataset by method coverage matrix">
      <figcaption>Genomics matrix status for 5 TCGA cohorts × 27 methods. Failed cells are retained as operational evidence, not converted to missing values.</figcaption>
    </figure>
  </section>

  <section data-contract-section="model-ranking">
    <h2>Paired Uno C rankings use only complete eligible evidence</h2>
    <p>Elo ratings summarize paired fold-level wins after restricting each track to cells with complete metric coverage.
    They are comparative scores, not effect sizes; close ratings with overlapping uncertainty should not be treated as practically distinct.</p>
    <figure>
      <p class="note">Clinical eligibility-complete comparison: {clinical_eligible_pairs}/189 dataset-method pairs.</p>
      <img src="../clinical_no_hpo_current_default/elo/elo_manuscript_no_hpo_uno_c.png" alt="Clinical Uno C Elo rankings">
      <figcaption>Clinical Uno C Elo ratings from the current-default no-HPO artifacts.</figcaption>
    </figure>
    {clinical_table}
    <figure>
      <p class="note">Genomics eligibility-complete comparison: {genomics_eligible_pairs}/135 dataset-method pairs.</p>
      <img src="../genomics_no_hpo_current_default/elo/elo_manuscript_no_hpo_uno_c.png" alt="Genomics Uno C Elo rankings">
      <figcaption>Genomics Uno C Elo ratings from eligibility-complete current-default no-HPO artifacts.</figcaption>
    </figure>
    {genomics_table}
  </section>

  <section data-contract-section="scope-data-and-metric-definitions">
    <h2>Scope, data, and metric definitions</h2>
    <p>The benchmark covers 7 clinical datasets and 5 TCGA genomics cohorts. Each configured dataset-method cell uses repeated
    nested cross-validation with 5 outer folds × 3 repeats. No hyperparameter search is performed. Uno's concordance is the primary
    discrimination metric; Harrell's concordance, integrated Brier score, time-dependent AUC, horizon Brier scores, calibration
    errors, and net benefit are secondary outputs.</p>
    <p class="note">Canonical configs: <code>configs/benchmark/manuscript_v1.yaml</code> and
    <code>configs/benchmark/manuscript_genomics_v1.yaml</code>.</p>
  </section>

  <section data-contract-section="methodology">
    <h2>Methodology preserves pairing, failures, and the CPU reference backend</h2>
    <p>All methods share deterministic outer splits within a dataset. Resume logic accepts a split as complete only when its status is
    successful and the primary metric is present. Direct TabPFN and TabICL adapters are pinned to CPU in the manuscript configs to
    avoid Apple MPS runtime instability and to keep the operational comparison aligned with the documented reference machine.</p>
    <p>Rankings are generated from successful, eligibility-complete fold rows. Coverage and failure reporting use the full raw
    artifact matrix, including numerical convergence errors, missing metrics, runtime failures, and native-process failures.</p>
  </section>

  <section data-contract-section="limitations-uncertainty-and-robustness-checks">
    <h2>Limitations and robustness checks</h2>
    <ul>
      <li>No-HPO results measure default-policy robustness, not each method's best attainable performance.</li>
      <li>Genomics failures are informative for practitioner reliability but make the fully paired comparison set smaller.</li>
      <li>Elo depends on the included datasets and eligible methods; it should be read with rank summaries and per-dataset results.</li>
      <li>Runtime reflects the local Apple Silicon CPU reference environment and is not a hardware-independent property.</li>
    </ul>
    <h3>Observed incomplete or failed cells</h3>
    {failure_table}
  </section>

  <section data-contract-section="recommended-next-steps">
    <h2>Recommended next steps</h2>
    <ol>
      <li>Use the clinical matrix as the main all-family benchmark and present genomics as a high-dimensional robustness analysis.</li>
      <li>Report coverage and failure rates beside performance rankings so default-policy reliability remains visible.</li>
      <li>Freeze an archival environment export before final manuscript submission and rerun the strict publishability audit.</li>
    </ol>
  </section>

  <section data-contract-section="further-questions">
    <h2>Further questions</h2>
    <p>Would modest, predeclared stability defaults—such as penalization for parametric AFT models or stronger Coxnet regularization—
    materially improve genomics coverage without turning the no-HPO track into implicit tuning? That should be evaluated as a separate
    sensitivity protocol rather than retrofitted into this run.</p>
  </section>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the manuscript-grade SurvArena no-HPO technical report.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/manuscript_grade/no_hpo_benchmark_report"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    clinical = inventory_matrix(
        repo_root / "configs/benchmark/manuscript_v1.yaml",
        repo_root / "results/manuscript_grade/clinical_no_hpo_current_default/dataset_model",
        track="Clinical",
    )
    genomics = inventory_matrix(
        repo_root / "configs/benchmark/manuscript_genomics_v1.yaml",
        repo_root / "results/manuscript_grade/genomics_no_hpo_current_default/dataset_model",
        track="Genomics",
    )
    inventory = pd.concat([clinical, genomics], ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "coverage_status.csv", index=False)
    failures = failure_summary(inventory)
    failures.to_csv(output_dir / "failure_summary.csv", index=False)
    render_coverage_heatmap(
        clinical,
        output_dir / "assets/clinical_coverage.png",
        title="Clinical no-HPO coverage",
        subtitle="7 datasets × 27 methods; status is based on successful and attempted outer splits",
    )
    render_coverage_heatmap(
        genomics,
        output_dir / "assets/genomics_coverage.png",
        title="Genomics no-HPO coverage",
        subtitle="5 TCGA cohorts × 27 methods; failures remain visible in the benchmark evidence",
    )
    clinical_elo = _uno_elo_rows(
        repo_root / "results/manuscript_grade/clinical_no_hpo_current_default/elo/elo_ratings.csv"
    )
    genomics_elo = _uno_elo_rows(
        repo_root / "results/manuscript_grade/genomics_no_hpo_current_default/elo/elo_ratings.csv"
    )
    clinical_eligibility = pd.read_csv(
        repo_root / "results/manuscript_grade/clinical_no_hpo_current_default/elo/eligibility_summary.csv"
    )
    genomics_eligibility = pd.read_csv(
        repo_root / "results/manuscript_grade/genomics_no_hpo_current_default/elo/eligibility_summary.csv"
    )
    build_html(
        inventory=inventory,
        failures=failures,
        clinical_elo=clinical_elo,
        genomics_elo=genomics_elo,
        clinical_eligible_pairs=int(clinical_eligibility["eligible"].sum()),
        genomics_eligible_pairs=int(genomics_eligibility["eligible"].sum()),
        output_path=output_dir / "report.html",
    )
    print(f"wrote {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
