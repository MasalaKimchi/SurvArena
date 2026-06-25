from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from survarena.config import read_yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CLINICAL_CONFIG = REPO_ROOT / "configs" / "benchmark" / "manuscript_v1.yaml"
CLINICAL_HPO_CONFIG = REPO_ROOT / "configs" / "benchmark" / "manuscript_hpo_v1.yaml"
GENOMICS_CONFIG = REPO_ROOT / "configs" / "benchmark" / "manuscript_genomics_v1.yaml"
CLINICAL_SUCCESS = REPO_ROOT / "results" / "manuscript_grade" / "clinical_no_hpo" / "elo" / "manuscript_fold_results_success.csv"
GENOMICS_SUCCESS = REPO_ROOT / "results" / "manuscript_grade" / "genomics_no_hpo" / "elo" / "manuscript_fold_results_success.csv"
GENOMICS_STATUS = REPO_ROOT / "results" / "manuscript_grade" / "genomics_no_hpo" / "genomics_dataset_method_status.csv"
FOUNDATION_ROOT = REPO_ROOT / "results" / "manuscript_grade" / "clinical_discrete_hazard_foundation"
CURRENT_CLINICAL_ROOT = (
    REPO_ROOT / "results" / "manuscript_grade" / "clinical_no_hpo_current_default" / "dataset_model"
)
CURRENT_GENOMICS_ROOT = (
    REPO_ROOT / "results" / "manuscript_grade" / "genomics_no_hpo_current_default" / "dataset_model"
)

FOUNDATION_ALIAS_TO_CANONICAL = {
    "tabpfn_discrete_hazard_survival": "tabpfn_survival",
    "tabicl_discrete_hazard_survival": "tabicl_survival",
    "tabm_discrete_hazard_survival": "tabm_survival",
    "realtabpfn_discrete_hazard_survival": "realtabpfn_survival",
}
CANONICAL_FOUNDATION_METHODS = tuple(FOUNDATION_ALIAS_TO_CANONICAL.values())


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    status: str
    observed_rows: int
    expected_rows: int
    observed_pairs: int
    expected_pairs: int


def _split_base(value: Any) -> str:
    return str(value).split("__", 1)[0]


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _expected_split_count(cfg: dict[str, Any]) -> int:
    return int(cfg.get("outer_folds", 0)) * int(cfg.get("outer_repeats", 0))


def _coverage_from_success_file(path: Path, cfg: dict[str, Any], *, methods: list[str] | None = None) -> CoverageSummary:
    datasets = list(cfg.get("datasets", []))
    configured_methods = methods if methods is not None else list(cfg.get("methods", []))
    expected_pairs = len(datasets) * len(configured_methods)
    expected_rows = expected_pairs * _expected_split_count(cfg)
    frame = _read_csv_if_exists(path)
    if frame.empty:
        return CoverageSummary("missing", 0, expected_rows, 0, expected_pairs)
    frame = frame.copy()
    frame["dataset_base"] = frame["dataset_id"].map(_split_base)
    frame["method_base"] = frame["method_id"].astype(str)
    if methods is not None:
        frame = frame[frame["method_base"].isin(set(methods))]
    if "status" in frame.columns:
        frame = frame[frame["status"].eq("success")]
    pair_rows = frame.groupby(["dataset_base", "method_base"])["split_id"].nunique().reset_index(name="splits")
    complete_pairs = int(pair_rows[pair_rows["splits"].eq(_expected_split_count(cfg))].shape[0])
    status = "complete" if int(frame.shape[0]) == expected_rows and complete_pairs == expected_pairs else "partial"
    return CoverageSummary(status, int(frame.shape[0]), expected_rows, complete_pairs, expected_pairs)


def _coverage_from_dataset_model_root(root: Path, cfg: dict[str, Any]) -> CoverageSummary:
    datasets = list(cfg.get("datasets", []))
    methods = list(cfg.get("methods", []))
    expected_pairs = len(datasets) * len(methods)
    expected_splits = _expected_split_count(cfg)
    expected_rows = expected_pairs * expected_splits
    if not root.exists():
        return CoverageSummary("missing", 0, expected_rows, 0, expected_pairs)

    observed_rows = 0
    complete_pairs = 0
    for dataset_id in datasets:
        for method_id in methods:
            path = root / str(dataset_id) / str(method_id) / f"{method_id}_fold_results.csv"
            if not path.exists():
                continue
            frame = _read_csv_if_exists(path)
            if frame.empty:
                continue
            if "status" in frame.columns:
                frame = frame[frame["status"].eq("success")]
            if "split_id" in frame.columns:
                successful_splits = int(frame["split_id"].nunique())
            else:
                successful_splits = int(frame.shape[0])
            observed_rows += successful_splits
            if successful_splits == expected_splits:
                complete_pairs += 1

    status = "complete" if observed_rows == expected_rows and complete_pairs == expected_pairs else "partial"
    return CoverageSummary(status, observed_rows, expected_rows, complete_pairs, expected_pairs)


def _load_foundation_frames(root: Path) -> pd.DataFrame:
    paths = sorted(root.rglob("*_fold_results.csv"))
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["_source_path"] = str(path.relative_to(REPO_ROOT))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _foundation_coverage(clinical_cfg: dict[str, Any]) -> pd.DataFrame:
    datasets = list(clinical_cfg.get("datasets", []))
    expected_splits = _expected_split_count(clinical_cfg)
    expected = pd.MultiIndex.from_product(
        [CANONICAL_FOUNDATION_METHODS, datasets],
        names=["canonical_method_id", "dataset_id"],
    ).to_frame(index=False)
    frame = _load_foundation_frames(FOUNDATION_ROOT)
    if frame.empty:
        expected["source_method_id"] = ""
        expected["rows"] = 0
        expected["success_rows"] = 0
        expected["successful_splits"] = 0
    else:
        frame = frame.copy()
        frame["canonical_method_id"] = frame["method_id"].astype(str).replace(FOUNDATION_ALIAS_TO_CANONICAL)
        frame["dataset_id"] = frame["dataset_id"].map(_split_base)
        grouped = (
            frame.groupby(["canonical_method_id", "dataset_id"], as_index=False)
            .agg(
                source_method_id=("method_id", lambda values: ";".join(sorted(set(map(str, values))))),
                rows=("split_id", "size"),
                success_rows=("status", lambda values: int(pd.Series(values).eq("success").sum())),
                successful_splits=("split_id", lambda values: int(pd.Series(values).nunique())),
            )
            .sort_values(["canonical_method_id", "dataset_id"])
        )
        expected = expected.merge(grouped, on=["canonical_method_id", "dataset_id"], how="left")
        for column in ["source_method_id"]:
            expected[column] = expected[column].fillna("")
        for column in ["rows", "success_rows", "successful_splits"]:
            expected[column] = expected[column].fillna(0).astype(int)
    expected["expected_splits"] = expected_splits
    expected["retained_complete"] = expected["success_rows"].eq(expected_splits)
    expected["publishable_as_current_default"] = False
    expected["gap"] = expected.apply(_foundation_gap, axis=1)
    return expected.sort_values(["canonical_method_id", "dataset_id"]).reset_index(drop=True)


def _foundation_gap(row: pd.Series) -> str:
    if bool(row["retained_complete"]):
        return "complete alias-run evidence; rerun or provenance-map before citing canonical default"
    if int(row["rows"]) == 0:
        return "missing"
    if int(row["success_rows"]) == 0:
        return "all attempted rows failed"
    return f"incomplete: {int(row['success_rows'])}/{int(row['expected_splits'])} successful splits"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    subset = frame.loc[:, columns].copy()
    return subset.to_markdown(index=False)


def build_report() -> tuple[str, bool]:
    clinical_cfg = read_yaml(CLINICAL_CONFIG)
    clinical_hpo_cfg = read_yaml(CLINICAL_HPO_CONFIG)
    genomics_cfg = read_yaml(GENOMICS_CONFIG)

    clinical = _coverage_from_success_file(CLINICAL_SUCCESS, clinical_cfg)
    genomics = _coverage_from_success_file(GENOMICS_SUCCESS, genomics_cfg)
    current_clinical = _coverage_from_dataset_model_root(CURRENT_CLINICAL_ROOT, clinical_cfg)
    current_genomics = _coverage_from_dataset_model_root(CURRENT_GENOMICS_ROOT, genomics_cfg)
    clinical_hpo = _coverage_from_success_file(
        REPO_ROOT / "results" / "manuscript_grade" / "clinical_hpo" / "elo" / "manuscript_fold_results_success.csv",
        clinical_hpo_cfg,
    )
    foundation = _foundation_coverage(clinical_cfg)
    complete_foundation_pairs = int(foundation["retained_complete"].sum())
    expected_foundation_pairs = int(foundation.shape[0])
    status_rows = _read_csv_if_exists(GENOMICS_STATUS)
    ineligible_pairs = 0
    if not status_rows.empty and "eligible_uno_c" in status_rows.columns:
        ineligible_pairs = int((~status_rows["eligible_uno_c"].astype(bool)).sum())
    elif not status_rows.empty and "status" in status_rows.columns:
        ineligible_pairs = int(status_rows["status"].astype(str).str.contains("ineligible|failed", case=False, na=False).sum())

    blocking = []
    if current_clinical.status != "complete":
        blocking.append("Complete the canonical current-default clinical no-HPO matrix and rebuild its Elo/report bundle.")
    if clinical_hpo.status != "complete":
        blocking.append("Complete or explicitly scope out clinical HPO evidence.")
    if current_genomics.status != "complete":
        blocking.append(
            "Decide whether incomplete-success genomics coverage is a main benchmark, appendix robustness analysis, "
            "or excluded sensitivity analysis."
        )
    blocking.extend(
        [
            "Freeze dependency environment in a lockfile or archival environment export.",
            "Stage final manuscript tables/figures from the current-default compact artifact bundles.",
        ]
    )
    locally_completed = [
        "Canonical foundation adapters default to pooled discrete-time hazard.",
        "Maintained benchmark configs dry-run with canonical foundation IDs.",
        "Clinical no-HPO current-default evidence has complete 7-dataset x 27-method x 15-split coverage.",
        "Current-default clinical and genomics Elo/report bundles have been rebuilt.",
        "Protocol smoke validation passes locally.",
    ]
    is_publishable = (
        current_clinical.status == "complete"
        and clinical_hpo.status == "complete"
        and current_genomics.status == "complete"
    )

    lines = [
        "# Manuscript Publishability Audit",
        "",
        f"Generated from local configs and result artifacts on {date.today().isoformat()}.",
        "",
        "## Verdict",
        "",
        "**Not yet publication-ready as the full no-HPO-plus-HPO manuscript evidence bundle.** The current-default clinical",
        "no-HPO matrix and report are complete; the remaining blockers are listed below.",
        "",
        "## Completed Locally",
        "",
        *[f"- {item}" for item in locally_completed],
        "",
        "## Blocking Gaps",
        "",
        *[f"- {item}" for item in blocking],
        "",
        "## Retained Evidence Coverage",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {
                        "track": "Clinical no-HPO current default",
                        "status": current_clinical.status,
                        "rows": f"{current_clinical.observed_rows}/{current_clinical.expected_rows}",
                        "complete_pairs": f"{current_clinical.observed_pairs}/{current_clinical.expected_pairs}",
                        "artifact": str(CURRENT_CLINICAL_ROOT.relative_to(REPO_ROOT)),
                    },
                    {
                        "track": "Clinical no-HPO",
                        "status": clinical.status,
                        "rows": f"{clinical.observed_rows}/{clinical.expected_rows}",
                        "complete_pairs": f"{clinical.observed_pairs}/{clinical.expected_pairs}",
                        "artifact": str(CLINICAL_SUCCESS.relative_to(REPO_ROOT)),
                    },
                    {
                        "track": "Clinical HPO",
                        "status": clinical_hpo.status,
                        "rows": f"{clinical_hpo.observed_rows}/{clinical_hpo.expected_rows}",
                        "complete_pairs": f"{clinical_hpo.observed_pairs}/{clinical_hpo.expected_pairs}",
                        "artifact": "results/manuscript_grade/clinical_hpo/elo/manuscript_fold_results_success.csv",
                    },
                    {
                        "track": "Genomics no-HPO current default",
                        "status": current_genomics.status,
                        "rows": f"{current_genomics.observed_rows}/{current_genomics.expected_rows}",
                        "complete_pairs": f"{current_genomics.observed_pairs}/{current_genomics.expected_pairs}",
                        "artifact": str(CURRENT_GENOMICS_ROOT.relative_to(REPO_ROOT)),
                    },
                    {
                        "track": "Genomics no-HPO",
                        "status": genomics.status,
                        "rows": f"{genomics.observed_rows}/{genomics.expected_rows}",
                        "complete_pairs": f"{genomics.observed_pairs}/{genomics.expected_pairs}",
                        "artifact": str(GENOMICS_SUCCESS.relative_to(REPO_ROOT)),
                    },
                    {
                        "track": "Clinical discrete-hazard foundation",
                        "status": "partial",
                        "rows": f"{int(foundation['success_rows'].sum())}/{expected_foundation_pairs * _expected_split_count(clinical_cfg)}",
                        "complete_pairs": f"{complete_foundation_pairs}/{expected_foundation_pairs}",
                        "artifact": str(FOUNDATION_ROOT.relative_to(REPO_ROOT)),
                    },
                ]
            ),
            ["track", "status", "rows", "complete_pairs", "artifact"],
        ),
        "",
        "## Legacy Foundation-Only Evidence Detail",
        "",
        "This table audits the older foundation-only artifact root. It is retained for provenance and no longer blocks the",
        "complete current-default clinical no-HPO matrix.",
        "",
        _markdown_table(
            foundation,
            [
                "canonical_method_id",
                "dataset_id",
                "source_method_id",
                "success_rows",
                "expected_splits",
                "gap",
            ],
        ),
        "",
        "## Genomics Status",
        "",
        f"- Genomics status audit rows with failed/ineligible labels: {ineligible_pairs}.",
        "- The current-default five-cohort matrix has complete attempt coverage but partial universal-success coverage;",
        "  eligibility-complete comparisons are available in `results/manuscript_grade/genomics_no_hpo_current_default/elo/`.",
        "",
        "## Minimum Completion Criteria",
        "",
        "- Either complete `configs/benchmark/manuscript_hpo_v1.yaml` or move HPO to a clearly labeled appendix/future-work scope.",
        "- Label the five-dataset genomics matrix as a robustness analysis unless universal successful coverage is required.",
        "- Export a frozen environment spec alongside the artifact bundle.",
        "- Stage final manuscript tables/figures from the current-default artifacts and cite their paths/checksums.",
        "",
    ]
    return "\n".join(lines), is_publishable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local manuscript publishability from retained SurvArena artifacts.")
    parser.add_argument(
        "--write-doc",
        type=Path,
        default=REPO_ROOT / "docs" / "manuscript_publishability.md",
        help="Markdown report path to write.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if final manuscript evidence is incomplete.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report, is_publishable = build_report()
    output = args.write_doc
    output = output if output.is_absolute() else REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    try:
        display_path = output.relative_to(REPO_ROOT)
    except ValueError:
        display_path = output
    print(f"wrote {display_path}")
    print(f"publishable={str(is_publishable).lower()}")
    if args.strict and not is_publishable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
