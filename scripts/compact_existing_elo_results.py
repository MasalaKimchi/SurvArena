from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd


TABLE_PATTERNS = {
    "elo": ("elo_ratings_", "elo_ratings.csv"),
    "pairwise": ("pairwise_win_rate_", "pairwise_win_rates.csv"),
    "ranks": ("rank_summary_", "rank_summary.csv"),
    "coverage": ("coverage_summary_", "coverage_summary.csv"),
    "summary": ("method_summary_", "method_summary.csv"),
}
FOLD_PREFIX = "manuscript_fold_results_success_"
FIGURE_PREFIX = "elo_manuscript_no_hpo_"
AGGREGATE_CSVS = {
    "elo_ratings.csv",
    "pairwise_win_rates.csv",
    "rank_summary.csv",
    "coverage_summary.csv",
    "method_summary.csv",
    "manuscript_fold_results_success.csv",
    "metric_suite_index.csv",
}
LOCAL_RUN_ROOTS = (
    "foundation_calibration_fix_smoke",
    "mitra_feasibility",
    "protocol_validation",
)


def _metric_from_name(path: Path, prefix: str) -> str:
    return path.stem.removeprefix(prefix)


def _read_metric_table(path: Path, *, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    metric = _metric_from_name(path, prefix)
    if "metric" not in frame.columns:
        frame["metric"] = metric
    else:
        frame["metric"] = frame["metric"].fillna(metric)
    return frame


def _write_aggregate_tables(elo_dir: Path) -> list[Path]:
    written: list[Path] = []
    for prefix, output_name in TABLE_PATTERNS.values():
        paths = sorted(path for path in elo_dir.glob(f"{prefix}*.csv") if path.name != output_name)
        if not paths:
            continue
        frame = pd.concat([_read_metric_table(path, prefix=prefix) for path in paths], ignore_index=True, sort=False)
        output = elo_dir / output_name
        frame.to_csv(output, index=False)
        written.append(output)
    return written


def _write_fold_results(elo_dir: Path) -> Path | None:
    paths = sorted(elo_dir.glob(f"{FOLD_PREFIX}*.csv"))
    if not paths:
        return None
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)
    frame = frame.drop(columns=["metric"], errors="ignore").drop_duplicates().reset_index(drop=True)
    output = elo_dir / "manuscript_fold_results_success.csv"
    frame.to_csv(output, index=False)
    return output


def _write_metric_suite_index(elo_dir: Path) -> Path | None:
    elo_path = elo_dir / "elo_ratings.csv"
    if not elo_path.exists():
        return None
    elo = pd.read_csv(elo_path, usecols=lambda column: column in {"metric"})
    if "metric" not in elo.columns:
        return None
    metrics = sorted(elo["metric"].dropna().astype(str).unique())
    if not metrics:
        return None
    rows = [
        {
            "metric": metric,
            "elo": str(elo_dir / "elo_ratings.csv"),
            "pairwise": str(elo_dir / "pairwise_win_rates.csv"),
            "ranks": str(elo_dir / "rank_summary.csv"),
            "coverage": str(elo_dir / "coverage_summary.csv"),
            "summary": str(elo_dir / "method_summary.csv"),
            "fold_results": str(elo_dir / "manuscript_fold_results_success.csv"),
            "figure_png": str(elo_dir / "figures" / f"{FIGURE_PREFIX}{metric}.png"),
        }
        for metric in metrics
    ]
    output = elo_dir / "metric_suite_index.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def _archive_metric_csvs(elo_dir: Path) -> int:
    legacy_dir = elo_dir / "legacy_metric_csvs"
    legacy_dir.mkdir(exist_ok=True)
    moved = 0
    prefixes = [FOLD_PREFIX, *(prefix for prefix, _ in TABLE_PATTERNS.values())]
    for prefix in prefixes:
        for path in sorted(elo_dir.glob(f"{prefix}*.csv")):
            if path.name in AGGREGATE_CSVS:
                continue
            target = legacy_dir / path.name
            if target.exists():
                target.unlink()
            shutil.move(str(path), target)
            moved += 1
    if moved == 0 and not any(legacy_dir.iterdir()):
        legacy_dir.rmdir()
    return moved


def _move_figures(elo_dir: Path) -> int:
    figures_dir = elo_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    moved = 0
    for path in sorted(elo_dir.glob(f"{FIGURE_PREFIX}*.png")):
        target = figures_dir / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), target)
        moved += 1
    return moved


def compact_elo_dir(elo_dir: Path) -> dict[str, int]:
    if not elo_dir.is_dir():
        raise ValueError(f"Not an Elo directory: {elo_dir}")
    written = _write_aggregate_tables(elo_dir)
    fold_results = _write_fold_results(elo_dir)
    index = _write_metric_suite_index(elo_dir)
    archived_csvs = _archive_metric_csvs(elo_dir)
    moved_figures = _move_figures(elo_dir)
    return {
        "aggregate_tables": len(written) + int(fold_results is not None) + int(index is not None),
        "archived_csvs": archived_csvs,
        "moved_figures": moved_figures,
    }


def _raw_dataset_model_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for compact_fold_results in root.glob("manuscript_grade/*/elo/manuscript_fold_results_success.csv"):
        dataset_model = compact_fold_results.parents[1] / "dataset_model"
        if dataset_model.exists():
            candidates.append(dataset_model)
    return sorted(set(candidates))


def _empty_dirs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_dir() and not any(path.iterdir())),
        key=lambda path: len(path.parts),
        reverse=True,
    )


def local_prune_candidates(
    root: Path,
    *,
    include_logs: bool = False,
    include_local_runs: bool = False,
    include_raw_dataset_model: bool = False,
) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(root.rglob(".DS_Store"))
    candidates.extend(root.rglob("*.pid"))
    if include_logs:
        candidates.extend(root.rglob("*.log"))
        candidates.append(root / "manuscript_grade" / "logs")
    if include_local_runs:
        candidates.extend(root / name for name in LOCAL_RUN_ROOTS)
    if include_raw_dataset_model:
        candidates.extend(_raw_dataset_model_candidates(root))
    candidates.extend(_empty_dirs(root))
    existing = [path for path in candidates if path.exists()]
    return sorted(set(existing), key=lambda path: (len(path.parts), str(path)))


def prune_local_artifacts(
    root: Path,
    *,
    apply: bool = False,
    include_logs: bool = False,
    include_local_runs: bool = False,
    include_raw_dataset_model: bool = False,
) -> int:
    candidates = local_prune_candidates(
        root,
        include_logs=include_logs,
        include_local_runs=include_local_runs,
        include_raw_dataset_model=include_raw_dataset_model,
    )
    action = "removed" if apply else "would remove"
    for path in candidates:
        if apply:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        print(f"{action} {path}")
    return len(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact existing manuscript Elo result folders.")
    parser.add_argument("elo_dirs", nargs="*", type=Path, help="Specific Elo directories to compact.")
    parser.add_argument("--root", type=Path, default=Path("results"), help="Root to search when no Elo dirs are given.")
    parser.add_argument(
        "--prune-local-artifacts",
        action="store_true",
        help="Preview removable local result cruft after compaction. Use --apply-prune to delete.",
    )
    parser.add_argument("--apply-prune", action="store_true", help="Delete paths reported by --prune-local-artifacts.")
    parser.add_argument("--include-logs", action="store_true", help="Include result logs in local-artifact pruning.")
    parser.add_argument(
        "--include-local-runs",
        action="store_true",
        help="Include known smoke/feasibility/protocol result roots that are not direct Elo inputs.",
    )
    parser.add_argument(
        "--include-raw-dataset-model",
        action="store_true",
        help="Include raw dataset_model roots when a sibling compact Elo fold-results bundle exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    elo_dirs = args.elo_dirs or sorted(path for path in args.root.rglob("elo") if path.is_dir())
    for elo_dir in elo_dirs:
        summary = compact_elo_dir(elo_dir)
        print(
            f"{elo_dir}: aggregate_tables={summary['aggregate_tables']} "
            f"archived_csvs={summary['archived_csvs']} moved_figures={summary['moved_figures']}"
        )
    if args.prune_local_artifacts:
        count = prune_local_artifacts(
            args.root,
            apply=bool(args.apply_prune),
            include_logs=bool(args.include_logs),
            include_local_runs=bool(args.include_local_runs),
            include_raw_dataset_model=bool(args.include_raw_dataset_model),
        )
        print(f"local_prune_candidates={count}")


if __name__ == "__main__":
    main()
