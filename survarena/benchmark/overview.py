from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from survarena.benchmark.governance import resolve_comparison_modes
from survarena.benchmark.runner import validate_benchmark_profile_contract
from survarena.config import read_yaml
from survarena.methods.foundation import foundation_runtime_status_for_method
from survarena.methods.registry import registered_method_ids


FOUNDATION_METHOD_PREFIXES = ("tabpfn_", "mitra_")


def _resolve_repo_path(repo_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def load_benchmark_config(repo_root: Path, config_path: str | Path) -> dict[str, Any]:
    return read_yaml(_resolve_repo_path(repo_root, config_path))


def _planned_split_count(cfg: dict[str, Any], seeds: list[int]) -> int:
    split_strategy = str(cfg.get("split_strategy", "fixed_split"))
    if split_strategy == "repeated_nested_cv":
        requested_repeats = int(cfg.get("outer_repeats", 1))
        outer_repeats = min(requested_repeats, len(seeds))
        return int(cfg.get("outer_folds", 5)) * outer_repeats
    return len(seeds)


def _planned_robustness_track_count(cfg: dict[str, Any]) -> int:
    robustness = cfg.get("robustness", {})
    if not isinstance(robustness, dict) or not bool(robustness.get("enabled", False)):
        return 1
    tracks = [str(track) for track in robustness.get("tracks", ["missingness", "covariate_noise", "label_noise"])]
    severity_count = len(list(robustness.get("severity_levels", [0.05, 0.15])))
    non_baseline = [track for track in tracks if track != "baseline"]
    return 1 + len(non_baseline) * severity_count


def benchmark_plan(
    repo_root: Path,
    benchmark_cfg: dict[str, Any],
    *,
    dataset_override: str | None = None,
    method_override: str | None = None,
    dataset_overrides: list[str] | None = None,
    method_overrides: list[str] | None = None,
    limit_seeds: int | None = None,
) -> dict[str, Any]:
    seeds = list(benchmark_cfg.get("seeds", []))
    if limit_seeds is not None:
        seeds = seeds[:limit_seeds]
    datasets = dataset_overrides or ([dataset_override] if dataset_override else list(benchmark_cfg.get("datasets", [])))
    methods = method_overrides or ([method_override] if method_override else list(benchmark_cfg.get("methods", [])))
    comparison_modes = resolve_comparison_modes(benchmark_cfg)
    split_count = _planned_split_count(benchmark_cfg, seeds)
    robustness_track_count = _planned_robustness_track_count(benchmark_cfg)
    run_units = len(datasets) * len(methods) * split_count * len(comparison_modes) * robustness_track_count
    hpo_cfg = dict(benchmark_cfg.get("hpo", {}))
    hpo_trials = int(hpo_cfg.get("max_trials", 0) or 0) if bool(hpo_cfg.get("enabled", False)) else 0
    hpo_modes = 1 if "hpo" in comparison_modes else 0
    estimated_inner_fits = (
        len(datasets)
        * len(methods)
        * split_count
        * robustness_track_count
        * hpo_modes
        * hpo_trials
        * int(benchmark_cfg.get("inner_folds", 1))
    )
    return {
        "benchmark_id": benchmark_cfg.get("benchmark_id"),
        "profile": benchmark_cfg.get("profile"),
        "primary_metric": benchmark_cfg.get("primary_metric", "harrell_c"),
        "split_strategy": benchmark_cfg.get("split_strategy"),
        "datasets": datasets,
        "methods": methods,
        "seeds": seeds,
        "comparison_modes": list(comparison_modes),
        "outer_folds": int(benchmark_cfg.get("outer_folds", 1)),
        "outer_repeats": int(benchmark_cfg.get("outer_repeats", 1)),
        "planned_splits_per_dataset": split_count,
        "robustness_track_count": robustness_track_count,
        "planned_run_units": run_units,
        "estimated_hpo_inner_fits": estimated_inner_fits,
        "output_layout": "one dataset folder per dataset, with fold results, leaderboard, diagnostics, manifest",
        "repo_root": str(repo_root),
    }


def benchmark_doctor(
    repo_root: Path,
    benchmark_cfg: dict[str, Any],
    *,
    dataset_override: str | None = None,
    method_override: str | None = None,
    dataset_overrides: list[str] | None = None,
    method_overrides: list[str] | None = None,
    limit_seeds: int | None = None,
) -> dict[str, Any]:
    plan = benchmark_plan(
        repo_root,
        benchmark_cfg,
        dataset_override=dataset_override,
        method_override=method_override,
        dataset_overrides=dataset_overrides,
        method_overrides=method_overrides,
        limit_seeds=limit_seeds,
    )
    issues: list[dict[str, str]] = []
    _append_profile_issues(benchmark_cfg, issues)
    _append_method_issues(repo_root, plan["methods"], issues)
    _append_dataset_issues(repo_root, plan["datasets"], issues)
    _append_plan_issues(plan, issues)

    return {
        "status": "ok" if not any(issue["severity"] == "error" for issue in issues) else "error",
        "plan": plan,
        "issues": issues,
    }


def _append_profile_issues(benchmark_cfg: dict[str, Any], issues: list[dict[str, str]]) -> None:
    try:
        validate_benchmark_profile_contract(benchmark_cfg)
    except ValueError as exc:
        issues.append({"severity": "error", "check": "profile_contract", "message": str(exc)})


def _append_method_issues(repo_root: Path, methods: list[str], issues: list[dict[str, str]]) -> None:
    registered = set(registered_method_ids())
    for method_id in methods:
        method_path = repo_root / "configs" / "methods" / f"{method_id}.yaml"
        if method_id not in registered:
            issues.append({"severity": "error", "check": "method_registry", "message": f"Unknown method_id: {method_id}"})
        if not method_path.exists():
            issues.append(
                {"severity": "error", "check": "method_config", "message": f"Missing method config: {method_path}"}
            )
        if str(method_id).startswith(FOUNDATION_METHOD_PREFIXES):
            status = foundation_runtime_status_for_method(str(method_id))
            if not status.dependency_installed:
                issues.append(
                    {
                        "severity": "warning",
                        "check": "foundation_dependency",
                        "message": f"{method_id} dependency is not installed; use {status.install_command}",
                    }
                )
            if status.warning_reason:
                issues.append({"severity": "warning", "check": "foundation_runtime", "message": status.warning_reason})


def _append_dataset_issues(repo_root: Path, datasets: list[str], issues: list[dict[str, str]]) -> None:
    for dataset_id in datasets:
        dataset_path = repo_root / "configs" / "datasets" / f"{dataset_id}.yaml"
        if not dataset_path.exists():
            issues.append(
                {"severity": "error", "check": "dataset_config", "message": f"Missing dataset config: {dataset_path}"}
            )


def _append_plan_issues(plan: dict[str, Any], issues: list[dict[str, str]]) -> None:
    if not plan["seeds"]:
        issues.append({"severity": "error", "check": "seeds", "message": "Benchmark seed list is empty."})
    if plan["planned_run_units"] == 0:
        issues.append({"severity": "error", "check": "plan", "message": "Benchmark has no planned run units."})


def benchmark_report(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    fold_paths = sorted(output_dir.rglob("*_fold_results.csv"))
    if not fold_paths:
        raise FileNotFoundError(f"No *_fold_results.csv files found under {output_dir}.")

    frames = [pd.read_csv(path).assign(artifact_path=str(path)) for path in fold_paths]
    fold_results = pd.concat(frames, ignore_index=True, sort=False)
    status_counts = fold_results.get("status", pd.Series(dtype=object)).fillna("unknown").astype(str).value_counts()
    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "fold_result_files": [str(path) for path in fold_paths],
        "n_rows": int(len(fold_results)),
        "datasets": sorted(fold_results.get("dataset_id", pd.Series(dtype=object)).dropna().astype(str).unique()),
        "methods": sorted(fold_results.get("method_id", pd.Series(dtype=object)).dropna().astype(str).unique()),
        "hpo_modes": sorted(fold_results.get("hpo_mode", pd.Series(dtype=object)).dropna().astype(str).unique()),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
    }
    metric_candidates = [col for col in ("uno_c", "harrell_c", "ibs", "runtime_sec") if col in fold_results.columns]
    if metric_candidates and {"method_id"}.issubset(fold_results.columns):
        grouped = fold_results.groupby("method_id", as_index=False)[metric_candidates].mean(numeric_only=True)
        summary["method_means"] = grouped.to_dict(orient="records")
    return summary
