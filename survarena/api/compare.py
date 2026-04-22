from __future__ import annotations

from pathlib import Path
from typing import Any

from survarena.automl.presets import resolve_preset
from survarena.benchmark.runner import evaluate_split
from survarena.config import read_yaml
from survarena.data.splitters import load_or_create_splits
from survarena.data.user_dataset import load_user_dataset
from survarena.logging.export import (
    create_experiment_dir,
    export_fold_results,
    export_leaderboard,
    export_overall_summary,
    export_run_ledger,
    export_seed_summary,
)
from survarena.logging.tracker import payload_sha256, write_json
from survarena.methods.registry import registered_method_ids


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize_seed_list(seeds: list[int] | tuple[int, ...] | None) -> list[int]:
    if seeds is None:
        return [11]
    resolved = [int(seed) for seed in seeds]
    if not resolved:
        raise ValueError("seeds must not be empty.")
    return resolved


def _resolve_compare_methods(
    *,
    n_rows: int,
    n_features: int,
    event_count: int | None,
    event_fraction: float | None,
    high_cardinality_feature_count: int,
    has_datetime_features: bool,
    has_text_features: bool,
    models: list[str] | None,
    excluded_models: list[str] | None,
    presets: str,
    enable_foundation_models: bool,
) -> tuple[list[str], list[str], str | None]:
    if models is not None:
        method_ids = _dedupe_preserve_order([str(model) for model in models])
        if excluded_models is not None:
            excluded_set = set(excluded_models)
            method_ids = [method_id for method_id in method_ids if method_id not in excluded_set]
        if not method_ids:
            raise ValueError("No methods remain after applying explicit models and excluded_models filters.")
        return method_ids, [], None

    preset = resolve_preset(
        presets,
        n_rows=n_rows,
        n_features=n_features,
        event_count=event_count,
        event_fraction=event_fraction,
        high_cardinality_feature_count=high_cardinality_feature_count,
        has_datetime_features=has_datetime_features,
        has_text_features=has_text_features,
        excluded_models=excluded_models,
        enable_foundation_models=enable_foundation_models,
    )
    return list(preset.method_ids), list(preset.portfolio_notes), preset.name


def compare_survival_models(
    data: Any,
    *,
    time_col: str,
    event_col: str,
    dataset_name: str = "user_dataset",
    id_col: str | None = None,
    drop_columns: list[str] | None = None,
    models: list[str] | None = None,
    excluded_models: list[str] | None = None,
    presets: str = "fast",
    enable_foundation_models: bool = False,
    primary_metric: str = "harrell_c",
    split_strategy: str = "fixed_split",
    outer_folds: int = 5,
    outer_repeats: int = 1,
    inner_folds: int = 3,
    seeds: list[int] | tuple[int, ...] | None = None,
    n_trials: int = 0,
    timeout_seconds: float | None = None,
    autogluon: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    benchmark_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    dataset = load_user_dataset(
        data,
        time_col=time_col,
        event_col=event_col,
        dataset_id=dataset_name,
        dataset_name=dataset_name,
        id_col=id_col,
        drop_columns=drop_columns,
    )
    diagnostics = dataset.metadata.diagnostics
    method_ids, portfolio_notes, resolved_preset = _resolve_compare_methods(
        n_rows=len(dataset.X),
        n_features=dataset.X.shape[1],
        event_count=diagnostics.n_events if diagnostics is not None else None,
        event_fraction=diagnostics.event_rate if diagnostics is not None else None,
        high_cardinality_feature_count=len(diagnostics.high_cardinality_features) if diagnostics is not None else 0,
        has_datetime_features="datetime" in dataset.metadata.feature_types,
        has_text_features="text" in dataset.metadata.feature_types,
        models=models,
        excluded_models=excluded_models,
        presets=presets,
        enable_foundation_models=enable_foundation_models,
    )

    registered_methods = set(registered_method_ids())
    unknown_methods = sorted(set(method_ids) - registered_methods)
    if unknown_methods:
        raise ValueError(f"Unknown method ids: {unknown_methods}. Registered: {sorted(registered_methods)}")

    resolved_seeds = _normalize_seed_list(seeds)
    if split_strategy not in {"fixed_split", "repeated_nested_cv"}:
        raise ValueError(
            f"Unsupported split_strategy '{split_strategy}'. Expected one of ['fixed_split', 'repeated_nested_cv']."
        )
    if split_strategy == "fixed_split":
        if outer_repeats != 1:
            raise ValueError("fixed_split supports exactly one repeat. Use repeated_nested_cv for repeated evaluation.")
        if len(resolved_seeds) != 1:
            raise ValueError("fixed_split supports exactly one seed. Provide a single seed or use repeated_nested_cv.")
    elif outer_repeats > len(resolved_seeds):
        raise ValueError(
            f"Requested {outer_repeats} outer repeats but only {len(resolved_seeds)} seeds were provided."
        )

    autogluon_cfg = dict(autogluon or {})
    if timeout_seconds is not None:
        autogluon_cfg.setdefault("time_limit_seconds", float(timeout_seconds))
    if int(n_trials) > 0:
        hpo_kwargs = dict(autogluon_cfg.get("hyperparameter_tune_kwargs") or {})
        hpo_kwargs.setdefault("num_trials", int(n_trials))
        autogluon_cfg["hyperparameter_tune_kwargs"] = hpo_kwargs

    resolved_benchmark_id = benchmark_id or (
        "user_compare_fixed" if split_strategy == "fixed_split" else "user_compare_cv"
    )
    benchmark_cfg = {
        "benchmark_id": resolved_benchmark_id,
        "dataset_id": dataset.metadata.dataset_id,
        "dataset_name": dataset.metadata.name,
        "primary_metric": primary_metric,
        "split_strategy": split_strategy,
        "outer_folds": int(outer_folds),
        "outer_repeats": int(outer_repeats),
        "inner_folds": int(inner_folds),
        "seeds": list(resolved_seeds),
        "methods": list(method_ids),
        "n_trials": int(n_trials),
        "timeout_seconds": None if timeout_seconds is None else float(timeout_seconds),
        "autogluon": autogluon_cfg,
        "resolved_preset": resolved_preset,
        "portfolio_notes": list(portfolio_notes),
    }
    benchmark_cfg_hash = payload_sha256(benchmark_cfg)
    summary = {
        "benchmark_id": resolved_benchmark_id,
        "dataset_id": dataset.metadata.dataset_id,
        "dataset_name": dataset.metadata.name,
        "methods": list(method_ids),
        "primary_metric": primary_metric,
        "split_strategy": split_strategy,
        "outer_folds": int(outer_folds),
        "outer_repeats": int(outer_repeats),
        "inner_folds": int(inner_folds),
        "seeds": list(resolved_seeds),
        "n_trials": int(n_trials),
        "timeout_seconds": None if timeout_seconds is None else float(timeout_seconds),
        "autogluon": autogluon_cfg,
        "resolved_preset": resolved_preset,
        "portfolio_notes": list(portfolio_notes),
    }
    if diagnostics is not None:
        summary["dataset_diagnostics"] = diagnostics.to_dict()

    if dry_run:
        return summary

    task_id = f"{dataset.metadata.dataset_id}_{resolved_benchmark_id}"
    splits = load_or_create_splits(
        root=repo_root,
        task_id=task_id,
        split_strategy=split_strategy,
        n_samples=len(dataset.X),
        event=dataset.event,
        seeds=resolved_seeds,
        outer_folds=int(outer_folds),
        outer_repeats=int(outer_repeats),
    )

    resolved_output_dir = create_experiment_dir(repo_root) if output_dir is None else Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        resolved_output_dir / "experiment_manifest.json",
        {
            **summary,
            "benchmark_config_hash": benchmark_cfg_hash,
            "output_dir": str(resolved_output_dir),
        },
    )

    method_cfg_cache = {
        method_id: read_yaml(repo_root / "configs" / "methods" / f"{method_id}.yaml")
        for method_id in method_ids
    }
    all_records: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    horizons_quantiles = (0.25, 0.5, 0.75)

    for method_id in method_ids:
        method_cfg = method_cfg_cache[method_id]
        for split in splits:
            record = evaluate_split(
                benchmark_id=resolved_benchmark_id,
                dataset_id=dataset.metadata.dataset_id,
                method_id=method_id,
                split=split,
                X=dataset.X,
                time=dataset.time,
                event=dataset.event,
                method_cfg=method_cfg,
                inner_folds=int(inner_folds),
                n_trials=int(n_trials),
                timeout_seconds=None if timeout_seconds is None else float(timeout_seconds),
                primary_metric=primary_metric,
                horizons_quantiles=horizons_quantiles,
                benchmark_cfg_hash=benchmark_cfg_hash,
                autogluon_cfg=autogluon_cfg,
            )
            run_records.append(record.pop("run_payload"))
            all_records.append(record)
            print(
                f"[{record['status']}] {dataset.metadata.dataset_id}/{method_id}/{split.split_id}/seed{split.seed} "
                f"{primary_metric}={record.get(primary_metric)}"
            )

    frame = export_fold_results(repo_root, all_records, output_dir=resolved_output_dir, file_prefix=resolved_benchmark_id)
    seed_summary = export_seed_summary(
        repo_root,
        frame,
        output_dir=resolved_output_dir,
        file_prefix=resolved_benchmark_id,
    )
    export_overall_summary(repo_root, frame, output_dir=resolved_output_dir, file_prefix=resolved_benchmark_id)
    export_leaderboard(
        repo_root,
        seed_summary,
        primary_metric=primary_metric,
        output_dir=resolved_output_dir,
        file_prefix=resolved_benchmark_id,
    )
    export_run_ledger(repo_root, run_records, benchmark_id=resolved_benchmark_id, output_dir=resolved_output_dir)
    return {
        **summary,
        "output_dir": str(resolved_output_dir),
        "split_count": int(len(splits)),
    }
