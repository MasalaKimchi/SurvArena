from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable


_PILOT_REPEATED_DEFAULT_FOLDS = 3
_PILOT_REPEATED_DEFAULT_REPEATS = 2
_PILOT_DEFAULT_INNER_FOLDS = 2


@dataclass(frozen=True)
class CliDependencies:
    survival_predictor_cls: type
    compare_survival_models: Callable[..., dict[str, Any]]
    foundation_runtime_catalog: Callable[[], Any]
    foundation_runtime_status_for_method: Callable[[str], Any]
    load_benchmark_config: Callable[[Path, str], dict[str, Any]]
    benchmark_plan: Callable[..., dict[str, Any]]
    benchmark_doctor: Callable[..., dict[str, Any]]
    benchmark_report: Callable[[Path], dict[str, Any]]
    run_benchmark: Callable[..., None]


def default_pilot_repeated_seeds(repeats: int) -> list[int]:
    return [11 + 12 * repeat for repeat in range(repeats)]


def hpo_config_from_args(args: Namespace) -> dict[str, object]:
    return {
        "enabled": bool(args.hpo_trials and args.hpo_trials > 0),
        "max_trials": int(args.hpo_trials or 0),
        "timeout_seconds": args.hpo_timeout_seconds,
        "sampler": "tpe",
        "pruner": "median",
        "n_startup_trials": 8,
    }


def run_cli_command(args: Namespace, *, deps: CliDependencies, repo_root: Path | None = None) -> None:
    if args.command == "fit":
        _run_fit(args, deps=deps)
        return
    if args.command in {"compare", "pilot"}:
        _run_compare_or_pilot(args, deps=deps)
        return
    if args.command == "foundation-check":
        _run_foundation_check(args, deps=deps)
        return
    if args.command == "benchmark":
        _run_benchmark_command(args, deps=deps, repo_root=repo_root)
        return
    raise ValueError(f"Unsupported command '{args.command}'.")


def _run_fit(args: Namespace, *, deps: CliDependencies) -> None:
    predictor = deps.survival_predictor_cls(
        label_time=args.time_col,
        label_event=args.event_col,
        eval_metric=args.eval_metric,
        presets=args.presets,
        included_models=args.models,
        excluded_models=args.exclude_models,
        retain_top_k_models=None if args.retain_all_models else args.retain_top_k_models,
        random_state=args.random_state,
        save_path=args.save_path,
        verbose=args.verbose,
        enable_foundation_models=args.enable_foundation_models,
    )
    hyperparameter_tune_kwargs = None
    if args.autogluon_num_trials is not None or args.tuning_timeout is not None:
        hyperparameter_tune_kwargs = {}
        if args.autogluon_num_trials is not None:
            hyperparameter_tune_kwargs["num_trials"] = args.autogluon_num_trials
        if args.tuning_timeout is not None:
            hyperparameter_tune_kwargs["timeout"] = args.tuning_timeout
    predictor.fit(
        args.train,
        tuning_data=args.tuning,
        test_data=args.test,
        dataset_name=args.dataset_name,
        holdout_frac=args.holdout_frac,
        time_limit=args.time_limit,
        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
        refit_full=args.refit_full,
        num_bag_folds=args.num_bag_folds,
        num_bag_sets=args.num_bag_sets,
    )
    _print_json(predictor.fit_summary())


def _run_compare_or_pilot(args: Namespace, *, deps: CliDependencies) -> None:
    split_strategy = args.split_strategy if args.command == "compare" else "fixed_split"
    outer_folds = args.outer_folds
    outer_repeats = args.outer_repeats
    seeds = args.seeds
    benchmark_id = None
    if args.command == "pilot":
        benchmark_id = "user_pilot_fixed"
        if args.repeated:
            split_strategy = "repeated_nested_cv"
            seeds = args.seeds or default_pilot_repeated_seeds(args.outer_repeats)
            benchmark_id = "user_pilot_cv"
        else:
            outer_repeats = 1
            seeds = args.seeds or [11]

    summary = deps.compare_survival_models(
        args.data,
        time_col=args.time_col,
        event_col=args.event_col,
        dataset_name=args.dataset_name,
        id_col=args.id_col,
        drop_columns=args.drop_columns,
        models=args.models,
        excluded_models=args.exclude_models,
        presets=args.presets,
        enable_foundation_models=args.enable_foundation_models,
        primary_metric=args.eval_metric,
        split_strategy=split_strategy,
        outer_folds=outer_folds,
        outer_repeats=outer_repeats,
        inner_folds=args.inner_folds,
        seeds=seeds,
        timeout_seconds=args.timeout_seconds,
        hpo=hpo_config_from_args(args),
        decision_curve_thresholds=args.decision_thresholds,
        output_dir=args.save_path,
        benchmark_id=benchmark_id,
        dry_run=args.dry_run,
    )
    _print_json(summary)


def _run_foundation_check(args: Namespace, *, deps: CliDependencies) -> None:
    if args.models is None:
        statuses = list(deps.foundation_runtime_catalog())
    else:
        statuses = [deps.foundation_runtime_status_for_method(method_id) for method_id in args.models]
    _print_json([asdict(status) for status in statuses])


def _run_benchmark_command(args: Namespace, *, deps: CliDependencies, repo_root: Path | None) -> None:
    resolved_repo_root = repo_root or Path(__file__).resolve().parents[2]
    if args.benchmark_command == "report":
        _print_json(deps.benchmark_report(Path(args.output_dir)))
        return

    benchmark_cfg = deps.load_benchmark_config(resolved_repo_root, args.benchmark_config)
    if args.benchmark_command == "plan":
        _print_json(deps.benchmark_plan(resolved_repo_root, benchmark_cfg, **_benchmark_plan_kwargs(args)))
        return
    if args.benchmark_command == "doctor":
        _print_json(
            deps.benchmark_doctor(
                resolved_repo_root,
                benchmark_cfg,
                **_benchmark_plan_kwargs(args),
                check_imports=args.check_imports,
                load_datasets=args.load_datasets,
            )
        )
        return
    if args.benchmark_command == "run":
        selected_cfg, dataset_override, method_override = _benchmark_run_config_and_overrides(benchmark_cfg, args)
        deps.run_benchmark(
            repo_root=resolved_repo_root,
            benchmark_cfg=selected_cfg,
            dataset_override=dataset_override,
            method_override=method_override,
            limit_seeds=args.limit_seeds,
            dry_run=args.dry_run,
            output_dir=_resolve_optional_repo_path(resolved_repo_root, args.output_dir),
            resume=bool(args.resume),
            max_retries=max(int(args.max_retries), 0),
            regenerate_splits=bool(args.regenerate_splits),
        )


def _benchmark_plan_kwargs(args: Namespace) -> dict[str, object]:
    return {
        "dataset_override": args.dataset,
        "method_override": args.method,
        "dataset_overrides": args.datasets,
        "method_overrides": args.methods,
        "limit_seeds": args.limit_seeds,
    }


def _benchmark_run_config_and_overrides(
    benchmark_cfg: dict[str, object],
    args: Namespace,
) -> tuple[dict[str, object], str | None, str | None]:
    selected_cfg = dict(benchmark_cfg)
    dataset_override = args.dataset
    method_override = args.method
    if args.datasets is not None:
        selected_cfg["datasets"] = args.datasets
        dataset_override = None
    if args.methods is not None:
        selected_cfg["methods"] = args.methods
        method_override = None
    return selected_cfg, dataset_override, method_override


def _resolve_optional_repo_path(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
