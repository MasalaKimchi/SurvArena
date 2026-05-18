from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from survarena.api import SurvivalPredictor, compare_survival_models
from survarena.benchmark.overview import benchmark_doctor, benchmark_plan, benchmark_report, load_benchmark_config
from survarena.benchmark.runner import run_benchmark
from survarena.methods.foundation import foundation_runtime_catalog, foundation_runtime_status_for_method

_PRESET_CHOICES = ("fast", "medium", "best", "all", "foundation")
_METRIC_CHOICES = ("harrell_c", "uno_c")
_PILOT_REPEATED_DEFAULT_FOLDS = 3
_PILOT_REPEATED_DEFAULT_REPEATS = 2
_PILOT_DEFAULT_INNER_FOLDS = 2


def _parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list with at least one value.")
    return items


def _parse_int_csv_list(value: str) -> list[int]:
    items = _parse_csv_list(value)
    try:
        return [int(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers.") from exc


def _parse_float_csv_list(value: str) -> list[float]:
    items = _parse_csv_list(value)
    try:
        return [float(item) for item in items]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of floats.") from exc


def _default_pilot_repeated_seeds(repeats: int) -> list[int]:
    return [11 + 12 * repeat for repeat in range(repeats)]


def _hpo_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "enabled": bool(args.hpo_trials and args.hpo_trials > 0),
        "max_trials": int(args.hpo_trials or 0),
        "timeout_seconds": args.hpo_timeout_seconds,
        "sampler": "tpe",
        "pruner": "median",
        "n_startup_trials": 8,
    }


def _add_foundation_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--foundation",
        dest="enable_foundation_models",
        action="store_true",
        help="Include supported tabular foundation-model adapters.",
    )


def _add_benchmark_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        "--benchmark-config",
        dest="benchmark_config",
        default="configs/benchmark/standard_v1.yaml",
        help="Path to benchmark YAML config.",
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset", default=None, help="Optional single dataset override.")
    dataset_group.add_argument("--datasets", type=_parse_csv_list, default=None, help="Optional comma-separated dataset ids.")
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--method", default=None, help="Optional single method override.")
    method_group.add_argument("--methods", type=_parse_csv_list, default=None, help="Optional comma-separated method ids.")
    parser.add_argument("--limit-seeds", type=int, default=None, help="Use first N seeds only.")


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _benchmark_plan_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dataset_override": args.dataset,
        "method_override": args.method,
        "dataset_overrides": args.datasets,
        "method_overrides": args.methods,
        "limit_seeds": args.limit_seeds,
    }


def _benchmark_run_config_and_overrides(
    benchmark_cfg: dict[str, object],
    args: argparse.Namespace,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SurvArena command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a survival predictor on user-provided data.")
    fit_parser.add_argument("--train", required=True, help="Path to training CSV or Parquet data.")
    fit_parser.add_argument("--tuning", default=None, help="Optional tuning/validation CSV or Parquet data.")
    fit_parser.add_argument("--test", default=None, help="Optional test CSV or Parquet data.")
    fit_parser.add_argument("--time-col", required=True, help="Name of the duration column.")
    fit_parser.add_argument("--event-col", required=True, help="Name of the event indicator column.")
    fit_parser.add_argument("--presets", default="all", choices=_PRESET_CHOICES)
    fit_parser.add_argument("--models", type=_parse_csv_list, default=None, help="Optional comma-separated model ids.")
    fit_parser.add_argument(
        "--exclude-models",
        type=_parse_csv_list,
        default=None,
        help="Optional comma-separated model ids to remove from the chosen preset.",
    )
    fit_parser.add_argument("--eval-metric", default="uno_c", choices=_METRIC_CHOICES)
    fit_parser.add_argument("--autogluon-num-trials", type=int, default=None)
    fit_parser.add_argument(
        "--tuning-timeout",
        type=float,
        default=None,
        help="Optional per-model HPO timeout in seconds.",
    )
    fit_parser.add_argument("--random-state", type=int, default=0)
    fit_parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Approximate wallclock budget in seconds for the full fit run.",
    )
    fit_parser.add_argument(
        "--holdout-frac",
        type=float,
        default=None,
        help="Override the preset holdout fraction when --tuning is not provided.",
    )
    fit_parser.add_argument(
        "--num-bag-folds",
        type=int,
        default=0,
        help="Enable bagged OOF selection with the given number of folds. Use 0 to disable bagging.",
    )
    fit_parser.add_argument(
        "--num-bag-sets",
        type=int,
        default=1,
        help="Repeat the bagging fold schedule this many times when --num-bag-folds is enabled.",
    )
    retention_group = fit_parser.add_mutually_exclusive_group()
    retention_group.add_argument(
        "--retain-top-k-models",
        type=int,
        default=1,
        help="Retain only the top K ranked fitted models for inference artifacts.",
    )
    retention_group.add_argument(
        "--retain-all-models",
        action="store_true",
        help="Retain every successful fitted model instead of only the top-ranked models.",
    )
    fit_parser.add_argument("--save-path", default=None, help="Optional artifact directory root.")
    fit_parser.add_argument("--dataset-name", default="user_dataset")
    fit_parser.add_argument(
        "--refit-full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to refit retained models on all available non-test data after selection.",
    )
    fit_parser.add_argument("--verbose", action="store_true", help="Show underlying tuning logs.")
    _add_foundation_flag(fit_parser)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Run a benchmark-style comparison on a user-provided dataset with explicit model control.",
    )
    compare_parser.add_argument("--data", required=True, help="Path to CSV or Parquet data.")
    compare_parser.add_argument("--time-col", required=True, help="Name of the duration column.")
    compare_parser.add_argument("--event-col", required=True, help="Name of the event indicator column.")
    compare_parser.add_argument("--dataset-name", default="user_dataset")
    compare_parser.add_argument("--id-col", default=None, help="Optional identifier column to drop before fitting.")
    compare_parser.add_argument(
        "--drop-columns",
        type=_parse_csv_list,
        default=None,
        help="Optional comma-separated feature columns to drop before fitting.",
    )
    compare_parser.add_argument("--presets", default="fast", choices=_PRESET_CHOICES)
    compare_parser.add_argument("--models", type=_parse_csv_list, default=None, help="Optional comma-separated model ids.")
    compare_parser.add_argument(
        "--exclude-models",
        type=_parse_csv_list,
        default=None,
        help="Optional comma-separated model ids to remove from the chosen preset.",
    )
    compare_parser.add_argument("--eval-metric", default="uno_c", choices=_METRIC_CHOICES)
    compare_parser.add_argument(
        "--split-strategy",
        default="fixed_split",
        choices=["fixed_split", "repeated_nested_cv"],
    )
    compare_parser.add_argument("--outer-folds", type=int, default=5)
    compare_parser.add_argument("--outer-repeats", type=int, default=1)
    compare_parser.add_argument("--inner-folds", type=int, default=3)
    compare_parser.add_argument(
        "--seeds",
        type=_parse_int_csv_list,
        default=None,
        help="Optional comma-separated random seeds. fixed_split expects exactly one seed.",
    )
    compare_parser.add_argument("--timeout-seconds", type=float, default=None)
    compare_parser.add_argument("--hpo-trials", type=int, default=0, help="Enable native HPO with this many trials.")
    compare_parser.add_argument("--hpo-timeout-seconds", type=float, default=None, help="Native HPO timeout.")
    compare_parser.add_argument(
        "--decision-thresholds",
        type=_parse_float_csv_list,
        default=None,
        help="Comma-separated decision thresholds for net-benefit reporting.",
    )
    compare_parser.add_argument("--save-path", default=None, help="Optional directory for benchmark outputs.")
    _add_foundation_flag(compare_parser)
    compare_parser.add_argument("--dry-run", action="store_true", help="Show resolved compare settings without fitting.")

    pilot_parser = subparsers.add_parser(
        "pilot",
        help="Run a small user-dataset pilot and print aggregate C-index metrics.",
    )
    pilot_parser.add_argument("--data", required=True, help="Path to CSV or Parquet data.")
    pilot_parser.add_argument("--time-col", required=True, help="Name of the duration column.")
    pilot_parser.add_argument("--event-col", required=True, help="Name of the event indicator column.")
    pilot_parser.add_argument("--dataset-name", default="user_dataset")
    pilot_parser.add_argument("--id-col", default=None, help="Optional identifier column to drop before fitting.")
    pilot_parser.add_argument(
        "--drop-columns",
        type=_parse_csv_list,
        default=None,
        help="Optional comma-separated feature columns to drop before fitting.",
    )
    pilot_parser.add_argument("--presets", default="fast", choices=_PRESET_CHOICES)
    pilot_parser.add_argument("--models", type=_parse_csv_list, default=None, help="Optional comma-separated model ids.")
    pilot_parser.add_argument(
        "--exclude-models",
        type=_parse_csv_list,
        default=None,
        help="Optional comma-separated model ids to remove from the chosen preset.",
    )
    pilot_parser.add_argument("--eval-metric", default="uno_c", choices=_METRIC_CHOICES)
    pilot_parser.add_argument(
        "--repeated",
        action="store_true",
        help="Use a small repeated nested-CV pilot instead of one fixed split.",
    )
    pilot_parser.add_argument(
        "--outer-folds",
        type=int,
        default=_PILOT_REPEATED_DEFAULT_FOLDS,
        help="Outer folds for --repeated.",
    )
    pilot_parser.add_argument(
        "--outer-repeats",
        type=int,
        default=_PILOT_REPEATED_DEFAULT_REPEATS,
        help="Outer repeats for --repeated.",
    )
    pilot_parser.add_argument(
        "--inner-folds",
        type=int,
        default=_PILOT_DEFAULT_INNER_FOLDS,
        help="Inner folds used for HPO selection.",
    )
    pilot_parser.add_argument(
        "--seeds",
        type=_parse_int_csv_list,
        default=None,
        help="Optional comma-separated random seeds. Fixed pilot expects one seed.",
    )
    pilot_parser.add_argument("--timeout-seconds", type=float, default=None)
    pilot_parser.add_argument("--hpo-trials", type=int, default=1, help="Native HPO trials for the HPO pilot track.")
    pilot_parser.add_argument("--hpo-timeout-seconds", type=float, default=None, help="Native HPO timeout.")
    pilot_parser.add_argument(
        "--decision-thresholds",
        type=_parse_float_csv_list,
        default=None,
        help="Comma-separated decision thresholds for net-benefit reporting.",
    )
    pilot_parser.add_argument("--save-path", default=None, help="Optional directory for pilot outputs.")
    _add_foundation_flag(pilot_parser)
    pilot_parser.add_argument("--dry-run", action="store_true", help="Show resolved pilot settings without fitting.")

    foundation_check_parser = subparsers.add_parser(
        "foundation-check",
        help="Inspect whether optional foundation-model adapters are installed and ready to run.",
    )
    foundation_check_parser.add_argument(
        "--models",
        type=_parse_csv_list,
        default=None,
        help="Optional comma-separated foundation method ids to inspect.",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Plan, inspect, run, and summarize config-driven benchmark experiments.",
    )
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)

    benchmark_plan_parser = benchmark_subparsers.add_parser("plan", help="Print benchmark run-unit counts.")
    _add_benchmark_common_args(benchmark_plan_parser)

    benchmark_doctor_parser = benchmark_subparsers.add_parser("doctor", help="Validate benchmark config readiness.")
    _add_benchmark_common_args(benchmark_doctor_parser)

    benchmark_run_parser = benchmark_subparsers.add_parser("run", help="Run a benchmark YAML config.")
    _add_benchmark_common_args(benchmark_run_parser)
    benchmark_run_parser.add_argument("--output-dir", default=None, help="Optional benchmark output directory.")
    benchmark_run_parser.add_argument("--resume", action="store_true", help="Resume from an existing output directory.")
    benchmark_run_parser.add_argument("--max-retries", type=int, default=0, help="Retry failed runs this many times.")
    benchmark_run_parser.add_argument(
        "--regenerate-splits",
        action="store_true",
        help="Allow split artifact regeneration when an existing manifest payload mismatches.",
    )
    benchmark_run_parser.add_argument("--dry-run", action="store_true", help="Validate setup without fitting models.")

    benchmark_report_parser = benchmark_subparsers.add_parser("report", help="Summarize benchmark output artifacts.")
    benchmark_report_parser.add_argument("output_dir", help="Benchmark output directory to summarize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        predictor = SurvivalPredictor(
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
        return

    if args.command in {"compare", "pilot"}:
        split_strategy = args.split_strategy if args.command == "compare" else "fixed_split"
        outer_folds = args.outer_folds
        outer_repeats = args.outer_repeats
        seeds = args.seeds
        benchmark_id = None
        if args.command == "pilot":
            benchmark_id = "user_pilot_fixed"
            if args.repeated:
                split_strategy = "repeated_nested_cv"
                seeds = args.seeds or _default_pilot_repeated_seeds(args.outer_repeats)
                benchmark_id = "user_pilot_cv"
            else:
                outer_repeats = 1
                seeds = args.seeds or [11]

        summary = compare_survival_models(
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
            hpo=_hpo_config_from_args(args),
            decision_curve_thresholds=args.decision_thresholds,
            output_dir=args.save_path,
            benchmark_id=benchmark_id,
            dry_run=args.dry_run,
        )
        _print_json(summary)
        return

    if args.command == "foundation-check":
        if args.models is None:
            statuses = list(foundation_runtime_catalog())
        else:
            statuses = [foundation_runtime_status_for_method(method_id) for method_id in args.models]
        _print_json([asdict(status) for status in statuses])
        return

    if args.command == "benchmark":
        repo_root = Path(__file__).resolve().parents[1]
        if args.benchmark_command == "report":
            _print_json(benchmark_report(Path(args.output_dir)))
            return

        benchmark_cfg = load_benchmark_config(repo_root, args.benchmark_config)
        if args.benchmark_command == "plan":
            _print_json(benchmark_plan(repo_root, benchmark_cfg, **_benchmark_plan_kwargs(args)))
            return
        if args.benchmark_command == "doctor":
            _print_json(benchmark_doctor(repo_root, benchmark_cfg, **_benchmark_plan_kwargs(args)))
            return
        if args.benchmark_command == "run":
            selected_cfg, dataset_override, method_override = _benchmark_run_config_and_overrides(benchmark_cfg, args)
            run_benchmark(
                repo_root=repo_root,
                benchmark_cfg=selected_cfg,
                dataset_override=dataset_override,
                method_override=method_override,
                limit_seeds=args.limit_seeds,
                dry_run=args.dry_run,
                output_dir=_resolve_optional_repo_path(repo_root, args.output_dir),
                resume=bool(args.resume),
                max_retries=max(int(args.max_retries), 0),
                regenerate_splits=bool(args.regenerate_splits),
            )
            return

    raise ValueError(f"Unsupported command '{args.command}'.")


if __name__ == "__main__":
    main()
