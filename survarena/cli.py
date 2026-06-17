from __future__ import annotations

import argparse
from pathlib import Path

from survarena.api import SurvivalPredictor, compare_survival_models
from survarena.benchmark.overview import benchmark_doctor, benchmark_plan, benchmark_report, load_benchmark_config
from survarena.benchmark.runner import run_benchmark
from survarena.commands.handlers import CliDependencies, run_cli_command
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
        default="configs/benchmark/manuscript_v1.yaml",
        help="Path to benchmark YAML config.",
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset", default=None, help="Optional single dataset override.")
    dataset_group.add_argument("--datasets", type=_parse_csv_list, default=None, help="Optional comma-separated dataset ids.")
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument("--method", default=None, help="Optional single method override.")
    method_group.add_argument("--methods", type=_parse_csv_list, default=None, help="Optional comma-separated method ids.")
    parser.add_argument("--limit-seeds", type=int, default=None, help="Use first N seeds only.")


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
    benchmark_doctor_parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Import selected method adapters to catch missing optional dependencies before fitting.",
    )
    benchmark_doctor_parser.add_argument(
        "--load-datasets",
        action="store_true",
        help="Load selected datasets and include health summaries in the readiness report.",
    )

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


def _cli_dependencies() -> CliDependencies:
    return CliDependencies(
        survival_predictor_cls=SurvivalPredictor,
        compare_survival_models=compare_survival_models,
        foundation_runtime_catalog=foundation_runtime_catalog,
        foundation_runtime_status_for_method=foundation_runtime_status_for_method,
        load_benchmark_config=load_benchmark_config,
        benchmark_plan=benchmark_plan,
        benchmark_doctor=benchmark_doctor,
        benchmark_report=benchmark_report,
        run_benchmark=run_benchmark,
    )


def main() -> None:
    run_cli_command(
        parse_args(),
        deps=_cli_dependencies(),
        repo_root=Path(__file__).resolve().parents[1],
    )


if __name__ == "__main__":
    main()
