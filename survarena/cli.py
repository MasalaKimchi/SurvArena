from __future__ import annotations

import argparse
import json

from survarena.api import SurvivalPredictor, compare_survival_models


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SurvArena command line interface.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit_parser = subparsers.add_parser("fit", help="Fit a survival predictor on user-provided data.")
    fit_parser.add_argument("--train", required=True, help="Path to training CSV or Parquet data.")
    fit_parser.add_argument("--tuning", default=None, help="Optional tuning/validation CSV or Parquet data.")
    fit_parser.add_argument("--test", default=None, help="Optional test CSV or Parquet data.")
    fit_parser.add_argument("--time-col", required=True, help="Name of the duration column.")
    fit_parser.add_argument("--event-col", required=True, help="Name of the event indicator column.")
    fit_parser.add_argument("--presets", default="all", choices=["fast", "medium", "best", "all", "foundation"])
    fit_parser.add_argument("--models", type=_parse_csv_list, default=None, help="Optional comma-separated model ids.")
    fit_parser.add_argument(
        "--exclude-models",
        type=_parse_csv_list,
        default=None,
        help="Optional comma-separated model ids to remove from the chosen preset.",
    )
    fit_parser.add_argument("--eval-metric", default="harrell_c", choices=["harrell_c", "uno_c"])
    fit_parser.add_argument("--num-trials", type=int, default=None)
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
    fit_parser.add_argument(
        "--enable-foundation-models",
        action="store_true",
        help="Include experimental tabular foundation-model adapters when supported.",
    )

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
    compare_parser.add_argument("--presets", default="fast", choices=["fast", "medium", "best", "all", "foundation"])
    compare_parser.add_argument("--models", type=_parse_csv_list, default=None, help="Optional comma-separated model ids.")
    compare_parser.add_argument(
        "--exclude-models",
        type=_parse_csv_list,
        default=None,
        help="Optional comma-separated model ids to remove from the chosen preset.",
    )
    compare_parser.add_argument("--eval-metric", default="harrell_c", choices=["harrell_c", "uno_c"])
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
    compare_parser.add_argument("--n-trials", type=int, default=0)
    compare_parser.add_argument("--timeout-seconds", type=float, default=None)
    compare_parser.add_argument("--save-path", default=None, help="Optional directory for benchmark outputs.")
    compare_parser.add_argument(
        "--enable-foundation-models",
        action="store_true",
        help="Include experimental tabular foundation-model adapters when supported.",
    )
    compare_parser.add_argument("--dry-run", action="store_true", help="Show resolved compare settings without fitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        predictor = SurvivalPredictor(
            label_time=args.time_col,
            label_event=args.event_col,
            eval_metric=args.eval_metric,
            presets=args.presets,
            num_trials=args.num_trials,
            included_models=args.models,
            excluded_models=args.exclude_models,
            retain_top_k_models=None if args.retain_all_models else args.retain_top_k_models,
            random_state=args.random_state,
            save_path=args.save_path,
            verbose=args.verbose,
            enable_foundation_models=args.enable_foundation_models,
        )
        hyperparameter_tune_kwargs = None
        if args.num_trials is not None or args.tuning_timeout is not None:
            hyperparameter_tune_kwargs = {}
            if args.num_trials is not None:
                hyperparameter_tune_kwargs["num_trials"] = args.num_trials
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
        print(json.dumps(predictor.fit_summary(), indent=2, sort_keys=True))
        return

    if args.command == "compare":
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
            split_strategy=args.split_strategy,
            outer_folds=args.outer_folds,
            outer_repeats=args.outer_repeats,
            inner_folds=args.inner_folds,
            seeds=args.seeds,
            n_trials=args.n_trials,
            timeout_seconds=args.timeout_seconds,
            output_dir=args.save_path,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    raise ValueError(f"Unsupported command '{args.command}'.")


if __name__ == "__main__":
    main()
