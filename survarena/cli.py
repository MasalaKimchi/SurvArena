from __future__ import annotations

import argparse
import json

from survarena.api import SurvivalPredictor


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
    fit_parser.add_argument("--eval-metric", default="harrell_c", choices=["harrell_c", "uno_c"])
    fit_parser.add_argument("--num-trials", type=int, default=None)
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
    fit_parser.add_argument("--save-path", default=None, help="Optional artifact directory root.")
    fit_parser.add_argument("--dataset-name", default="user_dataset")
    fit_parser.add_argument("--verbose", action="store_true", help="Show underlying tuning logs.")
    fit_parser.add_argument(
        "--enable-foundation-models",
        action="store_true",
        help="Include experimental tabular foundation-model adapters when supported.",
    )
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
            random_state=args.random_state,
            save_path=args.save_path,
            verbose=args.verbose,
            enable_foundation_models=args.enable_foundation_models,
        )
        predictor.fit(
            args.train,
            tuning_data=args.tuning,
            test_data=args.test,
            dataset_name=args.dataset_name,
            holdout_frac=args.holdout_frac,
            time_limit=args.time_limit,
        )
        print(json.dumps(predictor.fit_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
