from __future__ import annotations

import argparse
from pathlib import Path

from survarena.benchmark.runner import run_benchmark
from survarena.config import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SurvArena benchmark.")
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default="configs/benchmark/standard_v1.yaml",
        help="Path to benchmark YAML config.",
    )
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--method", type=str, default=None, help="Optional method override.")
    parser.add_argument("--limit-seeds", type=int, default=None, help="Use first N seeds only.")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials override.")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without fitting models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        benchmark_cfg = read_yaml(repo_root / args.benchmark_config)
    except ModuleNotFoundError as exc:
        if args.dry_run:
            print("Dry run completed with missing optional dependency.")
            print(f"missing_module={exc.name}")
            print("Install the required package extras before full benchmark execution.")
            return
        raise
    run_benchmark(
        repo_root=repo_root,
        benchmark_cfg=benchmark_cfg,
        dataset_override=args.dataset,
        method_override=args.method,
        limit_seeds=args.limit_seeds,
        n_trials_override=args.n_trials,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
