from __future__ import annotations

import argparse
from pathlib import Path

from survarena.config import read_yaml
from survarena.run_benchmark import main as run_benchmark_main


def default_output_dir(benchmark_id: str, method_id: str) -> Path:
    return Path("results") / "cloud" / benchmark_id / method_id


def build_run_benchmark_argv(
    *,
    config: Path,
    method: str,
    output_dir: Path,
    resume: bool,
    max_retries: int,
    regenerate_splits: bool,
) -> list[str]:
    argv = [
        "--config",
        str(config),
        "--method",
        method,
        "--output-dir",
        str(output_dir),
        "--max-retries",
        str(max_retries),
    ]
    if resume:
        argv.append("--resume")
    if regenerate_splits:
        argv.append("--regenerate-splits")
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one resumable cloud benchmark shard.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--regenerate-splits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config)
    methods = set(cfg.get("methods", []))
    if args.method not in methods:
        raise ValueError(f"Method '{args.method}' is not listed in {args.config}.")

    output_dir = args.output_dir or default_output_dir(str(cfg["benchmark_id"]), args.method)
    argv = build_run_benchmark_argv(
        config=args.config,
        method=args.method,
        output_dir=output_dir,
        resume=args.resume,
        max_retries=args.max_retries,
        regenerate_splits=args.regenerate_splits,
    )

    import sys

    previous_argv = sys.argv
    try:
        sys.argv = ["survarena.run_benchmark", *argv]
        run_benchmark_main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
