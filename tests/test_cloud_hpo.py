from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.merge_cloud_benchmark_shards import merge_shards
from scripts.run_cloud_benchmark_shard import build_run_benchmark_argv, default_output_dir
from survarena.config import read_yaml


def test_cloud_shard_manifest_matches_benchmark_methods() -> None:
    manifest = read_yaml(Path("configs/cloud/tabarena_like_hpo_v1_shards.yaml"))
    benchmark = read_yaml(Path(manifest["benchmark_config"]))

    benchmark_methods = list(benchmark["methods"])
    shard_methods = [shard["method_id"] for shard in manifest["shards"]]

    assert shard_methods == benchmark_methods
    assert len(shard_methods) == len(set(shard_methods))


def test_cloud_shard_manifest_uses_method_isolated_output_paths() -> None:
    manifest = read_yaml(Path("configs/cloud/tabarena_like_hpo_v1_shards.yaml"))
    output_root = Path(manifest["output_root"])

    for shard in manifest["shards"]:
        method_id = shard["method_id"]
        output_dir = Path(shard["output_dir"])
        assert output_dir == output_root / method_id
        assert f"--method {method_id}" in shard["command"]
        assert f"--output-dir {output_dir}" in shard["command"]


def test_cloud_runner_builds_resumable_method_shard_argv() -> None:
    argv = build_run_benchmark_argv(
        config=Path("configs/benchmark/tabarena_like_hpo_v1.yaml"),
        method="coxph",
        output_dir=default_output_dir("tabarena_like_hpo_v1", "coxph"),
        resume=True,
        max_retries=1,
        regenerate_splits=False,
    )

    assert argv == [
        "--config",
        "configs/benchmark/tabarena_like_hpo_v1.yaml",
        "--method",
        "coxph",
        "--output-dir",
        "results/cloud/tabarena_like_hpo_v1/coxph",
        "--max-retries",
        "1",
        "--resume",
    ]


def _write_shard(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_merge_cloud_shards_handles_partial_failures(tmp_path) -> None:
    base_rows = [
        {
            "benchmark_id": "cloud",
            "dataset_id": "whas500",
            "seed": 11,
            "split_id": "repeat_0_fold_0",
            "status": "success",
            "runtime_sec": 1.0,
            "requested_max_trials": 2,
            "realized_trial_count": 2,
            "requested_timeout_seconds": 30,
            "requested_sampler": "random",
            "requested_pruner": "nop",
            "hpo_budget_tier": "unit",
            "hpo_config_target": 2,
            "hpo_capped": False,
        },
        {
            "benchmark_id": "cloud",
            "dataset_id": "gbsg2",
            "seed": 11,
            "split_id": "repeat_0_fold_0",
            "status": "failed",
            "runtime_sec": 0.5,
            "requested_max_trials": 2,
            "realized_trial_count": 0,
            "hpo_budget_tier": "unit",
            "hpo_config_target": 2,
            "hpo_capped": False,
        },
    ]
    rows_by_method = {
        "coxph": [
            {**base_rows[0], "method_id": "coxph", "hpo_mode": "no_hpo", "uno_c": 0.60},
            {**base_rows[0], "method_id": "coxph", "hpo_mode": "hpo", "uno_c": 0.65},
            {**base_rows[1], "method_id": "coxph", "hpo_mode": "hpo", "uno_c": None},
        ],
        "coxnet": [
            {**base_rows[0], "method_id": "coxnet", "hpo_mode": "no_hpo", "uno_c": 0.55},
            {**base_rows[0], "method_id": "coxnet", "hpo_mode": "hpo", "uno_c": 0.58},
        ],
    }
    for method_id, rows in rows_by_method.items():
        _write_shard(tmp_path / "shards" / method_id / "whas500" / f"{method_id}_fold_results.csv", rows)

    outputs = merge_shards(shards_dir=tmp_path / "shards", output_dir=tmp_path / "combined", primary_metric="uno_c")

    for path in outputs.values():
        assert path.exists()
    combined = pd.read_csv(outputs["combined_all"])
    success = pd.read_csv(outputs["combined_success"])
    budget = pd.read_csv(outputs["hpo_budget_summary"])
    assert len(combined) == 5
    assert len(success) == 4
    assert {"combined_fold_results_success.csv", "combined_leaderboard_all.csv", "hpo_budget_summary.csv"}.issubset(
        {path.name for path in outputs.values()}
    )
    assert budget["realized_trials_total"].sum() < budget["requested_trials_total"].sum()
