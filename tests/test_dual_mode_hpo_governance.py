from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from survarena.benchmark.runner import run_benchmark
from survarena.data.splitters import SplitDefinition


def _base_cfg() -> dict[str, Any]:
    return {
        "benchmark_id": "dual_mode_contract",
        "profile": "smoke",
        "datasets": ["toy_dataset"],
        "methods": ["coxph"],
        "split_strategy": "repeated_nested_cv",
        "seeds": [11],
        "outer_folds": 3,
        "outer_repeats": 1,
        "inner_folds": 2,
        "primary_metric": "harrell_c",
        "hpo": {
            "enabled": True,
            "max_trials": 5,
            "timeout_seconds": 30,
            "sampler": "tpe",
            "pruner": "median",
            "n_startup_trials": 3,
        },
    }


def _split() -> SplitDefinition:
    return SplitDefinition(
        split_id="fixed_split_0",
        seed=11,
        repeat=0,
        fold=0,
        train_idx=np.asarray([0, 1, 2, 3], dtype=int),
        test_idx=np.asarray([4, 5], dtype=int),
        val_idx=np.asarray([2, 3], dtype=int),
    )


def _patch_runner_dependencies(monkeypatch, captured_run_records: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
        }
    )
    dataset = SimpleNamespace(
        X=frame[["age"]].copy(),
        time=frame["time"].to_numpy(dtype=float),
        event=frame["event"].to_numpy(dtype=int),
        metadata=SimpleNamespace(feature_types={"age": "continuous"}),
    )
    track = SimpleNamespace(track_id="identity")

    monkeypatch.setattr("survarena.data.loaders.load_dataset", lambda *_args, **_kwargs: dataset)
    monkeypatch.setattr("survarena.data.splitters.load_or_create_splits", lambda *_args, **_kwargs: [_split()])
    monkeypatch.setattr("survarena.data.robustness.resolve_robustness_tracks", lambda *_args, **_kwargs: [track])
    monkeypatch.setattr("survarena.data.robustness.apply_robustness_track", lambda X, **_kwargs: X)
    monkeypatch.setattr("survarena.data.robustness.apply_label_noise", lambda event, **_kwargs: event)
    monkeypatch.setattr(
        "survarena.benchmark.runner.read_yaml",
        lambda *_args, **_kwargs: {"method_id": "coxph", "default_params": {"alpha": 0.1}},
    )
    monkeypatch.setattr(
        "survarena.logging.export.export_fold_results",
        lambda *_args, **_kwargs: pd.DataFrame(
            [
                {
                    "benchmark_id": "dual_mode_contract",
                    "dataset_id": "toy_dataset__identity",
                    "method_id": "coxph",
                    "split_id": "fixed_split_0__identity",
                    "seed": 11,
                    "status": "success",
                    "harrell_c": 0.78,
                }
            ]
        ),
    )
    monkeypatch.setattr("survarena.logging.export.export_seed_summary", lambda *_args, **_kwargs: pd.DataFrame([]))
    monkeypatch.setattr("survarena.logging.export.export_overall_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("survarena.logging.export.export_leaderboard", lambda *_args, **_kwargs: pd.DataFrame([]))
    monkeypatch.setattr("survarena.logging.export.export_manuscript_comparison", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("survarena.logging.export.export_dataset_curation_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "survarena.logging.export.export_run_ledger",
        lambda *_args, **_kwargs: captured_run_records.extend(_args[1]),
    )
    monkeypatch.setattr("survarena.logging.export.export_hpo_trials", lambda *_args, **_kwargs: None)


def test_run_records_include_hpo_mode_and_parity_key(tmp_path: Path, monkeypatch) -> None:
    captured_run_records: list[dict[str, Any]] = []
    _patch_runner_dependencies(monkeypatch, captured_run_records)

    def fake_evaluate_split(**kwargs) -> dict[str, Any]:
        return {
            "run_payload": {
                "manifest": {"run_id": "toy_dataset_coxph_fixed_split_0_seed11"},
                "metrics": {"status": "success"},
                "hpo_metadata": {"status": "disabled", "trial_count": 0},
                "hpo_trials": [],
                "failure": None,
            },
            "benchmark_id": kwargs["benchmark_id"],
            "dataset_id": kwargs["dataset_id"],
            "method_id": kwargs["method_id"],
            "split_id": kwargs["split"].split_id,
            "seed": kwargs["split"].seed,
            "primary_metric": kwargs["primary_metric"],
            "harrell_c": 0.78,
            "status": "success",
        }

    monkeypatch.setattr("survarena.benchmark.runner.evaluate_split", fake_evaluate_split)

    run_benchmark(repo_root=tmp_path, benchmark_cfg=_base_cfg(), output_dir=tmp_path / "outputs")

    assert captured_run_records
    metrics = captured_run_records[0]["metrics"]
    assert metrics["hpo_mode"] in {"no_hpo", "hpo"}
    assert metrics["parity_key"] == "toy_dataset__identity|fixed_split_0__identity|11|coxph"


def test_dual_mode_execution_order_is_no_hpo_then_hpo(tmp_path: Path, monkeypatch) -> None:
    captured_run_records: list[dict[str, Any]] = []
    _patch_runner_dependencies(monkeypatch, captured_run_records)
    observed_modes: list[str] = []

    def fake_evaluate_split(**kwargs) -> dict[str, Any]:
        observed_modes.append("hpo" if kwargs["hpo_cfg"].get("enabled") else "no_hpo")
        return {
            "run_payload": {
                "manifest": {"run_id": "toy_dataset_coxph_fixed_split_0_seed11"},
                "metrics": {"status": "success"},
                "hpo_metadata": {"status": "disabled", "trial_count": 0},
                "hpo_trials": [],
                "failure": None,
            },
            "benchmark_id": kwargs["benchmark_id"],
            "dataset_id": kwargs["dataset_id"],
            "method_id": kwargs["method_id"],
            "split_id": kwargs["split"].split_id,
            "seed": kwargs["split"].seed,
            "primary_metric": kwargs["primary_metric"],
            "harrell_c": 0.78,
            "status": "success",
        }

    monkeypatch.setattr("survarena.benchmark.runner.evaluate_split", fake_evaluate_split)

    run_benchmark(repo_root=tmp_path, benchmark_cfg=_base_cfg(), output_dir=tmp_path / "outputs")

    assert observed_modes == ["no_hpo", "hpo"]


def test_missing_mode_marks_pairing_unit_ineligible(tmp_path: Path, monkeypatch) -> None:
    captured_run_records: list[dict[str, Any]] = []
    _patch_runner_dependencies(monkeypatch, captured_run_records)

    def fake_evaluate_split(**kwargs) -> dict[str, Any]:
        return {
            "run_payload": {
                "manifest": {"run_id": "toy_dataset_coxph_fixed_split_0_seed11"},
                "metrics": {"status": "success"},
                "hpo_metadata": {"status": "disabled", "trial_count": 0},
                "hpo_trials": [],
                "failure": None,
            },
            "benchmark_id": kwargs["benchmark_id"],
            "dataset_id": kwargs["dataset_id"],
            "method_id": kwargs["method_id"],
            "split_id": kwargs["split"].split_id,
            "seed": kwargs["split"].seed,
            "primary_metric": kwargs["primary_metric"],
            "harrell_c": 0.78,
            "status": "success",
        }

    monkeypatch.setattr("survarena.benchmark.runner.evaluate_split", fake_evaluate_split)

    run_benchmark(repo_root=tmp_path, benchmark_cfg=_base_cfg(), output_dir=tmp_path / "outputs")

    assert captured_run_records
    row = captured_run_records[0]
    assert row["metrics"]["parity_eligible"] is False
