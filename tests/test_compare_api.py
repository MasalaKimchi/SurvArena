from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from survarena.api.compare import compare_survival_models
from survarena.data.splitters import SplitDefinition


def test_compare_survival_models_writes_benchmark_style_outputs(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    split = SplitDefinition(
        split_id="fixed_split_0",
        seed=11,
        repeat=0,
        fold=0,
        train_idx=np.asarray([0, 1, 2, 3], dtype=int),
        test_idx=np.asarray([4, 5], dtype=int),
        val_idx=np.asarray([2, 3], dtype=int),
    )

    monkeypatch.setattr("survarena.api.compare.load_or_create_splits", lambda **kwargs: [split])

    def fake_evaluate_split(**kwargs) -> dict[str, object]:
        return {
            "run_payload": {
                "manifest": {"run_id": "toy_coxph_fixed_split_0_seed11"},
                "metrics": {"status": "success"},
                "failure": None,
            },
            "benchmark_id": kwargs["benchmark_id"],
            "dataset_id": kwargs["dataset_id"],
            "method_id": kwargs["method_id"],
            "split_id": kwargs["split"].split_id,
            "seed": kwargs["split"].seed,
            "primary_metric": kwargs["primary_metric"],
            "validation_score": 0.8,
            "uno_c": 0.79,
            "harrell_c": 0.8,
            "ibs": 0.2,
            "td_auc_25": 0.78,
            "td_auc_50": 0.79,
            "td_auc_75": 0.8,
            "tuning_time_sec": 0.1,
            "runtime_sec": 0.2,
            "fit_time_sec": 0.05,
            "infer_time_sec": 0.02,
            "peak_memory_mb": 128.0,
            "status": "success",
        }

    monkeypatch.setattr("survarena.api.compare.evaluate_split", fake_evaluate_split)

    output_dir = tmp_path / "compare_outputs"
    summary = compare_survival_models(
        frame,
        time_col="time",
        event_col="event",
        dataset_name="toy_dataset",
        models=["coxph"],
        output_dir=output_dir,
    )

    assert summary["benchmark_id"] == "user_compare_fixed"
    assert summary["methods"] == ["coxph"]
    assert summary["split_count"] == 1
    assert summary["output_dir"] == str(output_dir)
    assert (output_dir / "experiment_manifest.json").exists()
    assert (output_dir / "fold_results.csv").exists()
    assert (output_dir / "user_compare_fixed_seed_summary.csv").exists()
    assert (output_dir / "user_compare_fixed_overall_summary.json").exists()
    assert (output_dir / "user_compare_fixed_leaderboard.csv").exists()
    assert (output_dir / "user_compare_fixed_leaderboard.json").exists()
    assert (output_dir / "user_compare_fixed_rank_summary.csv").exists()
    assert (output_dir / "user_compare_fixed_pairwise_win_rate.csv").exists()
    assert (output_dir / "user_compare_fixed_bootstrap_ci.csv").exists()
    assert (output_dir / "user_compare_fixed_elo_ratings.csv").exists()
    assert (output_dir / "user_compare_fixed_failure_summary.csv").exists()
    assert (output_dir / "user_compare_fixed_missing_metric_summary.csv").exists()
    assert (output_dir / "user_compare_fixed_dataset_curation.csv").exists()
    assert (output_dir / "user_compare_fixed_manuscript_summary.json").exists()
    assert not (output_dir / "user_compare_fixed_run_records.jsonl.gz").exists()
    assert (output_dir / "user_compare_fixed_run_records_compact.jsonl.gz").exists()
    assert (output_dir / "user_compare_fixed_run_records_compact_index.json").exists()
    assert (output_dir / "experiment_navigator.json").exists()
    assert (output_dir / "README.md").exists()


def test_compare_exports_exclude_parity_ineligible_rows(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    split = SplitDefinition(
        split_id="fixed_split_0",
        seed=11,
        repeat=0,
        fold=0,
        train_idx=np.asarray([0, 1, 2, 3], dtype=int),
        test_idx=np.asarray([4, 5], dtype=int),
        val_idx=np.asarray([2, 3], dtype=int),
    )
    monkeypatch.setattr("survarena.api.compare.load_or_create_splits", lambda **kwargs: [split])

    def fake_evaluate_split(**kwargs) -> dict[str, object]:
        enabled = bool(kwargs.get("hpo_cfg", {}).get("enabled", False))
        status = "failed" if enabled else "success"
        return {
            "run_payload": {
                "manifest": {"run_id": "toy_coxph_fixed_split_0_seed11"},
                "metrics": {"status": status},
                "hpo_metadata": {"status": status, "trial_count": 0},
                "failure": None if status == "success" else {"traceback": "boom"},
                "status": status,
            },
            "benchmark_id": kwargs["benchmark_id"],
            "dataset_id": kwargs["dataset_id"],
            "method_id": kwargs["method_id"],
            "split_id": kwargs["split"].split_id,
            "seed": kwargs["split"].seed,
            "primary_metric": kwargs["primary_metric"],
            "validation_score": 0.8,
            "uno_c": 0.79,
            "harrell_c": 0.8,
            "ibs": 0.2,
            "td_auc_25": 0.78,
            "td_auc_50": 0.79,
            "td_auc_75": 0.8,
            "tuning_time_sec": 0.1,
            "runtime_sec": 0.2,
            "fit_time_sec": 0.05,
            "infer_time_sec": 0.02,
            "peak_memory_mb": 128.0,
            "status": status,
        }

    monkeypatch.setattr("survarena.api.compare.evaluate_split", fake_evaluate_split)

    output_dir = tmp_path / "compare_outputs"
    compare_survival_models(
        frame,
        time_col="time",
        event_col="event",
        dataset_name="toy_dataset",
        models=["coxph"],
        output_dir=output_dir,
    )

    fold_results = pd.read_csv(output_dir / "fold_results.csv")
    assert "parity_eligible" in fold_results.columns
    assert not fold_results["parity_eligible"].any()

    rank_summary = pd.read_csv(output_dir / "user_compare_fixed_rank_summary.csv")
    pairwise = pd.read_csv(output_dir / "user_compare_fixed_pairwise_win_rate.csv")
    try:
        pairwise_sig = pd.read_csv(output_dir / "user_compare_fixed_pairwise_significance.csv")
    except EmptyDataError:
        pairwise_sig = pd.DataFrame()
    assert rank_summary.empty
    assert pairwise.empty
    assert pairwise_sig.empty


def test_compare_summary_includes_requested_vs_realized_budget_fields(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    split = SplitDefinition(
        split_id="fixed_split_0",
        seed=11,
        repeat=0,
        fold=0,
        train_idx=np.asarray([0, 1, 2, 3], dtype=int),
        test_idx=np.asarray([4, 5], dtype=int),
        val_idx=np.asarray([2, 3], dtype=int),
    )
    monkeypatch.setattr("survarena.api.compare.load_or_create_splits", lambda **kwargs: [split])

    def fake_evaluate_split(**kwargs) -> dict[str, object]:
        enabled = bool(kwargs.get("hpo_cfg", {}).get("enabled", False))
        trial_count = 4 if enabled else 0
        return {
            "run_payload": {
                "manifest": {"run_id": "toy_coxph_fixed_split_0_seed11"},
                "metrics": {"status": "success"},
                "hpo_metadata": {"status": "success", "trial_count": trial_count},
                "failure": None,
                "status": "success",
            },
            "benchmark_id": kwargs["benchmark_id"],
            "dataset_id": kwargs["dataset_id"],
            "method_id": kwargs["method_id"],
            "split_id": kwargs["split"].split_id,
            "seed": kwargs["split"].seed,
            "primary_metric": kwargs["primary_metric"],
            "validation_score": 0.8,
            "uno_c": 0.79,
            "harrell_c": 0.8,
            "ibs": 0.2,
            "td_auc_25": 0.78,
            "td_auc_50": 0.79,
            "td_auc_75": 0.8,
            "tuning_time_sec": 0.1,
            "runtime_sec": 0.2,
            "fit_time_sec": 0.05,
            "infer_time_sec": 0.02,
            "peak_memory_mb": 128.0,
            "status": "success",
        }

    monkeypatch.setattr("survarena.api.compare.evaluate_split", fake_evaluate_split)

    output_dir = tmp_path / "compare_outputs"
    compare_survival_models(
        frame,
        time_col="time",
        event_col="event",
        dataset_name="toy_dataset",
        models=["coxph"],
        output_dir=output_dir,
        hpo={"max_trials": 7, "timeout_seconds": 123.0, "sampler": "tpe", "pruner": "median"},
    )

    fold_results = pd.read_csv(output_dir / "fold_results.csv")
    for required in ("hpo_mode", "requested_max_trials", "requested_timeout_seconds", "realized_trial_count"):
        assert required in fold_results.columns
    assert set(fold_results["hpo_mode"]) == {"no_hpo", "hpo"}
    assert set(fold_results["requested_max_trials"]) == {7}
    assert set(fold_results["requested_timeout_seconds"]) == {123.0}
    assert fold_results["realized_trial_count"].max() == 4
