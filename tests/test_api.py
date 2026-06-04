from __future__ import annotations

import sys
import pandas as pd
from survarena import cli
from survarena import SurvivalPredictions
from survarena.benchmark import overview
from survarena.methods.foundation.readiness import FoundationRuntimeStatus
from pathlib import Path
import numpy as np
from survarena.api.compare import compare_survival_models
from survarena.data.splitters import SplitDefinition
import json
import math
import pytest
from types import SimpleNamespace
from survarena.api.predictor import PredictorModelResult, SurvivalPredictor
from survarena.automl.presets import PresetConfig
from survarena.evaluation.metrics import MetricBundle
from survarena.methods.base import BaseSurvivalMethod
import importlib


# --- test_cli.py ---


def test_survival_predictions_is_exported_from_public_api() -> None:
    predictions = SurvivalPredictions(risk=np.asarray([0.2]), survival=np.asarray([[0.9]]))

    assert predictions.risk.tolist() == [0.2]
    assert predictions.survival.tolist() == [[0.9]]


def test_fit_cli_passes_models_and_retention_flags(monkeypatch, capsys) -> None:
    class FakePredictor:
        init_kwargs: dict[str, object] | None = None
        fit_kwargs: dict[str, object] | None = None

        def __init__(self, **kwargs) -> None:
            FakePredictor.init_kwargs = kwargs

        def fit(self, train_data, **kwargs) -> "FakePredictor":  # noqa: ANN001
            FakePredictor.fit_kwargs = {"train_data": train_data, **kwargs}
            return self

        def fit_summary(self) -> dict[str, object]:
            return {"best_method_id": "rsf", "trained_models": ["rsf"]}

    monkeypatch.setattr(cli, "SurvivalPredictor", FakePredictor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "fit",
            "--train",
            "train.csv",
            "--time-col",
            "time",
            "--event-col",
            "event",
            "--models",
            "coxph,rsf",
            "--exclude-models",
            "coxph",
            "--retain-top-k-models",
            "2",
            "--autogluon-num-trials",
            "4",
            "--tuning-timeout",
            "30",
            "--dataset-name",
            "toy",
        ],
    )

    cli.main()

    assert FakePredictor.init_kwargs is not None
    assert FakePredictor.init_kwargs["included_models"] == ["coxph", "rsf"]
    assert FakePredictor.init_kwargs["excluded_models"] == ["coxph"]
    assert FakePredictor.init_kwargs["retain_top_k_models"] == 2
    assert FakePredictor.fit_kwargs is not None
    assert FakePredictor.fit_kwargs["train_data"] == "train.csv"
    assert FakePredictor.fit_kwargs["dataset_name"] == "toy"
    assert FakePredictor.fit_kwargs["hyperparameter_tune_kwargs"] == {"num_trials": 4, "timeout": 30.0}
    assert '"best_method_id": "rsf"' in capsys.readouterr().out


def test_compare_cli_invokes_user_compare_workflow(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_compare_survival_models(data, **kwargs):  # noqa: ANN001
        captured["data"] = data
        captured.update(kwargs)
        return {"benchmark_id": "user_compare_fixed", "methods": kwargs["models"], "output_dir": "tmp/results"}

    monkeypatch.setattr(cli, "compare_survival_models", fake_compare_survival_models)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "compare",
            "--data",
            "toy.csv",
            "--time-col",
            "time",
            "--event-col",
            "event",
            "--models",
            "coxph,rsf",
            "--split-strategy",
            "fixed_split",
            "--seeds",
            "11",
            "--save-path",
            "tmp/results",
        ],
    )

    cli.main()

    assert captured["data"] == "toy.csv"
    assert captured["time_col"] == "time"
    assert captured["event_col"] == "event"
    assert captured["models"] == ["coxph", "rsf"]
    assert captured["split_strategy"] == "fixed_split"
    assert captured["seeds"] == [11]
    assert captured["output_dir"] == "tmp/results"
    assert '"benchmark_id": "user_compare_fixed"' in capsys.readouterr().out


def test_pilot_cli_invokes_compare_with_fixed_pilot_defaults(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_compare_survival_models(data, **kwargs):  # noqa: ANN001
        captured["data"] = data
        captured.update(kwargs)
        return {
            "benchmark_id": kwargs["benchmark_id"],
            "leaderboard": [{"method_id": "coxph", "uno_c": 0.71, "harrell_c": 0.7}],
            "artifacts": {"leaderboard": "tmp/pilot/coxph_leaderboard.csv"},
        }

    monkeypatch.setattr(cli, "compare_survival_models", fake_compare_survival_models)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "pilot",
            "--data",
            "toy.csv",
            "--time-col",
            "time",
            "--event-col",
            "event",
        ],
    )

    cli.main()

    assert captured["data"] == "toy.csv"
    assert captured["split_strategy"] == "fixed_split"
    assert captured["outer_repeats"] == 1
    assert captured["inner_folds"] == 2
    assert captured["seeds"] == [11]
    assert captured["presets"] == "fast"
    assert captured["benchmark_id"] == "user_pilot_fixed"
    assert captured["hpo"] == {
        "enabled": True,
        "max_trials": 1,
        "timeout_seconds": None,
        "sampler": "tpe",
        "pruner": "median",
        "n_startup_trials": 8,
    }
    output = capsys.readouterr().out
    assert '"benchmark_id": "user_pilot_fixed"' in output
    assert '"uno_c": 0.71' in output
    assert "coxph_leaderboard.csv" in output


def test_pilot_cli_repeated_uses_small_cv_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_compare_survival_models(data, **kwargs):  # noqa: ANN001
        captured["data"] = data
        captured.update(kwargs)
        return {"benchmark_id": kwargs["benchmark_id"]}

    monkeypatch.setattr(cli, "compare_survival_models", fake_compare_survival_models)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "pilot",
            "--data",
            "toy.csv",
            "--time-col",
            "time",
            "--event-col",
            "event",
            "--repeated",
        ],
    )

    cli.main()

    assert captured["split_strategy"] == "repeated_nested_cv"
    assert captured["outer_folds"] == 3
    assert captured["outer_repeats"] == 2
    assert captured["seeds"] == [11, 23]
    assert captured["benchmark_id"] == "user_pilot_cv"


def test_pilot_cli_foundation_flag_includes_foundation_models(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_compare_survival_models(data, **kwargs):  # noqa: ANN001
        captured["data"] = data
        captured.update(kwargs)
        return {"benchmark_id": kwargs["benchmark_id"]}

    monkeypatch.setattr(cli, "compare_survival_models", fake_compare_survival_models)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "pilot",
            "--data",
            "toy.csv",
            "--time-col",
            "time",
            "--event-col",
            "event",
            "--foundation",
        ],
    )

    cli.main()

    assert captured["presets"] == "fast"
    assert captured["enable_foundation_models"] is True


def test_foundation_check_cli_emits_runtime_status(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "foundation_runtime_catalog",
        lambda: [
            FoundationRuntimeStatus(
                method_id="tabpfn_survival",
                dependency_module="tabpfn",
                install_extra="foundation-tabpfn",
                dependency_installed=True,
                runtime_ready=True,
                requires_hf_auth=True,
                auth_configured=False,
                install_command='python -m pip install -e ".[foundation-tabpfn]"',
                blocked_reason=None,
                warning_reason="Run `hf auth login` first.",
            )
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "foundation-check",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert '"method_id": "tabpfn_survival"' in output
    assert '"warning_reason": "Run `hf auth login` first."' in output


def test_benchmark_plan_cli_emits_run_unit_counts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "benchmark",
            "plan",
            "--config",
            "configs/benchmark/manuscript_v1.yaml",
            "--datasets",
            "whas500,gbsg2",
            "--methods",
            "coxph,rsf",
            "--limit-seeds",
            "1",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert '"benchmark_id": "manuscript_v1"' in output
    assert '"whas500"' in output
    assert '"gbsg2"' in output
    assert '"coxph"' in output
    assert '"rsf"' in output
    assert '"planned_run_units"' in output


def test_benchmark_doctor_cli_reports_missing_dataset(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "benchmark",
            "doctor",
            "--benchmark-config",
            "configs/benchmark/manuscript_v1.yaml",
            "--dataset",
            "not_a_dataset",
            "--method",
            "coxph",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert '"status": "error"' in output
    assert "Missing dataset config" in output


def test_benchmark_doctor_cli_supports_deeper_checks(monkeypatch, capsys) -> None:
    monkeypatch.setattr(overview, "get_method_class", lambda method_id: object())
    monkeypatch.setattr(
        overview,
        "_dataset_summary",
        lambda repo_root, dataset_id: {
            "dataset_id": dataset_id,
            "n_rows": 10,
            "n_features": 3,
            "n_events": 4,
            "event_rate": 0.4,
            "censoring_rate": 0.6,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "benchmark",
            "doctor",
            "--config",
            "configs/benchmark/manuscript_v1.yaml",
            "--dataset",
            "whas500",
            "--method",
            "coxph",
            "--check-imports",
            "--load-datasets",
        ],
    )

    cli.main()

    output = capsys.readouterr().out
    assert '"method_imports": true' in output
    assert '"dataset_load": true' in output
    assert '"dataset_summaries"' in output
    assert '"dataset_id": "whas500"' in output


def test_benchmark_doctor_checks_tabicl_foundation_readiness(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        overview,
        "foundation_runtime_status_for_method",
        lambda method_id: SimpleNamespace(
            dependency_installed=False,
            install_command='python -m pip install -e ".[foundation-tabarena]"',
            warning_reason=None,
        ),
    )

    issues: list[dict[str, str]] = []
    overview._append_method_issues(tmp_path, ["tabicl_survival"], issues, check_imports=False)

    assert any(issue["check"] == "foundation_dependency" for issue in issues)


def test_benchmark_run_cli_delegates_to_runner(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_benchmark(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "survarena",
            "benchmark",
            "run",
            "--config",
            "configs/benchmark/manuscript_v1.yaml",
            "--datasets",
            "whas500,gbsg2",
            "--methods",
            "coxph,rsf",
            "--limit-seeds",
            "1",
            "--output-dir",
            "tmp/benchmark",
            "--resume",
            "--max-retries",
            "2",
            "--dry-run",
        ],
    )

    cli.main()

    assert captured["benchmark_cfg"]["datasets"] == ["whas500", "gbsg2"]
    assert captured["benchmark_cfg"]["methods"] == ["coxph", "rsf"]
    assert captured["dataset_override"] is None
    assert captured["method_override"] is None
    assert captured["limit_seeds"] == 1
    assert captured["resume"] is True
    assert captured["max_retries"] == 2
    assert captured["dry_run"] is True
    assert str(captured["output_dir"]).endswith("tmp/benchmark")


def test_benchmark_report_cli_summarizes_fold_results(monkeypatch, tmp_path, capsys) -> None:
    pd.DataFrame(
        [
            {
                "dataset_id": "toy",
                "method_id": "coxph",
                "hpo_mode": "no_hpo",
                "status": "success",
                "uno_c": 0.7,
                "runtime_sec": 1.5,
            },
            {
                "dataset_id": "toy",
                "method_id": "rsf",
                "hpo_mode": "no_hpo",
                "status": "failed",
                "uno_c": 0.6,
                "runtime_sec": 2.5,
            },
        ]
    ).to_csv(tmp_path / "coxph_fold_results.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["survarena", "benchmark", "report", str(tmp_path)])

    cli.main()

    output = capsys.readouterr().out
    assert '"n_rows": 2' in output
    assert '"success": 1' in output
    assert '"failed": 1' in output
    assert '"primary_metric": "uno_c"' in output
    assert '"top_methods"' in output
    assert '"coverage"' in output
    assert '"method_id": "coxph"' in output


# --- test_compare_api.py ---


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
    assert summary["fold_results_rows"] == 1
    assert summary["comparison_modes"] == ["no_hpo"]
    assert summary["output_dir"] == str(output_dir)
    assert summary["artifacts"] == {
        "experiment_manifest": str(output_dir / "experiment_manifest.json"),
        "fold_results": str(output_dir / "coxph_fold_results.csv"),
        "leaderboard": str(output_dir / "coxph_leaderboard.csv"),
        "run_diagnostics": str(output_dir / "coxph_run_diagnostics.csv"),
    }
    assert {row["hpo_mode"] for row in summary["leaderboard"]} == {"no_hpo"}
    assert {row["method_id"] for row in summary["leaderboard"]} == {"coxph"}
    assert {row["uno_c"] for row in summary["leaderboard"]} == {0.79}
    assert (output_dir / "experiment_manifest.json").exists()
    assert (output_dir / "coxph_fold_results.csv").exists()
    assert (output_dir / "coxph_leaderboard.csv").exists()
    assert (output_dir / "coxph_run_diagnostics.csv").exists()
    assert not (output_dir / "coxph_leaderboard.json").exists()


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
        hpo={"enabled": True, "max_trials": 1},
    )

    fold_results = pd.read_csv(output_dir / "coxph_fold_results.csv")
    assert "parity_eligible" in fold_results.columns
    assert not fold_results["parity_eligible"].any()

    diagnostics = pd.read_csv(output_dir / "coxph_run_diagnostics.csv")
    assert not diagnostics.empty


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

    fold_results = pd.read_csv(output_dir / "coxph_fold_results.csv")
    for required in ("hpo_mode", "requested_max_trials", "requested_timeout_seconds", "realized_trial_count"):
        assert required in fold_results.columns
    assert set(fold_results["hpo_mode"]) == {"no_hpo", "hpo"}
    assert set(fold_results["requested_max_trials"]) == {7}
    assert set(fold_results["requested_timeout_seconds"]) == {123.0}
    assert fold_results["realized_trial_count"].max() == 4


# --- test_predictor_edge_cases.py ---


def test_predictor_save_writes_pickle_and_manifest(tmp_path: Path) -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")
    output_path = tmp_path / "predictor.pkl"

    saved_path = predictor.save(output_path)

    manifest_path = tmp_path / "predictor_manifest.json"
    assert saved_path == output_path
    assert output_path.exists()
    assert manifest_path.exists()


def test_predictor_load_round_trips_unfitted_predictor(tmp_path: Path) -> None:
    predictor = SurvivalPredictor(
        label_time="duration",
        label_event="observed",
        eval_metric="uno_c",
        enable_foundation_models=True,
    )
    output_path = tmp_path / "predictor.pkl"
    predictor.save(output_path)

    loaded = SurvivalPredictor.load(output_path)

    assert loaded.label_time == "duration"
    assert loaded.label_event == "observed"
    assert loaded.eval_metric == "uno_c"
    assert loaded.enable_foundation_models is True


def test_predictor_save_requires_artifact_dir_when_no_path_is_provided() -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")

    with pytest.raises(RuntimeError, match="No artifact directory is available"):
        predictor.save()


def test_predictor_load_rejects_unsupported_serialization_versions(tmp_path: Path) -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")
    output_path = tmp_path / "predictor.pkl"
    predictor.save(output_path)

    manifest_path = tmp_path / "predictor_manifest.json"
    manifest_path.write_text(
        json.dumps({"serialization_version": 99}, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Unsupported predictor serialization version 99"):
        SurvivalPredictor.load(output_path)


def test_predictor_fit_surfaces_when_all_candidate_models_fail(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0],
        }
    )

    monkeypatch.setattr(
        "survarena.api.predictor.resolve_preset",
        lambda *args, **kwargs: PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25),
    )
    monkeypatch.setattr("survarena.api.predictor.read_yaml", lambda path: {"default_params": {}})
    monkeypatch.setattr(
        "survarena.api.predictor.prepare_validation_fold_cache",
        lambda **kwargs: [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 0], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ],
    )

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        raise RuntimeError(f"{method_id} exploded")

    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )

    with pytest.raises(RuntimeError, match="All candidate models failed during fitting"):
        predictor.fit(frame, tuning_data=frame, dataset_name="toy")


def test_predictor_selection_sort_places_finite_scores_before_nan() -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")
    results = [
        PredictorModelResult("first_nan", float("nan"), {}, 0.0, 1, {}),
        PredictorModelResult("valid", 0.62, {}, 0.0, 1, {}),
        PredictorModelResult("lower", 0.51, {}, 0.0, 1, {}),
    ]

    best = max(results, key=predictor._selection_sort_key)
    ordered = sorted(results, key=predictor._selection_sort_key, reverse=True)

    assert best.method_id == "valid"
    assert [result.method_id for result in ordered] == ["valid", "lower", "first_nan"]


def test_compute_metric_bundle_safe_clips_inputs_after_training_support_error(monkeypatch) -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")
    calls: list[dict[str, np.ndarray | tuple[float, float, float]]] = []

    def fake_compute_survival_metrics(**kwargs) -> MetricBundle:
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("largest observed training event time point")
        return MetricBundle(uno_c=0.71, harrell_c=0.72, ibs=0.2, td_auc_25=0.73, td_auc_50=0.74, td_auc_75=0.75)

    monkeypatch.setattr("survarena.api.predictor.compute_survival_metrics", fake_compute_survival_metrics)

    metrics = predictor._compute_metric_bundle_safe(
        train_time=np.asarray([1.0, 3.0, 5.0]),
        train_event=np.asarray([1, 1, 0]),
        test_time=np.asarray([2.0, 4.0]),
        test_event=np.asarray([1, 0]),
        risk_scores=np.asarray([0.2, 0.4]),
        survival_probs=np.asarray([[0.9, 0.8, 0.7], [0.95, 0.85, 0.75]]),
        survival_times=np.asarray([1.0, 2.0, 4.0]),
    )

    assert metrics["uno_c"] == 0.71
    assert len(calls) == 2
    np.testing.assert_allclose(calls[1]["test_time"], np.asarray([2.0]))
    np.testing.assert_allclose(calls[1]["risk_scores"], np.asarray([0.2]))
    np.testing.assert_allclose(calls[1]["survival_times"], np.asarray([1.0, 2.0]))
    assert calls[1]["survival_probs"].shape == (1, 2)
    assert max(calls[1]["horizons"]) < 3.0


def test_compute_metric_bundle_safe_falls_back_to_harrell_only_when_no_rows_are_supported(monkeypatch) -> None:
    predictor = SurvivalPredictor(label_time="time", label_event="event")

    monkeypatch.setattr(
        "survarena.api.predictor.compute_survival_metrics",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("largest observed training event time point")),
    )
    monkeypatch.setattr("survarena.api.predictor.compute_harrell_c_index", lambda **kwargs: 0.61)

    metrics = predictor._compute_metric_bundle_safe(
        train_time=np.asarray([1.0, 2.0]),
        train_event=np.asarray([1, 1]),
        test_time=np.asarray([3.0, 4.0]),
        test_event=np.asarray([1, 0]),
        risk_scores=np.asarray([0.2, 0.4]),
        survival_probs=np.asarray([[0.9, 0.8], [0.95, 0.85]]),
        survival_times=np.asarray([3.0, 4.0]),
    )

    assert metrics["harrell_c"] == 0.61
    assert math.isnan(metrics["uno_c"])
    assert math.isnan(metrics["ibs"])
    assert math.isnan(metrics["td_auc_25"])


# --- test_predictor_registry.py ---


class MockSurvivalMethod(BaseSurvivalMethod):
    def __init__(self, bias: float = 0.0, seed: int | None = None) -> None:
        self.bias = float(bias)
        self.seed = seed

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "MockSurvivalMethod":
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return X.sum(axis=1).astype(float) + self.bias

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        risk = self.predict_risk(X)
        return np.exp(-np.outer(np.maximum(risk, 0.1), np.asarray(times, dtype=float)))


class MockFrameAwareMethod:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.fit_input_type: type | None = None
        self.fit_columns: list[str] | None = None
        self.risk_input_type: type | None = None
        self.risk_columns: list[str] | None = None
        self.survival_input_type: type | None = None
        self.survival_columns: list[str] | None = None

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "MockFrameAwareMethod":
        self.fit_input_type = type(X_train)
        self.fit_columns = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        return self

    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.risk_input_type = type(X)
        self.risk_columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return np.linspace(0.2, 1.0, num=len(X), dtype=float)

    def predict_survival(self, X: pd.DataFrame | np.ndarray, times: np.ndarray) -> np.ndarray:
        self.survival_input_type = type(X)
        self.survival_columns = list(X.columns) if isinstance(X, pd.DataFrame) else None
        risk = np.maximum(self.predict_risk(X), 0.1)
        return np.exp(-np.outer(risk, np.asarray(times, dtype=float)))


def test_fold_cache_metric_summary_uses_bundle_contract(monkeypatch) -> None:
    class BundleOnlyMockMethod(MockSurvivalMethod):
        bundle_calls = 0

        def predict_risk(self, X: np.ndarray) -> np.ndarray:
            raise AssertionError("fold metric summary should use predict_bundle")

        def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
            raise AssertionError("fold metric summary should use predict_bundle")

        def predict_bundle(self, X: np.ndarray, times: np.ndarray):
            BundleOnlyMockMethod.bundle_calls += 1
            risk = X.sum(axis=1).astype(float) + self.bias
            survival = np.exp(-np.outer(np.maximum(risk, 0.1), np.asarray(times, dtype=float)))
            return SimpleNamespace(risk=risk, survival=survival)

    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: BundleOnlyMockMethod)
    monkeypatch.setattr(
        SurvivalPredictor,
        "_compute_metric_bundle_safe",
        lambda self, **kwargs: {
            "uno_c": 0.7,
            "harrell_c": 0.7,
            "ibs": 0.2,
            "td_auc_25": 0.7,
            "td_auc_50": 0.7,
            "td_auc_75": 0.7,
        },
    )
    fold_cache = [
        {
            "X_train": np.asarray([[0.0], [1.0]], dtype=float),
            "X_val": np.asarray([[0.5], [1.5]], dtype=float),
            "time_train": np.asarray([1.0, 2.0], dtype=float),
            "event_train": np.asarray([1, 1], dtype=int),
            "time_val": np.asarray([1.5, 2.5], dtype=float),
            "event_val": np.asarray([1, 0], dtype=int),
        }
    ]

    summary = SurvivalPredictor(label_time="time", label_event="event")._fold_cache_metric_summary(
        method_id="mock_a",
        params={},
        fold_cache=fold_cache,
    )

    assert BundleOnlyMockMethod.bundle_calls == 1
    assert summary["validation_uno_c"] == 0.7


def test_predictor_tracks_multiple_fitted_models_and_roundtrips(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_inner_cv_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        X_train = np.asarray([[0.0], [1.0]], dtype=float)
        X_val = np.asarray([[0.5], [1.5]], dtype=float)
        time_train = np.asarray([1.0, 2.0], dtype=float)
        event_train = np.asarray([1, 1], dtype=int)
        time_val = np.asarray([1.5, 2.5], dtype=float)
        event_val = np.asarray([1, 0], dtype=int)
        return [
            {
                "X_train": X_train,
                "X_val": X_val,
                "time_train": time_train,
                "event_train": event_train,
                "time_val": time_val,
                "event_val": event_val,
            }
        ]

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        bias = 1.0 if method_id == "mock_a" else 2.0
        return {
            "best_params": {"bias": bias},
            "best_score": bias,
            "best_metric_rows": [
                {
                    "uno_c": bias,
                    "harrell_c": bias,
                    "ibs": 0.2,
                    "td_auc_25": bias,
                    "td_auc_50": bias,
                    "td_auc_75": bias,
                }
            ],
        }

    def fake_metric_bundle(self: SurvivalPredictor, **kwargs) -> dict[str, float]:
        score = float(np.mean(kwargs["risk_scores"]))
        return {
            "uno_c": score,
            "harrell_c": score,
            "ibs": 1.0 - min(score / 10.0, 0.9),
            "td_auc_25": score,
            "td_auc_50": score,
            "td_auc_75": score,
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_inner_cv_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_compute_metric_bundle_safe", fake_metric_bundle)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        retain_top_k_models=2,
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, test_data=frame, dataset_name="toy")

    assert predictor.best_method_id_ == "mock_b"
    assert set(predictor.model_names()) == {"mock_a", "mock_b"}

    best_risk = predictor.predict_risk(frame)
    best_predictions = predictor.predict_bundle(frame)
    alt_risk = predictor.predict_risk(frame, model="mock_a")
    assert not np.allclose(best_risk, alt_risk)

    summary = predictor.fit_summary()
    assert set(summary["trained_models"]) == {"mock_a", "mock_b"}
    assert summary["validation_strategy"] == "tuning_data"
    assert set(summary["per_model_test_metrics"]) == {"mock_a", "mock_b"}

    saved_path = tmp_path / "toy" / "predictor.pkl"
    assert saved_path.exists()

    loaded = SurvivalPredictor.load(saved_path)
    np.testing.assert_allclose(loaded.predict_risk(frame, model="mock_b"), best_risk)
    np.testing.assert_allclose(best_predictions.risk, best_risk)
    assert best_predictions.survival.shape[0] == len(frame)


def test_predictor_retains_only_the_best_model_by_default(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_inner_cv_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        return [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 1], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        bias = 1.0 if method_id == "mock_a" else 2.0
        return {
            "best_params": {"bias": bias},
            "best_score": bias,
            "best_metric_rows": [
                {
                    "uno_c": bias,
                    "harrell_c": bias,
                    "ibs": 0.2,
                    "td_auc_25": bias,
                    "td_auc_50": bias,
                    "td_auc_75": bias,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_inner_cv_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod}[method_id],
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy")

    assert predictor.best_method_id_ == "mock_b"
    assert predictor.model_names() == ["mock_b"]

    leaderboard = predictor.leaderboard().set_index("method_id")
    assert bool(leaderboard.loc["mock_b", "retained_for_inference"]) is True
    assert bool(leaderboard.loc["mock_a", "retained_for_inference"]) is False


def test_predictor_reuses_metric_rows_from_tuning(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_inner_cv_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        return [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 1], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_inner_cv_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id]
    )
    monkeypatch.setattr(
        SurvivalPredictor,
        "_fold_cache_metric_summary",
        lambda self, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected extra CV refit")),
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy")

    leaderboard = predictor.leaderboard()
    assert float(leaderboard.loc[0, "validation_harrell_c"]) == 0.8
    assert float(leaderboard.loc[0, "validation_td_auc_75"]) == 0.77


def test_predictor_uses_automatic_holdout_when_tuning_data_is_absent(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    fold_sizes: list[tuple[int, int]] = []

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.5)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_select_hyperparameters(*, fold_cache: list[dict[str, np.ndarray]], **kwargs) -> dict[str, object]:
        fold_sizes.append((int(fold_cache[0]["X_train"].shape[0]), int(fold_cache[0]["X_val"].shape[0])))
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id]
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", holdout_frac=0.5)

    summary = predictor.fit_summary()
    assert summary["validation_strategy"] == "auto_holdout"
    assert summary["validation_holdout_frac"] == 0.5
    assert summary["selection_train_rows"] == 3
    assert summary["validation_rows"] == 3
    assert fold_sizes == [(3, 3)]


def test_predictor_uses_bagged_oof_selection_when_num_bag_folds_enabled(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    fold_shapes: list[list[tuple[int, int]]] = []

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_select_hyperparameters(*, fold_cache: list[dict[str, np.ndarray]], **kwargs) -> dict[str, object]:
        fold_shapes.append([(int(fold["X_train"].shape[0]), int(fold["X_val"].shape[0])) for fold in fold_cache])
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
                for _ in fold_cache
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id]
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", num_bag_folds=3, num_bag_sets=2)

    summary = predictor.fit_summary()
    assert summary["validation_strategy"] == "bagged_oof"
    assert summary["num_bag_folds"] == 3
    assert summary["num_bag_sets"] == 2
    assert summary["selection_train_rows"] == 4
    assert summary["validation_rows"] == 12
    assert len(fold_shapes) == 1
    assert fold_shapes[0] == [(4, 2)] * 6


def test_predictor_bagged_models_average_fold_members_for_inference(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )
    fitted_member_ids: list[int] = []

    class AveragingMockMethod(MockSurvivalMethod):
        def fit(
            self,
            X_train: np.ndarray,
            time_train: np.ndarray,
            event_train: np.ndarray,
            X_val: np.ndarray | None = None,
            time_val: np.ndarray | None = None,
            event_val: np.ndarray | None = None,
        ) -> "AveragingMockMethod":
            self.member_id = len(fitted_member_ids) + 1
            fitted_member_ids.append(self.member_id)
            return self

        def predict_risk(self, X: np.ndarray) -> np.ndarray:
            return np.full(X.shape[0], float(self.member_id))

        def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
            return np.full((X.shape[0], len(times)), float(self.member_id))

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": AveragingMockMethod}[method_id]
    )
    monkeypatch.setattr(SurvivalPredictor, "_persist_artifacts", lambda self, dataset_name, results: None)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", num_bag_folds=3)

    summary = predictor.fit_summary()
    risk = predictor.predict_risk(frame)
    assert summary["validation_strategy"] == "bagged_oof"
    assert summary["trained_models"] == ["mock_a"]
    assert fitted_member_ids == [1, 2, 3]
    np.testing.assert_allclose(risk, np.full(len(frame), 2.0))


def test_predictor_bagged_model_round_trips(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {"bias": 1.0}}

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id]
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", num_bag_folds=2)

    saved_path = tmp_path / "toy" / "predictor.pkl"
    loaded = SurvivalPredictor.load(saved_path)

    np.testing.assert_allclose(loaded.predict_risk(frame), predictor.predict_risk(frame))


def test_predictor_num_bag_sets_requires_bagged_folds(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )

    with pytest.raises(ValueError, match="num_bag_sets > 1 requires num_bag_folds >= 2"):
        predictor.fit(frame, dataset_name="toy", num_bag_sets=2)


def test_predictor_refit_full_uses_tuning_data_for_final_training(tmp_path: Path, monkeypatch) -> None:
    train_frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    tuning_frame = pd.DataFrame(
        {
            "time": [5.0, 6.0],
            "event": [0, 1],
            "age": [59.0, 63.0],
            "stage": ["i", "iii"],
        }
    )
    fit_row_counts: list[int] = []

    class RecordingMockMethod(MockSurvivalMethod):
        def fit(
            self,
            X_train: np.ndarray,
            time_train: np.ndarray,
            event_train: np.ndarray,
            X_val: np.ndarray | None = None,
            time_val: np.ndarray | None = None,
            event_val: np.ndarray | None = None,
        ) -> "RecordingMockMethod":
            fit_row_counts.append(int(X_train.shape[0]))
            return self

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        return [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 1], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": RecordingMockMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_persist_artifacts", lambda self, dataset_name, results: None)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(train_frame, tuning_data=tuning_frame, dataset_name="toy", refit_full=True)

    summary = predictor.fit_summary()
    assert fit_row_counts == [6]
    assert summary["refit_full"] is True
    assert summary["final_train_rows"] == 6


def test_predictor_refit_full_false_keeps_explicit_tuning_rows_out_of_final_training(
    tmp_path: Path, monkeypatch
) -> None:
    train_frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    tuning_frame = pd.DataFrame(
        {
            "time": [5.0, 6.0],
            "event": [0, 1],
            "age": [59.0, 63.0],
            "stage": ["i", "iii"],
        }
    )
    fit_row_counts: list[int] = []

    class RecordingMockMethod(MockSurvivalMethod):
        def fit(
            self,
            X_train: np.ndarray,
            time_train: np.ndarray,
            event_train: np.ndarray,
            X_val: np.ndarray | None = None,
            time_val: np.ndarray | None = None,
            event_val: np.ndarray | None = None,
        ) -> "RecordingMockMethod":
            fit_row_counts.append(int(X_train.shape[0]))
            return self

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        return [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 1], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": RecordingMockMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_persist_artifacts", lambda self, dataset_name, results: None)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(train_frame, tuning_data=tuning_frame, dataset_name="toy", refit_full=False)

    summary = predictor.fit_summary()
    assert fit_row_counts == [4]
    assert summary["refit_full"] is False
    assert summary["final_train_rows"] == 4


def test_predictor_fit_level_autogluon_kwargs_are_normalized(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    recorded_timeout: list[float | None] = []

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a",), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        return [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 1], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        recorded_timeout.append(kwargs.get("method_cfg", {}).get("default_params", {}).get("time_limit"))
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class", lambda method_id: {"mock_a": MockSurvivalMethod}[method_id]
    )

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(
        frame,
        tuning_data=frame,
        dataset_name="toy",
        hyperparameter_tune_kwargs={"num_trials": 5, "timeout": 12.0},
    )

    summary = predictor.fit_summary()
    assert recorded_timeout == [None]
    assert summary["hyperparameter_tune_kwargs"] == {"num_trials": 5, "timeout_seconds": 12.0}


def test_predictor_time_limit_skips_candidates_when_budget_is_exhausted(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    assigned_budgets = iter([1.5, 0.0])
    selection_calls: list[str] = []

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        return [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 1], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        selection_calls.append(method_id)
        return {
            "best_params": {"bias": 1.0},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.7,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.75,
                    "td_auc_50": 0.76,
                    "td_auc_75": 0.77,
                }
            ],
        }

    def fake_next_method_time_limit(self: SurvivalPredictor, **kwargs) -> float | None:
        return next(assigned_budgets)

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_next_method_time_limit", fake_next_method_time_limit)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy", time_limit=3.0)

    summary = predictor.fit_summary()
    assert summary["time_limit_sec"] == 3.0
    assert summary["selection_time_budget_sec"] == pytest.approx(2.4)
    assert summary["trained_models"] == ["mock_a"]
    assert selection_calls == ["mock_a"]

    leaderboard = predictor.leaderboard().set_index("method_id")
    assert bool(leaderboard.loc["mock_a", "retained_for_inference"]) is True
    assert leaderboard.loc["mock_b", "status"] == "skipped"
    assert float(leaderboard.loc["mock_b", "time_limit_sec"]) == 0.0


def test_predictor_time_limit_prioritizes_refitting_the_best_model(tmp_path: Path, monkeypatch) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 1],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )
    assigned_budgets = iter([1.0, 1.0])

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=("mock_a", "mock_b"), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, np.ndarray]]:
        return [
            {
                "X_train": np.asarray([[0.0], [1.0]], dtype=float),
                "X_val": np.asarray([[0.5], [1.5]], dtype=float),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 1], dtype=int),
                "time_val": np.asarray([1.5, 2.5], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(*, method_id: str, **kwargs) -> dict[str, object]:
        bias = 1.0 if method_id == "mock_a" else 2.0
        return {
            "best_params": {"bias": bias},
            "best_score": bias,
            "best_metric_rows": [
                {
                    "uno_c": bias,
                    "harrell_c": bias,
                    "ibs": 0.2,
                    "td_auc_25": bias,
                    "td_auc_50": bias,
                    "td_auc_75": bias,
                }
            ],
        }

    def fake_next_method_time_limit(self: SurvivalPredictor, **kwargs) -> float | None:
        return next(assigned_budgets)

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr(
        "survarena.api.predictor._get_method_class",
        lambda method_id: {"mock_a": MockSurvivalMethod, "mock_b": MockSurvivalMethod}[method_id],
    )
    monkeypatch.setattr(SurvivalPredictor, "_next_method_time_limit", fake_next_method_time_limit)
    monkeypatch.setattr(SurvivalPredictor, "_remaining_fit_time", lambda self, fit_started_at, time_limit: 0.0)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="medium",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy", time_limit=3.0)

    assert predictor.best_method_id_ == "mock_b"
    assert predictor.model_names() == ["mock_b"]

    leaderboard = predictor.leaderboard().set_index("method_id")
    assert bool(leaderboard.loc["mock_b", "retained_for_inference"]) is True
    assert bool(leaderboard.loc["mock_a", "retained_for_inference"]) is False


@pytest.mark.parametrize("method_id", ["catboost_cox", "catboost_survival_aft"])
def test_predictor_preserves_native_categorical_frames_for_catboost_method(
    tmp_path: Path,
    monkeypatch,
    method_id: str,
) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0],
            "event": [1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0],
            "stage": ["i", "ii", "ii", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=(method_id,), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_prepare_validation_fold_cache(**kwargs) -> list[dict[str, object]]:
        return [
            {
                "X_train": pd.DataFrame({"age": [61.0, 57.0], "stage": ["i", "ii"]}),
                "X_val": pd.DataFrame({"age": [70.0, 66.0], "stage": ["ii", "iii"]}),
                "time_train": np.asarray([1.0, 2.0], dtype=float),
                "event_train": np.asarray([1, 0], dtype=int),
                "time_val": np.asarray([3.0, 4.0], dtype=float),
                "event_val": np.asarray([1, 0], dtype=int),
            }
        ]

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.8,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.8,
                    "td_auc_50": 0.8,
                    "td_auc_75": 0.8,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.prepare_validation_fold_cache", fake_prepare_validation_fold_cache)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: MockFrameAwareMethod)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="fast",
        save_path=tmp_path,
    )
    predictor.fit(frame, tuning_data=frame, dataset_name="toy")

    fitted = predictor.fitted_models_[method_id]
    assert fitted.fit_input_type is pd.DataFrame
    assert fitted.fit_columns == ["age", "stage"]

    predictor.predict_risk(frame)
    predictor.predict_survival(frame)

    assert fitted.risk_input_type is pd.DataFrame
    assert fitted.risk_columns == ["age", "stage"]
    assert fitted.survival_input_type is pd.DataFrame
    assert fitted.survival_columns == ["age", "stage"]


@pytest.mark.parametrize("method_id", ["catboost_cox", "catboost_survival_aft"])
def test_bagged_predictor_preserves_native_categorical_frames_for_catboost_method(
    tmp_path: Path,
    monkeypatch,
    method_id: str,
) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "event": [1, 0, 1, 0, 1, 0],
            "age": [61.0, 57.0, 70.0, 66.0, 59.0, 63.0],
            "stage": ["i", "ii", "ii", "iii", "i", "iii"],
        }
    )

    def fake_resolve_preset(*args, **kwargs) -> PresetConfig:
        return PresetConfig(name="test", method_ids=(method_id,), holdout_frac=0.25)

    def fake_read_yaml(path: Path) -> dict[str, object]:
        return {"default_params": {}}

    def fake_select_hyperparameters(**kwargs) -> dict[str, object]:
        return {
            "best_params": {},
            "best_score": 0.8,
            "best_metric_rows": [
                {
                    "uno_c": 0.8,
                    "harrell_c": 0.8,
                    "ibs": 0.2,
                    "td_auc_25": 0.8,
                    "td_auc_50": 0.8,
                    "td_auc_75": 0.8,
                }
            ],
        }

    monkeypatch.setattr("survarena.api.predictor.resolve_preset", fake_resolve_preset)
    monkeypatch.setattr("survarena.api.predictor.read_yaml", fake_read_yaml)
    monkeypatch.setattr("survarena.api.predictor.select_hyperparameters", fake_select_hyperparameters)
    monkeypatch.setattr("survarena.api.predictor._get_method_class", lambda method_id: MockFrameAwareMethod)

    predictor = SurvivalPredictor(
        label_time="time",
        label_event="event",
        presets="fast",
        save_path=tmp_path,
    )
    predictor.fit(frame, dataset_name="toy", num_bag_folds=2)

    fitted = predictor.fitted_models_[method_id]
    members = fitted.members
    assert len(members) == 2
    assert all(member.model.fit_input_type is pd.DataFrame for member in members)
    assert all(member.model.fit_columns == ["age", "stage"] for member in members)

    predictor.predict_risk(frame)
    predictor.predict_survival(frame)

    assert all(member.model.risk_input_type is pd.DataFrame for member in members)
    assert all(member.model.risk_columns == ["age", "stage"] for member in members)
    assert all(member.model.survival_input_type is pd.DataFrame for member in members)
    assert all(member.model.survival_columns == ["age", "stage"] for member in members)


def test_public_package_exports_survival_predictor() -> None:
    survarena = importlib.import_module("survarena")

    assert survarena.SurvivalPredictor is SurvivalPredictor
    assert callable(survarena.compare_survival_models)
