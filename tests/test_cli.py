from __future__ import annotations

import sys

import pandas as pd

from survarena import cli
from survarena.benchmark import overview
from survarena.methods.foundation.readiness import FoundationRuntimeStatus


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
            }
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
