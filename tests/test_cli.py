from __future__ import annotations

import sys

from survarena import cli
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
