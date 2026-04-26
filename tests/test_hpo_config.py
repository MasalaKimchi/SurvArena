from __future__ import annotations

from typing import Any

from survarena.benchmark import tuning


def test_parse_hpo_config_defaults_disabled_without_search_space() -> None:
    cfg = tuning._parse_hpo_config({"default_params": {"a": 1}}, {"enabled": True})
    assert cfg["enabled"] is False


def test_parse_hpo_config_enables_with_space() -> None:
    cfg = tuning._parse_hpo_config(
        {"search_space": {"alpha": {"type": "float", "low": 0.1, "high": 0.9}}},
        {"enabled": True, "max_trials": 5, "sampler": "random"},
    )
    assert cfg["enabled"] is True
    assert cfg["max_trials"] == 5
    assert cfg["sampler"] == "random"


def test_parse_hpo_config_normalizes_uniform_budget_fields() -> None:
    cfg = tuning._parse_hpo_config(
        {"search_space": {"alpha": {"type": "float", "low": 0.1, "high": 0.9}}},
        {"enabled": True, "max_trials": 9, "timeout_seconds": 120, "sampler": "TPE", "pruner": "MEDIAN"},
    )

    assert cfg["enabled"] is True
    assert cfg["max_trials"] == 9
    assert cfg["timeout_seconds"] == 120.0
    assert cfg["sampler"] == "tpe"
    assert cfg["pruner"] == "median"
    assert cfg["n_startup_trials"] == 8


def test_parse_hpo_config_keeps_trials_and_timeout_together() -> None:
    cfg = tuning._parse_hpo_config(
        {"search_space": {"alpha": {"type": "float", "low": 0.1, "high": 0.9}}},
        {
            "enabled": True,
            "max_trials": 0,
            "timeout_seconds": -3,
            "sampler": "RANDOM",
            "pruner": "NOP",
            "n_startup_trials": 0,
        },
    )

    assert cfg["enabled"] is True
    assert cfg["max_trials"] == 1
    assert cfg["timeout_seconds"] == 0.0
    assert cfg["sampler"] == "random"
    assert cfg["pruner"] == "nop"
    assert cfg["n_startup_trials"] == 1


def test_select_hyperparameters_emits_requested_and_realized_budget_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        tuning,
        "_inner_cv_evaluate",
        lambda **_kwargs: {"primary_score": 0.5},
    )

    result = tuning.select_hyperparameters(
        method_id="coxph",
        method_cfg={"default_params": {"alpha": 0.1}},
        fold_cache=[{"dummy": True}],
        primary_metric="harrell_c",
        seed=11,
        hpo_config={
            "enabled": False,
            "max_trials": 9,
            "timeout_seconds": 120,
            "sampler": "random",
            "pruner": "nop",
            "n_startup_trials": 4,
        },
    )

    metadata: dict[str, Any] = result["hpo_metadata"]
    assert metadata["requested_max_trials"] == 9
    assert metadata["requested_timeout_seconds"] == 120.0
    assert metadata["requested_sampler"] == "random"
    assert metadata["requested_pruner"] == "nop"
    assert metadata["realized_trial_count"] == 0
    assert metadata["trial_count"] == 0


def test_select_hyperparameters_can_skip_default_inner_cv_when_disabled(monkeypatch) -> None:
    def _raise_inner_cv_evaluate(**_kwargs):
        raise AssertionError("default inner CV should be skipped")

    monkeypatch.setattr(tuning, "_inner_cv_evaluate", _raise_inner_cv_evaluate)

    result = tuning.select_hyperparameters(
        method_id="coxph",
        method_cfg={"default_params": {"alpha": 0.1}},
        fold_cache=[],
        primary_metric="harrell_c",
        seed=11,
        hpo_config={"enabled": False, "max_trials": 9, "timeout_seconds": 120},
        evaluate_defaults_when_disabled=False,
    )

    assert result["best_params"] == {"alpha": 0.1}
    assert result["best_score"] != result["best_score"]
    assert result["best_metric_rows"] is None
    assert result["hpo_metadata"]["status"] == "disabled"


def test_select_hyperparameters_reuses_best_trial_score_without_metric_rows(monkeypatch) -> None:
    import pytest

    pytest.importorskip("optuna")

    calls = {"count": 0}

    def _fake_inner_cv_evaluate(**kwargs):
        calls["count"] += 1
        alpha = kwargs["params"]["alpha"]
        return {"primary_score": float(alpha)}

    monkeypatch.setattr(tuning, "_inner_cv_evaluate", _fake_inner_cv_evaluate)

    result = tuning.select_hyperparameters(
        method_id="coxph",
        method_cfg={
            "default_params": {"alpha": 0.1},
            "search_space": {"alpha": {"type": "categorical", "choices": [0.2]}},
        },
        fold_cache=[{"dummy": True}],
        primary_metric="harrell_c",
        seed=11,
        hpo_config={"enabled": True, "max_trials": 1, "sampler": "random", "pruner": "nop"},
    )

    assert calls["count"] == 2
    assert result["best_score"] == 0.2
    assert result["best_metric_rows"] is None


def test_select_hyperparameters_falls_back_to_defaults_when_trial_fails(monkeypatch) -> None:
    import pytest

    pytest.importorskip("optuna")

    def _fake_inner_cv_evaluate(**kwargs):
        if kwargs["params"]["alpha"] == 0.2:
            raise RuntimeError("fit failed")
        return {"primary_score": 0.5}

    monkeypatch.setattr(tuning, "_inner_cv_evaluate", _fake_inner_cv_evaluate)

    result = tuning.select_hyperparameters(
        method_id="coxph",
        method_cfg={
            "default_params": {"alpha": 0.1},
            "search_space": {"alpha": {"type": "categorical", "choices": [0.2]}},
        },
        fold_cache=[{"dummy": True}],
        primary_metric="harrell_c",
        seed=11,
        hpo_config={"enabled": True, "max_trials": 1, "sampler": "random", "pruner": "nop"},
    )

    assert result["best_params"] == {"alpha": 0.1}
    assert result["best_score"] == 0.5
    assert result["hpo_trials"][0]["user_attrs"]["failure_type"] == "RuntimeError"
