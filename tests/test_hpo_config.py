from __future__ import annotations

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
