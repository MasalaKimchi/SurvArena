from __future__ import annotations

import pytest

from survarena.benchmark.runner import validate_benchmark_profile_contract


def _base_cfg(profile: str) -> dict[str, object]:
    return {
        "benchmark_id": "unit_test",
        "profile": profile,
        "split_strategy": "repeated_nested_cv",
        "seeds": [11, 22, 33],
        "outer_folds": 5,
        "outer_repeats": 3,
        "inner_folds": 3,
    }


def test_profile_contract_smoke_passes() -> None:
    cfg = _base_cfg("smoke")
    cfg["outer_repeats"] = 1

    validate_benchmark_profile_contract(cfg)


def test_profile_contract_standard_passes() -> None:
    cfg = _base_cfg("standard")

    validate_benchmark_profile_contract(cfg)


def test_profile_contract_rejects_invalid_profile() -> None:
    cfg = _base_cfg("research")

    with pytest.raises(ValueError, match="Invalid profile"):
        validate_benchmark_profile_contract(cfg)


def test_profile_contract_rejects_missing_deterministic_keys() -> None:
    cfg = _base_cfg("manuscript")
    del cfg["split_strategy"]
    del cfg["seeds"]

    with pytest.raises(ValueError, match="Missing required deterministic fields"):
        validate_benchmark_profile_contract(cfg)


def test_profile_contract_error_messages_are_actionable() -> None:
    with pytest.raises(ValueError, match="Allowed profiles: smoke, standard, manuscript"):
        validate_benchmark_profile_contract(_base_cfg("research"))

    cfg = _base_cfg("smoke")
    del cfg["split_strategy"]
    del cfg["seeds"]
    with pytest.raises(ValueError, match="seeds, split_strategy"):
        validate_benchmark_profile_contract(cfg)
