from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from survarena.benchmark.runner import validate_benchmark_profile_contract
from survarena.config import read_yaml
from survarena.data.splitters import load_or_create_splits


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


def _event_labels() -> np.ndarray:
    return np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)


def test_manifest_mismatch_raises_by_default(tmp_path) -> None:
    task_id = "determinism_mismatch_default"
    event = _event_labels()
    load_or_create_splits(
        root=tmp_path,
        task_id=task_id,
        split_strategy="repeated_nested_cv",
        n_samples=event.size,
        event=event,
        seeds=[11],
        outer_folds=2,
        outer_repeats=1,
    )

    with pytest.raises(ValueError, match="manifest payload mismatch"):
        load_or_create_splits(
            root=tmp_path,
            task_id=task_id,
            split_strategy="repeated_nested_cv",
            n_samples=event.size,
            event=event,
            seeds=[22],
            outer_folds=2,
            outer_repeats=1,
        )


def test_manifest_mismatch_allows_explicit_regenerate(tmp_path) -> None:
    task_id = "determinism_mismatch_regenerate"
    event = _event_labels()
    load_or_create_splits(
        root=tmp_path,
        task_id=task_id,
        split_strategy="repeated_nested_cv",
        n_samples=event.size,
        event=event,
        seeds=[11],
        outer_folds=2,
        outer_repeats=1,
    )

    regenerated = load_or_create_splits(
        root=tmp_path,
        task_id=task_id,
        split_strategy="repeated_nested_cv",
        n_samples=event.size,
        event=event,
        seeds=[22],
        outer_folds=2,
        outer_repeats=1,
        regenerate_on_mismatch=True,
    )

    assert regenerated
    assert all(split.seed == 22 for split in regenerated)


def test_matching_manifest_reuses_existing_splits(tmp_path) -> None:
    task_id = "determinism_manifest_reuse"
    event = _event_labels()
    created = load_or_create_splits(
        root=tmp_path,
        task_id=task_id,
        split_strategy="repeated_nested_cv",
        n_samples=event.size,
        event=event,
        seeds=[11],
        outer_folds=2,
        outer_repeats=1,
    )

    reused = load_or_create_splits(
        root=tmp_path,
        task_id=task_id,
        split_strategy="repeated_nested_cv",
        n_samples=event.size,
        event=event,
        seeds=[11],
        outer_folds=2,
        outer_repeats=1,
    )

    assert [split.split_id for split in reused] == [split.split_id for split in created]


def test_profile_contract_configs_use_canonical_tier_intent() -> None:
    smoke_cfg = read_yaml(Path("configs/benchmark/smoke_all_models_no_hpo.yaml"))
    standard_cfg = read_yaml(Path("configs/benchmark/standard_v1.yaml"))
    manuscript_cfg = read_yaml(Path("configs/benchmark/manuscript_v1.yaml"))

    assert smoke_cfg["profile"] == "smoke"
    assert standard_cfg["profile"] == "standard"
    assert manuscript_cfg.get("profile", "manuscript") == "manuscript"


def test_event_fingerprint_rejects_non_binary_labels() -> None:
    from survarena.data.splitters import _event_fingerprint

    with pytest.raises(ValueError, match="binary"):
        _event_fingerprint(np.array([0, 1, 2], dtype=int))


def test_unknown_method_rejected_before_read_yaml(tmp_path, monkeypatch) -> None:
    from survarena.benchmark import runner as runner_mod

    read_calls: list[int] = []

    def _tracking_read(*_a, **_k) -> dict:
        read_calls.append(1)
        return {"method_id": "coxph", "default_params": {}}

    monkeypatch.setattr(runner_mod, "read_yaml", _tracking_read)
    monkeypatch.setattr(runner_mod, "registered_method_ids", lambda: {"coxph", "aft"})
    cfg: dict = {
        "benchmark_id": "t",
        "profile": "smoke",
        "datasets": ["d"],
        "methods": ["not_registered_xyz"],
        "split_strategy": "repeated_nested_cv",
        "seeds": [1],
        "outer_folds": 3,
        "outer_repeats": 1,
        "inner_folds": 2,
    }
    with pytest.raises(ValueError, match="Unknown method_id"):
        runner_mod.run_benchmark(
            repo_root=tmp_path,
            benchmark_cfg=cfg,
            output_dir=tmp_path / "out",
            dry_run=False,
        )
    assert read_calls == []
