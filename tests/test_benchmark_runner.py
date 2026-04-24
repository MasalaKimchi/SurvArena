from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from survarena.benchmark import runner
from survarena.benchmark.runner import validate_benchmark_profile_contract
from survarena.config import read_yaml
from survarena.data.splitters import SplitDefinition, load_or_create_splits

# --- Profile contract & split determinism ---


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


# --- Resume & retry (mocked runner) ---


def _resume_benchmark_cfg(benchmark_id: str = "resume_test") -> dict[str, object]:
    return {
        "benchmark_id": benchmark_id,
        "profile": "smoke",
        "datasets": ["toy_dataset"],
        "methods": ["coxph"],
        "split_strategy": "repeated_nested_cv",
        "outer_folds": 2,
        "outer_repeats": 1,
        "inner_folds": 2,
        "seeds": [11],
        "primary_metric": "uno_c",
    }


def _fake_dataset() -> SimpleNamespace:
    frame = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [0.0, 1.0, 0.0, 1.0]})
    return SimpleNamespace(
        X=frame,
        time=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
        event=np.asarray([1, 0, 1, 0], dtype=int),
        metadata=SimpleNamespace(feature_types={"x1": "numerical", "x2": "categorical"}),
    )


def _single_split(seed: int = 11) -> list[SplitDefinition]:
    return [
        SplitDefinition(
            split_id="fixed_split_0",
            seed=seed,
            repeat=0,
            fold=0,
            train_idx=np.asarray([0, 1], dtype=int),
            test_idx=np.asarray([2, 3], dtype=int),
            val_idx=np.asarray([1], dtype=int),
        )
    ]


def _resume_record(status: str = "success") -> dict[str, object]:
    return {
        "run_payload": {
            "manifest": {"run_id": "toy_run"},
            "metrics": {"status": status, "uno_c": 0.71},
            "failure": None if status == "success" else {"traceback": "boom"},
        },
        "benchmark_id": "resume_test",
        "dataset_id": "toy_dataset__base",
        "method_id": "coxph",
        "split_id": "fixed_split_0__base",
        "seed": 11,
        "primary_metric": "uno_c",
        "validation_score": 0.7,
        "uno_c": 0.71,
        "harrell_c": 0.7,
        "ibs": 0.2,
        "td_auc_25": 0.68,
        "td_auc_50": 0.69,
        "td_auc_75": 0.7,
        "tuning_time_sec": 0.01,
        "runtime_sec": 0.02,
        "fit_time_sec": 0.01,
        "infer_time_sec": 0.01,
        "peak_memory_mb": 64.0,
        "status": status,
    }


def _install_common_monkeypatches(monkeypatch, call_counter: dict[str, int], *, status: str = "success") -> None:
    monkeypatch.setattr("survarena.data.loaders.load_dataset", lambda *_args, **_kwargs: _fake_dataset())
    monkeypatch.setattr("survarena.data.splitters.load_or_create_splits", lambda **_kwargs: _single_split())
    monkeypatch.setattr(
        "survarena.data.robustness.resolve_robustness_tracks",
        lambda *_args, **_kwargs: [SimpleNamespace(track_id="base")],
    )
    monkeypatch.setattr("survarena.data.robustness.apply_robustness_track", lambda X, **_kwargs: X)
    monkeypatch.setattr("survarena.data.robustness.apply_label_noise", lambda event, **_kwargs: event)
    monkeypatch.setattr("survarena.benchmark.runner.read_yaml", lambda *_args, **_kwargs: {"method_id": "coxph"})

    def _fake_evaluate_split(**_kwargs):
        call_counter["count"] += 1
        return _resume_record(status=status)

    monkeypatch.setattr("survarena.benchmark.runner.evaluate_split", _fake_evaluate_split)
    monkeypatch.setattr("survarena.logging.export.create_experiment_dir", lambda _root: Path("."))
    monkeypatch.setattr(
        "survarena.logging.export.export_fold_results",
        lambda *_args, **_kwargs: pd.DataFrame([{"dataset_id": "toy_dataset__base", "method_id": "coxph"}]),
    )
    monkeypatch.setattr(
        "survarena.logging.export.export_seed_summary",
        lambda *_args, **_kwargs: pd.DataFrame([{"dataset_id": "toy_dataset__base", "method_id": "coxph"}]),
    )
    monkeypatch.setattr("survarena.logging.export.export_overall_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "survarena.logging.export.export_leaderboard",
        lambda *_args, **_kwargs: pd.DataFrame([{"dataset_id": "toy_dataset__base", "method_id": "coxph"}]),
    )
    monkeypatch.setattr("survarena.logging.export.export_manuscript_comparison", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("survarena.logging.export.export_dataset_curation_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("survarena.logging.export.export_run_ledger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("survarena.logging.export.export_hpo_trials", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("survarena.logging.tracker.write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("survarena.benchmark.runner.registered_method_ids", lambda: {"coxph"})


def _install_retry_monkeypatches(
    monkeypatch,
    *,
    statuses: list[str],
    captured_run_records: list[dict[str, object]],
) -> dict[str, int]:
    calls = {"count": 0}
    _install_common_monkeypatches(monkeypatch, calls)

    state = {"idx": 0}

    def _fake_evaluate_split(**_kwargs):
        status = statuses[min(state["idx"], len(statuses) - 1)]
        state["idx"] += 1
        calls["count"] += 1
        return _resume_record(status=status)

    def _capture_run_ledger(_root, run_records, **_kwargs):
        captured_run_records.extend(run_records)

    monkeypatch.setattr("survarena.benchmark.runner.evaluate_split", _fake_evaluate_split)
    monkeypatch.setattr("survarena.logging.export.export_run_ledger", _capture_run_ledger)
    return calls


def test_exec04_resume_preserves_successful_outputs(tmp_path: Path, monkeypatch) -> None:
    fold_results = pd.DataFrame(
        [
            {
                "dataset_id": "toy_dataset__base",
                "method_id": "coxph",
                "split_id": "fixed_split_0__base",
                "seed": 11,
                "status": "success",
                "uno_c": 0.77,
            }
        ]
    )
    fold_results.to_csv(tmp_path / "resume_test_fold_results.csv", index=False)

    calls = {"count": 0}
    _install_common_monkeypatches(monkeypatch, calls)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0)

    assert calls["count"] == 0


def test_exec04_resume_reruns_incomplete_success_outputs(tmp_path: Path, monkeypatch) -> None:
    fold_results = pd.DataFrame(
        [
            {
                "dataset_id": "toy_dataset__base",
                "method_id": "coxph",
                "split_id": "fixed_split_0__base",
                "seed": 11,
                "status": "success",
                "uno_c": np.nan,
            }
        ]
    )
    fold_results.to_csv(tmp_path / "resume_test_fold_results.csv", index=False)

    calls = {"count": 0}
    _install_common_monkeypatches(monkeypatch, calls)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0)

    # Dual-mode governance reruns both no-HPO and HPO when legacy rows are incomplete.
    assert calls["count"] == 2


def test_exec04_resume_ignores_non_success_completed_keys(tmp_path: Path, monkeypatch) -> None:
    fold_results = pd.DataFrame(
        [
            {
                "dataset_id": "toy_dataset__base",
                "method_id": "coxph",
                "split_id": "fixed_split_0__base",
                "seed": 11,
                "status": "failed",
                "uno_c": np.nan,
            }
        ]
    )
    fold_results.to_csv(tmp_path / "resume_test_fold_results.csv", index=False)

    calls = {"count": 0}
    _install_common_monkeypatches(monkeypatch, calls)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0)

    assert calls["count"] == 2


def test_exec04_retry_budget_caps_failed_rows(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    calls = _install_retry_monkeypatches(monkeypatch, statuses=["failed", "failed", "failed"], captured_run_records=run_records)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=1)

    assert calls["count"] == 4
    assert len(run_records) == 4
    assert run_records[-1]["metrics"]["status"] == "failed"


def test_exec04_failure_records_include_attempt_metadata(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    _install_retry_monkeypatches(monkeypatch, statuses=["failed", "failed"], captured_run_records=run_records)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=1)

    assert len(run_records) == 4
    assert [record["retry_attempt"] for record in run_records] == [0, 1, 0, 1]
    assert [record["status"] for record in run_records] == ["failed", "failed", "failed", "failed"]
    assert all(record["failure"] is not None for record in run_records)


def test_exec04_successful_retry_keeps_failed_attempt_evidence(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    calls = _install_retry_monkeypatches(monkeypatch, statuses=["failed", "success"], captured_run_records=run_records)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=2)

    assert calls["count"] == 3
    assert len(run_records) == 3
    assert run_records[0]["status"] == "failed"
    assert run_records[0]["failure"] is not None
    assert run_records[1]["status"] == "success"
    assert run_records[2]["status"] == "success"
