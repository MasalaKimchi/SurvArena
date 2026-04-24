from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from survarena.benchmark import runner
from survarena.data.splitters import SplitDefinition


def _benchmark_cfg(benchmark_id: str = "resume_test") -> dict[str, object]:
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


def _record(status: str = "success") -> dict[str, object]:
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
        return _record(status=status)

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
        return _record(status=status)

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

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0)

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

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0)

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

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0)

    assert calls["count"] == 2


def test_exec04_retry_budget_caps_failed_rows(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    calls = _install_retry_monkeypatches(monkeypatch, statuses=["failed", "failed", "failed"], captured_run_records=run_records)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=1)

    assert calls["count"] == 4
    assert len(run_records) == 4
    assert run_records[-1]["metrics"]["status"] == "failed"


def test_exec04_failure_records_include_attempt_metadata(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    _install_retry_monkeypatches(monkeypatch, statuses=["failed", "failed"], captured_run_records=run_records)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=1)

    assert len(run_records) == 4
    assert [record["retry_attempt"] for record in run_records] == [0, 1, 0, 1]
    assert [record["status"] for record in run_records] == ["failed", "failed", "failed", "failed"]
    assert all(record["failure"] is not None for record in run_records)


def test_exec04_successful_retry_keeps_failed_attempt_evidence(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    calls = _install_retry_monkeypatches(monkeypatch, statuses=["failed", "success"], captured_run_records=run_records)

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=2)

    assert calls["count"] == 3
    assert len(run_records) == 3
    assert run_records[0]["status"] == "failed"
    assert run_records[0]["failure"] is not None
    assert run_records[1]["status"] == "success"
    assert run_records[2]["status"] == "success"
