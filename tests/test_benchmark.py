from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from survarena.benchmark import runner
from survarena.benchmark.runner import validate_benchmark_profile_contract
from survarena.config import read_yaml
from survarena.data.splitters import SplitDefinition, load_or_create_splits
from survarena.logging.export import export_coverage_matrix, export_hpo_budget_summary
from survarena.logging.export import export_run_diagnostics, export_runtime_failure_summary


# --- test_benchmark_runner.py ---

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


def test_profile_contract_manuscript_passes() -> None:
    validate_benchmark_profile_contract(_base_cfg("manuscript"))


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
    with pytest.raises(ValueError, match="Allowed profiles: manuscript"):
        validate_benchmark_profile_contract(_base_cfg("research"))

    cfg = _base_cfg("manuscript")
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
    manuscript_cfg = read_yaml(Path("configs/benchmark/manuscript_v1.yaml"))
    hpo_cfg = read_yaml(Path("configs/benchmark/manuscript_hpo_v1.yaml"))
    genomics_cfg = read_yaml(Path("configs/benchmark/manuscript_genomics_v1.yaml"))

    assert manuscript_cfg["profile"] == "manuscript"
    assert manuscript_cfg["comparison_modes"] == ["no_hpo"]
    assert manuscript_cfg["exports"]["manuscript_artifact_layout"] == "compact"
    assert "tabpfn_survival" in manuscript_cfg["methods"]
    assert "mitra_survival_frozen" not in manuscript_cfg["methods"]
    assert {"tabicl_survival", "tabm_survival", "tabdpt_survival", "realtabpfn_survival"}.issubset(
        manuscript_cfg["methods"]
    )
    assert hpo_cfg["profile"] == "manuscript"
    assert hpo_cfg["comparison_modes"] == ["hpo"]
    assert hpo_cfg["hpo"]["enabled"] is True
    assert hpo_cfg["datasets"] == manuscript_cfg["datasets"]
    assert hpo_cfg["methods"] == manuscript_cfg["methods"]
    assert genomics_cfg["profile"] == manuscript_cfg["profile"]
    assert genomics_cfg["comparison_modes"] == manuscript_cfg["comparison_modes"]
    assert genomics_cfg["hpo"] == manuscript_cfg["hpo"]
    assert genomics_cfg["methods"] == manuscript_cfg["methods"]
    assert genomics_cfg["datasets"] == [
        "tcga_brca_xena",
        "tcga_luad_xena",
        "tcga_kirc_xena",
        "tcga_skcm_xena",
        "tcga_ov_xena",
    ]


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
        "profile": "manuscript",
        "datasets": ["d"],
        "methods": ["not_registered_xyz"],
        "split_strategy": "repeated_nested_cv",
        "seeds": [1, 2, 3],
        "outer_folds": 3,
        "outer_repeats": 3,
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
        "profile": "manuscript",
        "datasets": ["toy_dataset"],
        "methods": ["coxph"],
        "split_strategy": "repeated_nested_cv",
        "outer_folds": 3,
        "outer_repeats": 3,
        "inner_folds": 2,
        "seeds": [11, 22, 33],
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


@pytest.mark.parametrize(
    ("method_id", "failure_message"),
    [
        ("coxph", "classical fit failed"),
        ("rsf", "tree fit failed"),
        ("xgboost_cox", "boosting fit failed"),
        ("deepsurv", "deep fit failed"),
        ("mitra_survival_frozen", "foundation AutoGluon fit failed"),
        ("tabpfn_survival", "Dependency 'tabpfn' is not installed."),
    ],
)
def test_evaluate_split_failure_payload_covers_major_method_families(
    method_id: str,
    failure_message: str,
    monkeypatch,
) -> None:
    class FailingMethod:
        def __init__(self, **_params) -> None:
            return None

        def fit(self, *_args, **_kwargs) -> None:
            raise RuntimeError(failure_message)

    split = _single_split()[0]
    dataset = _fake_dataset()
    monkeypatch.setattr(runner, "get_method_class", lambda _method_id: FailingMethod)
    monkeypatch.setattr(runner, "peak_process_memory_mb", lambda: 64.0, raising=False)

    record = runner.evaluate_split(
        benchmark_id="failure_payload",
        dataset_id="toy_dataset",
        method_id=method_id,
        split=split,
        X=dataset.X,
        time=dataset.time,
        event=dataset.event,
        method_cfg={"method_id": method_id, "default_params": {}, "search_space": {}},
        inner_folds=2,
        timeout_seconds=None,
        primary_metric="uno_c",
        horizons_quantiles=(0.25, 0.5, 0.75),
        decision_thresholds=(0.2,),
        benchmark_cfg_hash="cfg-hash",
        hpo_cfg={"enabled": False},
    )

    run_payload = record["run_payload"]
    manifest = run_payload["manifest"]
    metrics = run_payload["metrics"]

    assert record["status"] == "failed"
    assert record["method_id"] == method_id
    assert record["dataset_id"] == "toy_dataset"
    assert record["split_id"] == split.split_id
    assert record["seed"] == split.seed
    assert record["failure_type"] == "RuntimeError"
    assert record["exception_message"] == failure_message
    assert np.isnan(record["uno_c"])
    assert np.isnan(record["harrell_c"])
    assert np.isnan(record["ibs"])
    assert manifest["status"] == "failed"
    assert manifest["method_id"] == method_id
    assert manifest["dataset_id"] == "toy_dataset"
    assert manifest["split_id"] == split.split_id
    assert manifest["seed"] == split.seed
    assert manifest["notes"] == failure_message
    assert metrics["status"] == "failed"
    assert metrics["failure_type"] == "RuntimeError"
    assert metrics["exception_message"] == failure_message
    assert "RuntimeError" in run_payload["failure"]["traceback"]
    assert failure_message in run_payload["failure"]["traceback"]


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
        "survarena.logging.export.export_leaderboard",
        lambda *_args, **_kwargs: pd.DataFrame([{"dataset_id": "toy_dataset__base", "method_id": "coxph"}]),
    )
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

    def _capture_run_diagnostics(_root, *, fold_results, **_kwargs):
        captured_run_records.extend(fold_results.to_dict(orient="records"))

    monkeypatch.setattr("survarena.benchmark.runner.evaluate_split", _fake_evaluate_split)
    monkeypatch.setattr(
        "survarena.logging.export.export_fold_results",
        lambda _root, records, **_kwargs: pd.DataFrame(records),
    )
    monkeypatch.setattr("survarena.logging.export.export_run_diagnostics", _capture_run_diagnostics)
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
    fold_results.to_csv(tmp_path / "coxph_fold_results.csv", index=False)

    calls = {"count": 0}
    _install_common_monkeypatches(monkeypatch, calls)

    runner.run_benchmark(
        repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0
    )

    assert calls["count"] == 0


def test_resume_export_merges_existing_and_new_fold_rows(tmp_path: Path) -> None:
    existing = pd.DataFrame(
        [
            {
                "dataset_id": "toy_dataset__base",
                "method_id": "coxph",
                "split_id": "fixed_split_0__base",
                "seed": 11,
                "hpo_mode": "no_hpo",
                "status": "success",
                "uno_c": 0.77,
            }
        ]
    )
    fold_results_path = tmp_path / "coxph_fold_results.csv"
    existing.to_csv(fold_results_path, index=False)

    records = runner._records_with_existing_resume_rows(
        existing_fold_results=fold_results_path,
        new_records=[
            {
                "dataset_id": "toy_dataset__base",
                "method_id": "coxph",
                "split_id": "fixed_split_1__base",
                "seed": 11,
                "hpo_mode": "no_hpo",
                "status": "success",
                "uno_c": 0.73,
            }
        ],
        resume=True,
    )

    by_split = {row["split_id"]: row for row in records}
    assert set(by_split) == {"fixed_split_0__base", "fixed_split_1__base"}
    assert by_split["fixed_split_0__base"]["uno_c"] == 0.77
    assert by_split["fixed_split_1__base"]["uno_c"] == 0.73


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
    fold_results.to_csv(tmp_path / "coxph_fold_results.csv", index=False)

    calls = {"count": 0}
    _install_common_monkeypatches(monkeypatch, calls)

    runner.run_benchmark(
        repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0
    )

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
    fold_results.to_csv(tmp_path / "coxph_fold_results.csv", index=False)

    calls = {"count": 0}
    _install_common_monkeypatches(monkeypatch, calls)

    runner.run_benchmark(
        repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=True, max_retries=0
    )

    assert calls["count"] == 2


def test_comparison_modes_can_run_only_no_hpo(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}
    hpo_enabled_values: list[bool] = []
    _install_common_monkeypatches(monkeypatch, calls)

    def _fake_evaluate_split(**kwargs):
        calls["count"] += 1
        hpo_enabled_values.append(bool(kwargs["hpo_cfg"]["enabled"]))
        return _resume_record(status="success")

    monkeypatch.setattr("survarena.benchmark.runner.evaluate_split", _fake_evaluate_split)
    cfg = _resume_benchmark_cfg()
    cfg["comparison_modes"] = ["no_hpo"]

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=cfg, output_dir=tmp_path, resume=False, max_retries=0)

    assert calls["count"] == 1
    assert hpo_enabled_values == [False]


def test_benchmark_profiling_manifest_and_artifact_emitted(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}
    writes: dict[str, dict[str, object]] = {}
    _install_common_monkeypatches(monkeypatch, calls)

    def _capture_write_json(path: Path, payload: dict[str, object]) -> None:
        writes[path.name] = payload

    monkeypatch.setattr("survarena.logging.tracker.write_json", _capture_write_json)
    cfg = _resume_benchmark_cfg()
    cfg["comparison_modes"] = ["no_hpo"]

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=cfg, output_dir=tmp_path, resume=False, max_retries=0)

    manifest = writes["experiment_manifest.json"]
    assert calls["count"] == 1
    assert manifest["profiling"]["schema_version"] == "benchmark_profiling_v1"
    assert "artifact" not in manifest["profiling"]
    assert set(manifest["profiling"]["phase_timings_sec"]) == {"loading", "split_prep", "evaluation", "exports"}
    assert manifest["profiling"]["total_wall_time_sec"] >= 0.0
    assert all(value >= 0.0 for value in manifest["profiling"]["phase_timings_sec"].values())


def test_benchmark_run_emits_compact_artifact_links(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}
    writes: dict[str, dict[str, object]] = {}
    _install_common_monkeypatches(monkeypatch, calls)

    def _capture_write_json(path: Path, payload: dict[str, object]) -> None:
        writes[path.name] = payload

    monkeypatch.setattr("survarena.logging.tracker.write_json", _capture_write_json)
    cfg = _resume_benchmark_cfg()
    cfg["comparison_modes"] = ["no_hpo"]

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=cfg, output_dir=tmp_path, resume=False, max_retries=0)

    assert calls["count"] == 1
    assert (tmp_path / "README.md").exists()
    assert not (tmp_path / "coxph_coverage_matrix.csv").exists()
    assert not (tmp_path / "coxph_coverage_matrix.md").exists()
    assert "coverage_matrix" not in writes["experiment_navigator.json"]["artifacts"]
    assert "coverage_matrix" not in writes["experiment_manifest.json"]["artifacts"]
    assert "coverage_matrix" not in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_benchmark_run_emits_single_compact_multi_method_artifact_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("survarena.data.loaders.load_dataset", lambda *_args, **_kwargs: _fake_dataset())
    monkeypatch.setattr("survarena.data.splitters.load_or_create_splits", lambda **_kwargs: _single_split())
    monkeypatch.setattr(
        "survarena.data.robustness.resolve_robustness_tracks",
        lambda *_args, **_kwargs: [SimpleNamespace(track_id="base")],
    )
    monkeypatch.setattr("survarena.data.robustness.apply_robustness_track", lambda X, **_kwargs: X)
    monkeypatch.setattr("survarena.data.robustness.apply_label_noise", lambda event, **_kwargs: event)
    monkeypatch.setattr("survarena.benchmark.runner.registered_method_ids", lambda: {"coxph", "rsf"})
    monkeypatch.setattr(
        "survarena.benchmark.runner.read_yaml",
        lambda path: {"method_id": Path(path).stem, "default_params": {}, "search_space": {}},
    )

    def _fake_evaluate_split(**kwargs):
        status = "failed" if kwargs["method_id"] == "rsf" else "success"
        failure = None if status == "success" else {"traceback": "RuntimeError: simulated rsf failure"}
        failure_fields = (
            {}
            if status == "success"
            else {"failure_type": "RuntimeError", "exception_message": "simulated rsf failure"}
        )
        return {
            "run_payload": {
                "manifest": {
                    "run_id": f"toy_{kwargs['method_id']}_{kwargs['split'].split_id}_seed{kwargs['split'].seed}",
                    "benchmark_id": kwargs["benchmark_id"],
                    "dataset_id": kwargs["dataset_id"],
                    "method_id": kwargs["method_id"],
                    "split_id": kwargs["split"].split_id,
                    "seed": kwargs["split"].seed,
                    "status": status,
                },
                "metrics": {"status": status, **failure_fields},
                "hpo_metadata": {"status": "disabled", "trial_count": 0},
                "hpo_trials": [],
                "failure": failure,
            },
            "benchmark_id": kwargs["benchmark_id"],
            "dataset_id": kwargs["dataset_id"],
            "method_id": kwargs["method_id"],
            "split_id": kwargs["split"].split_id,
            "seed": kwargs["split"].seed,
            "primary_metric": kwargs["primary_metric"],
            "validation_score": 0.7 if status == "success" else np.nan,
            "uno_c": 0.71 if status == "success" else np.nan,
            "harrell_c": 0.7 if status == "success" else np.nan,
            "ibs": 0.2 if status == "success" else np.nan,
            "td_auc_25": 0.68 if status == "success" else np.nan,
            "td_auc_50": 0.69 if status == "success" else np.nan,
            "td_auc_75": 0.7 if status == "success" else np.nan,
            "tuning_time_sec": 0.01,
            "runtime_sec": 0.02,
            "fit_time_sec": 0.01 if status == "success" else np.nan,
            "infer_time_sec": 0.01 if status == "success" else np.nan,
            "peak_memory_mb": 64.0,
            "status": status,
            **failure_fields,
        }

    monkeypatch.setattr("survarena.benchmark.runner.evaluate_split", _fake_evaluate_split)
    cfg = _resume_benchmark_cfg("compact_artifacts")
    cfg["methods"] = ["coxph", "rsf"]
    cfg["comparison_modes"] = ["no_hpo"]

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=cfg, output_dir=tmp_path, resume=False, max_retries=0)

    expected_artifacts = {
        "multi_model_fold_results.csv",
        "multi_model_leaderboard.csv",
        "multi_model_hpo_budget_summary.csv",
        "multi_model_run_diagnostics.csv",
        "multi_model_runtime_failure_summary.csv",
        "experiment_navigator.json",
        "experiment_manifest.json",
        "README.md",
    }
    assert expected_artifacts.issubset({path.name for path in tmp_path.iterdir()})
    assert not (tmp_path / "multi_model_leaderboard.json").exists()
    assert not (tmp_path / "coxph_fold_results.csv").exists()
    assert not (tmp_path / "rsf_fold_results.csv").exists()

    navigator = json.loads((tmp_path / "experiment_navigator.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert navigator["model_name"] == "multi_model"
    assert navigator["artifacts"] == manifest["artifacts"]
    assert set(navigator["artifacts"].values()) == expected_artifacts - {
        "experiment_navigator.json",
        "experiment_manifest.json",
        "README.md",
    }
    for artifact_name in navigator["artifacts"].values():
        assert artifact_name in readme

    fold_results = pd.read_csv(tmp_path / "multi_model_fold_results.csv")
    assert set(fold_results["method_id"]) == {"coxph", "rsf"}
    assert set(fold_results["status"]) == {"success", "failed"}


def test_comparison_modes_reject_unknown_mode() -> None:
    cfg = _resume_benchmark_cfg()
    cfg["comparison_modes"] = ["no_hpo", "turbo"]

    with pytest.raises(ValueError, match="Invalid comparison mode"):
        runner._resolve_comparison_modes(cfg)


def test_benchmark_execution_defaults_to_serial(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}
    _install_common_monkeypatches(monkeypatch, calls)

    class RaisingExecutor:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("ThreadPoolExecutor should not be used for default serial execution")

    monkeypatch.setattr(runner, "ThreadPoolExecutor", RaisingExecutor)
    cfg = _resume_benchmark_cfg()
    cfg["comparison_modes"] = ["no_hpo"]

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=cfg, output_dir=tmp_path, resume=False, max_retries=0)

    assert calls["count"] == 1


def test_benchmark_execution_uses_configured_n_jobs(tmp_path: Path, monkeypatch) -> None:
    calls = {"count": 0}
    captured_workers: list[int] = []
    _install_common_monkeypatches(monkeypatch, calls)
    monkeypatch.setattr("survarena.data.splitters.load_or_create_splits", lambda **_kwargs: _single_split() * 2)

    class RecordingExecutor:
        def __init__(self, *, max_workers: int) -> None:
            captured_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def map(self, func, units):
            return [func(unit) for unit in units]

    monkeypatch.setattr(runner, "ThreadPoolExecutor", RecordingExecutor)
    cfg = _resume_benchmark_cfg()
    cfg["comparison_modes"] = ["no_hpo"]
    cfg["execution"] = {"n_jobs": 2}

    runner.run_benchmark(repo_root=tmp_path, benchmark_cfg=cfg, output_dir=tmp_path, resume=False, max_retries=0)

    assert calls["count"] == 2
    assert captured_workers == [2]


def test_benchmark_execution_rejects_invalid_n_jobs() -> None:
    cfg = _resume_benchmark_cfg()
    cfg["execution"] = {"n_jobs": 0}

    with pytest.raises(ValueError, match="execution.n_jobs"):
        runner._resolve_execution_n_jobs(cfg)


def test_exec04_retry_budget_caps_failed_rows(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    calls = _install_retry_monkeypatches(
        monkeypatch, statuses=["failed", "failed", "failed"], captured_run_records=run_records
    )

    runner.run_benchmark(
        repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=1
    )

    assert calls["count"] == 4
    assert run_records


def test_exec04_failure_records_include_attempt_metadata(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    _install_retry_monkeypatches(monkeypatch, statuses=["failed", "failed"], captured_run_records=run_records)

    runner.run_benchmark(
        repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=1
    )

    assert run_records


def test_exec04_successful_retry_keeps_failed_attempt_evidence(tmp_path: Path, monkeypatch) -> None:
    run_records: list[dict[str, object]] = []
    calls = _install_retry_monkeypatches(monkeypatch, statuses=["failed", "success"], captured_run_records=run_records)

    runner.run_benchmark(
        repo_root=tmp_path, benchmark_cfg=_resume_benchmark_cfg(), output_dir=tmp_path, resume=False, max_retries=2
    )

    assert calls["count"] == 3
    assert run_records


# --- test_benchmark_exports.py ---


def test_coverage_matrix_exports_csv_only(tmp_path) -> None:
    fold_results = pd.DataFrame(
        [
            {
                "benchmark_id": "manuscript_v1",
                "dataset_id": "whas500__base",
                "method_id": "coxph",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_0__base",
                "status": "success",
                "runtime_sec": 1.25,
                "uno_c": 0.72,
                "harrell_c": 0.70,
                "robustness_track_id": "base",
            },
            {
                "benchmark_id": "manuscript_v1",
                "dataset_id": "whas500__base",
                "method_id": "coxnet",
                "hpo_mode": "hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_1__base",
                "status": "success",
                "runtime_sec": 1.0,
                "uno_c": None,
                "harrell_c": 0.69,
            },
            {
                "benchmark_id": "manuscript_v1",
                "dataset_id": "whas500__base",
                "method_id": "rsf",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_2__base",
                "status": "failed",
                "runtime_sec": 0.5,
                "failure_type": "ValueError",
                "exception_message": "bad split",
                "uno_c": None,
            },
        ]
    )

    coverage = export_coverage_matrix(
        tmp_path,
        fold_results,
        primary_metric="uno_c",
        output_dir=tmp_path,
        file_prefix="coxph",
    )

    assert (tmp_path / "coxph_coverage_matrix.csv").exists()
    assert not (tmp_path / "coxph_coverage_matrix.md").exists()
    by_method = coverage.set_index("method_id")
    assert by_method.loc["coxph", "coverage_status"] == "success"
    assert by_method.loc["coxnet", "coverage_status"] == "missing_metric"
    assert by_method.loc["rsf", "coverage_status"] == "failed"
    assert by_method["repeat"].tolist() == [0, 0, 0]
    assert by_method.loc["coxph", "fold"] == 0
    assert by_method.loc["coxnet", "failure_reason"] == "missing_primary_metric"
    assert by_method.loc["rsf", "failure_reason"] == "ValueError"
    assert by_method["artifact_path"].tolist() == ["coxph_fold_results.csv"] * 3


def test_hpo_budget_summary_reports_requested_vs_realized_trials(tmp_path) -> None:
    fold_results = pd.DataFrame(
        [
            {
                "benchmark_id": "cloud",
                "dataset_id": "whas500",
                "method_id": "fast_survival_svm",
                "hpo_mode": "hpo",
                "requested_max_trials": 50,
                "realized_trial_count": 42,
                "requested_timeout_seconds": 2400,
                "requested_sampler": "tpe",
                "requested_pruner": "median",
                "hpo_budget_tier": "reduced",
                "hpo_config_target": 100,
                "hpo_cap_reason": "runtime cap",
            },
            {
                "benchmark_id": "cloud",
                "dataset_id": "whas500",
                "method_id": "fast_survival_svm",
                "hpo_mode": "hpo",
                "requested_max_trials": 50,
                "realized_trial_count": 50,
                "requested_timeout_seconds": 2400,
                "requested_sampler": "tpe",
                "requested_pruner": "median",
                "hpo_budget_tier": "reduced",
                "hpo_config_target": 100,
                "hpo_cap_reason": "runtime cap",
            },
        ]
    )

    summary = export_hpo_budget_summary(tmp_path, fold_results, output_dir=tmp_path, file_prefix="combined")

    row = summary.iloc[0]
    assert (tmp_path / "combined_hpo_budget_summary.csv").exists()
    assert row["requested_trials_total"] == 100
    assert row["realized_trials_total"] == 92
    assert row["trial_realization_rate"] == 0.92
    assert row["hpo_config_target"] == 100
    assert bool(row["hpo_capped"]) is True
    assert row["hpo_cap_reason"] == "runtime cap"


# --- test_runtime_failure_summary.py ---


def _synthetic_fold_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "benchmark_id": "bench",
                "dataset_id": "toy",
                "method_id": "coxph",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_0__base",
                "status": "success",
                "validation_score": np.nan,
                "uno_c": 0.71,
                "harrell_c": 0.72,
                "ibs": 0.2,
                "td_auc_25": 0.68,
                "td_auc_50": 0.69,
                "td_auc_75": 0.7,
                "runtime_sec": 1.2,
            },
            {
                "benchmark_id": "bench",
                "dataset_id": "toy",
                "method_id": "rsf",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_0__base",
                "status": "failed",
                "validation_score": np.nan,
                "uno_c": np.nan,
                "harrell_c": np.nan,
                "ibs": np.nan,
                "td_auc_25": np.nan,
                "td_auc_50": np.nan,
                "td_auc_75": np.nan,
                "runtime_sec": 0.4,
            },
            {
                "benchmark_id": "bench",
                "dataset_id": "toy",
                "method_id": "coxnet",
                "hpo_mode": "hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_1__base",
                "status": "success",
                "validation_score": 0.6,
                "uno_c": np.nan,
                "harrell_c": 0.61,
                "ibs": 0.25,
                "td_auc_25": np.nan,
                "td_auc_50": np.nan,
                "td_auc_75": np.nan,
                "runtime_sec": 2.0,
            },
        ]
    )


def test_runtime_failure_summary_exports_csv_only(tmp_path: Path) -> None:
    run_records = [
        {
            "metrics": {
                "dataset_id": "toy",
                "method_id": "rsf",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "repeat_0_fold_0__base",
                "failure_type": "ImportError",
                "exception_message": "No module named sksurv",
            },
            "failure": {"traceback": "ModuleNotFoundError: No module named sksurv"},
        }
    ]

    summary = export_runtime_failure_summary(
        tmp_path,
        benchmark_id="bench",
        fold_results=_synthetic_fold_results(),
        run_records=run_records,
        output_dir=tmp_path,
        file_prefix="toy",
    )

    assert (tmp_path / "toy_runtime_failure_summary.csv").exists()
    assert not (tmp_path / "toy_runtime_failure_summary.md").exists()
    by_method = summary.set_index("method_id")
    assert by_method.loc["coxph", "failure_category"] == "success"
    assert by_method.loc["coxph", "missing_metric_columns"] == ""
    assert by_method.loc["rsf", "failure_category"] == "dependency_missing"
    assert by_method.loc["rsf", "n_crashed"] == 1
    assert by_method.loc["coxnet", "failure_category"] == "missing_metrics"
    assert by_method.loc["coxnet", "n_missing_metrics"] == 1
    assert by_method.loc["coxnet", "missing_metric_columns"] == "uno_c, td_auc_25, td_auc_50, td_auc_75"
    assert by_method.loc["coxph", "runtime_sec_mean"] == 1.2


def test_run_diagnostics_writes_runtime_failure_summary(tmp_path: Path) -> None:
    export_run_diagnostics(
        tmp_path,
        benchmark_id="bench",
        fold_results=_synthetic_fold_results(),
        dataset_curation_rows=[],
        hpo_trial_rows=[],
        output_dir=tmp_path,
        file_prefix="toy",
    )

    assert (tmp_path / "toy_run_diagnostics.csv").exists()
    assert (tmp_path / "toy_runtime_failure_summary.csv").exists()
    assert not (tmp_path / "toy_runtime_failure_summary.md").exists()


def test_runtime_failure_summary_classifies_foundation_readiness_failures(tmp_path: Path) -> None:
    fold_results = pd.DataFrame(
        [
            {
                "benchmark_id": "toy",
                "dataset_id": "toy_dataset__base",
                "method_id": "tabpfn_survival",
                "hpo_mode": "no_hpo",
                "seed": 11,
                "split_id": "fixed_split_0__base",
                "status": "failed",
                "runtime_sec": 0.1,
                "uno_c": np.nan,
            },
            {
                "benchmark_id": "toy",
                "dataset_id": "toy_dataset__base",
                "method_id": "tabpfn_survival",
                "hpo_mode": "hpo",
                "seed": 11,
                "split_id": "fixed_split_0__base",
                "status": "failed",
                "runtime_sec": 0.2,
                "uno_c": np.nan,
            },
        ]
    )
    summary = export_runtime_failure_summary(
        tmp_path,
        benchmark_id="toy",
        fold_results=fold_results,
        run_records=[
            {
                "manifest": {
                    "dataset_id": "toy_dataset__base",
                    "method_id": "tabpfn_survival",
                    "seed": 11,
                    "split_id": "fixed_split_0__base",
                },
                "metrics": {
                    "hpo_mode": "no_hpo",
                    "failure_type": "RuntimeError",
                    "exception_message": "Dependency 'tabpfn' is not installed. Install it with `python -m pip install -e \".[foundation-tabpfn]\"`.",
                },
                "failure": {"traceback": "RuntimeError: Dependency 'tabpfn' is not installed."},
            },
            {
                "manifest": {
                    "dataset_id": "toy_dataset__base",
                    "method_id": "tabpfn_survival",
                    "seed": 11,
                    "split_id": "fixed_split_0__base",
                },
                "metrics": {
                    "hpo_mode": "hpo",
                    "failure_type": "RuntimeError",
                    "exception_message": "TabPFN access is not ready for the default gated checkpoint. Run `hf auth login`.",
                },
                "failure": {"traceback": "RuntimeError: Hugging Face authentication was not detected."},
            },
        ],
        output_dir=tmp_path,
        file_prefix="toy",
    )

    by_mode = summary.set_index("hpo_mode")
    assert by_mode.loc["no_hpo", "failure_category"] == "dependency_missing"
    assert by_mode.loc["hpo", "failure_category"] == "auth_missing"
