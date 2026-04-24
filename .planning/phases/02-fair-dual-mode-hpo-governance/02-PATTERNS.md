# Phase 2: Fair Dual-Mode HPO Governance - Pattern Map

**Mapped:** 2026-04-23  
**Files analyzed:** 7  
**Analogs found:** 7 / 7

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `survarena/benchmark/runner.py` | service | batch | `survarena/benchmark/runner.py` | exact |
| `survarena/benchmark/tuning.py` | service | transform | `survarena/benchmark/tuning.py` | exact |
| `survarena/logging/export.py` | utility | file-I/O | `survarena/logging/export.py` | exact |
| `survarena/api/compare.py` | service | request-response | `survarena/api/compare.py` | exact |
| `tests/test_hpo_config.py` | test | transform | `tests/test_hpo_config.py` | exact |
| `tests/test_compare_api.py` | test | request-response | `tests/test_compare_api.py` | exact |
| `tests/test_dual_mode_hpo_governance.py` | test | batch | `tests/test_compare_api.py` | role+flow |

## Pattern Assignments

### `survarena/benchmark/runner.py` (service, batch)

**Analog:** `survarena/benchmark/runner.py`

**Orchestration import pattern** (lines 331-345):
```python
from survarena.data.loaders import load_dataset
from survarena.data.robustness import apply_label_noise, apply_robustness_track, resolve_robustness_tracks
from survarena.data.splitters import load_or_create_splits
from survarena.logging.export import (
    create_experiment_dir,
    export_dataset_curation_table,
    export_fold_results,
    export_hpo_trials,
    export_leaderboard,
    export_manuscript_comparison,
    export_overall_summary,
    export_run_ledger,
    export_seed_summary,
)
```

**Config normalization pattern** (lines 382-389):
```python
hpo_cfg = dict(benchmark_cfg.get("hpo", {}))
hpo_cfg.setdefault("enabled", False)
hpo_cfg.setdefault("max_trials", 20)
hpo_cfg.setdefault("timeout_seconds", None)
hpo_cfg.setdefault("sampler", "tpe")
hpo_cfg.setdefault("pruner", "median")
hpo_cfg.setdefault("n_startup_trials", 8)
```

**Core run-loop pattern for deterministic unit execution** (lines 491-559):
```python
for method_id in methods:
    if method_id not in registered_methods:
        raise ValueError(f"Unknown method_id '{method_id}'. Registered: {sorted(registered_methods)}")
    method_cfg = method_cfg_cache[method_id]
    for split in filtered_splits:
        for track in robustness_tracks:
            ...
            while True:
                record = evaluate_split(...)
                run_payload = record.pop("run_payload")
                run_payload["metrics"]["retry_attempt"] = int(attempt)
                run_records.append(run_payload)
                all_records.append(record)
                if record["status"] == "success" or attempt >= max_retries:
                    print(f"[{record['status']}] ...")
                    break
                attempt += 1
```

**Structured failure record pattern** (lines 212-251):
```python
except Exception as exc:  # noqa: BLE001
    tb_str = traceback.format_exc()
    run_payload = {
        "manifest": manifest.to_dict(),
        "metrics": {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "method_id": method_id,
            "status": "failed",
            "failure_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        "failure": {"traceback": tb_str},
    }
```

---

### `survarena/benchmark/tuning.py` (service, transform)

**Analog:** `survarena/benchmark/tuning.py`

**Default policy parse + coercion pattern** (lines 63-82):
```python
def _parse_hpo_config(method_cfg: dict[str, Any], hpo_config: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "enabled": bool(method_cfg.get("search_space")) and False,
        "max_trials": 20,
        "timeout_seconds": None,
        "sampler": "tpe",
        "pruner": "median",
        "n_startup_trials": 8,
    }
    cfg = dict(base)
    if hpo_config:
        cfg.update({k: v for k, v in hpo_config.items() if v is not None})
    cfg["enabled"] = bool(cfg.get("enabled", False)) and bool(method_cfg.get("search_space"))
    cfg["max_trials"] = max(int(cfg.get("max_trials", 20)), 1)
    cfg["timeout_seconds"] = None if cfg.get("timeout_seconds") is None else max(float(cfg["timeout_seconds"]), 0.0)
```

**Disabled/early-return governance pattern** (lines 177-203):
```python
resolved_hpo = _parse_hpo_config(method_cfg, hpo_config)
default_result = {
    "best_params": defaults,
    "best_score": default_score,
    "hpo_metadata": {"enabled": bool(resolved_hpo["enabled"]), "status": "disabled", ...},
    "hpo_trials": [],
}
if not resolved_hpo["enabled"]:
    return default_result

search_space = dict(method_cfg.get("search_space", {}))
if not search_space:
    return default_result
```

**Backend-status reporting pattern** (lines 204-210, 288-303):
```python
try:
    import optuna
except ModuleNotFoundError:
    default_result["hpo_metadata"]["status"] = "optuna_missing"
    return default_result

return {
    "best_params": selected,
    "best_score": float(best_eval["primary_score"]),
    "hpo_metadata": {
        "enabled": True,
        "backend": "optuna",
        "status": "success",
        "trial_count": int(len(study.trials)),
        "max_trials": int(resolved_hpo["max_trials"]),
        "timeout_seconds": resolved_hpo["timeout_seconds"],
    },
    "hpo_trials": trial_rows,
}
```

---

### `survarena/logging/export.py` (utility, file-I/O)

**Analog:** `survarena/logging/export.py`

**Stable sort + canonical table write pattern** (lines 94-110):
```python
frame = pd.DataFrame(records)
sort_cols = [col for col in ["benchmark_id", "dataset_id", "method_id", "seed", "split_id"] if col in frame.columns]
if sort_cols:
    frame.sort_values(sort_cols, inplace=True)
...
output = output_dir / f"{prefix}_fold_results.csv"
frame.to_csv(output, index=False)
```

**Comparative summary assembly pattern** (lines 212-219):
```python
rank_summary = aggregate_rank_summary(leaderboard, metric=primary_metric)
pairwise = pairwise_win_rate(leaderboard, metric=primary_metric)
significance_source = fold_results if fold_results is not None else leaderboard
pairwise_sig = pairwise_significance(significance_source, metric=primary_metric, correction="holm")
cd_summary = critical_difference_summary(leaderboard, metric=primary_metric)
ci = bootstrap_metric_ci(leaderboard, metric=primary_metric, n_bootstrap=1000, seed=0)
elo = elo_ratings(leaderboard, metric=primary_metric)
```

**Run-ledger schema/index export pattern** (lines 306-391):
```python
normalized_records = [{"schema_version": RUN_LEDGER_SCHEMA_VERSION, **record} for record in run_records]
write_jsonl_gz(output, normalized_records)
...
write_json(
    index_output,
    {
        "schema_version": RUN_LEDGER_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "record_count": len(normalized_records),
        "record_sections": ["schema_version", "manifest", "metrics", "backend_metadata", "hpo_metadata", "hpo_trials", "failure"],
    },
)
```

---

### `survarena/api/compare.py` (service, request-response)

**Analog:** `survarena/api/compare.py`

**API boundary import pattern** (lines 6-23):
```python
from survarena.automl.presets import resolve_preset
from survarena.benchmark.runner import evaluate_split
from survarena.data.splitters import load_or_create_splits
from survarena.data.user_dataset import load_user_dataset
from survarena.logging.export import (
    create_experiment_dir,
    export_dataset_curation_table,
    export_experiment_navigator,
    export_fold_results,
    export_leaderboard,
    export_manuscript_comparison,
    export_overall_summary,
    export_run_ledger,
    export_seed_summary,
)
```

**Fail-fast input validation pattern** (lines 128-146):
```python
registered_methods = set(registered_method_ids())
unknown_methods = sorted(set(method_ids) - registered_methods)
if unknown_methods:
    raise ValueError(f"Unknown method ids: {unknown_methods}. Registered: {sorted(registered_methods)}")

if split_strategy not in {"fixed_split", "repeated_nested_cv"}:
    raise ValueError(...)
if split_strategy == "fixed_split":
    if outer_repeats != 1:
        raise ValueError("fixed_split supports exactly one repeat. ...")
```

**Summary + manifest contract pattern** (lines 156-220):
```python
benchmark_cfg = {
    "benchmark_id": resolved_benchmark_id,
    "methods": list(method_ids),
    "hpo": hpo_cfg,
    "decision_curve_thresholds": list(resolved_thresholds),
}
benchmark_cfg_hash = payload_sha256(benchmark_cfg)
summary = {
    "benchmark_id": resolved_benchmark_id,
    "methods": list(method_ids),
    "hpo": hpo_cfg,
}
write_json(
    resolved_output_dir / "experiment_manifest.json",
    {**summary, "benchmark_config_hash": benchmark_cfg_hash, "output_dir": str(resolved_output_dir)},
)
```

---

### `tests/test_hpo_config.py` (test, transform)

**Analog:** `tests/test_hpo_config.py`

**Direct module-under-test import pattern** (lines 3-4):
```python
from survarena.benchmark import tuning
```

**Focused parse/coercion assertion pattern** (lines 6-19):
```python
def test_parse_hpo_config_defaults_disabled_without_search_space() -> None:
    cfg = tuning._parse_hpo_config({"default_params": {"a": 1}}, {"enabled": True})
    assert cfg["enabled"] is False

def test_parse_hpo_config_enables_with_space() -> None:
    cfg = tuning._parse_hpo_config(..., {"enabled": True, "max_trials": 5, "sampler": "random"})
    assert cfg["enabled"] is True
    assert cfg["max_trials"] == 5
    assert cfg["sampler"] == "random"
```

---

### `tests/test_compare_api.py` (test, request-response)

**Analog:** `tests/test_compare_api.py`

**Fixture + split-definition pattern** (lines 12-29):
```python
frame = pd.DataFrame({...})
split = SplitDefinition(
    split_id="fixed_split_0",
    seed=11,
    repeat=0,
    fold=0,
    train_idx=np.asarray([0, 1, 2, 3], dtype=int),
    test_idx=np.asarray([4, 5], dtype=int),
    val_idx=np.asarray([2, 3], dtype=int),
)
```

**Monkeypatch orchestration seam pattern** (lines 31-33, 61):
```python
monkeypatch.setattr("survarena.api.compare.load_or_create_splits", lambda **kwargs: [split])
...
monkeypatch.setattr("survarena.api.compare.evaluate_split", fake_evaluate_split)
```

**Artifact contract assertion pattern** (lines 77-95):
```python
assert (output_dir / "experiment_manifest.json").exists()
assert (output_dir / "user_compare_fixed_fold_results.csv").exists()
assert (output_dir / "user_compare_fixed_leaderboard.csv").exists()
assert (output_dir / "user_compare_fixed_run_records.jsonl.gz").exists()
assert (output_dir / "experiment_navigator.json").exists()
```

---

### `tests/test_dual_mode_hpo_governance.py` (test, batch)

**Analog:** `tests/test_compare_api.py` and `tests/test_benchmark_runner.py`

**Contract-style benchmark validation pattern** (`test_benchmark_runner.py` lines 18-26, 43-46 for `_base_cfg` + contract rejection):
```python
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

with pytest.raises(ValueError, match="Missing required deterministic fields"):
    validate_benchmark_profile_contract(cfg)
```

**Batch orchestration test-double pattern** (`test_compare_api.py` lines 33-59):
```python
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
        "status": "success",
    }
```

**Governance outcome assertion style** (`test_compare_api.py` lines 73-95):
```python
assert summary["benchmark_id"] == "user_compare_fixed"
assert summary["methods"] == ["coxph"]
assert summary["split_count"] == 1
assert (output_dir / "user_compare_fixed_fold_results.csv").exists()
```

## Shared Patterns

### Single canonical artifact set + per-row metadata
**Source:** `survarena/benchmark/runner.py` (lines 148-183, 534-557)  
**Apply to:** dual-mode labeling (`hpo_mode`), parity identifiers, and mode-level eligibility markers in run payload rows
```python
run_payload = {
    "manifest": manifest.to_dict(),
    "metrics": {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "method_id": method_id,
        "seed": split.seed,
        "split_id": split.split_id,
        ...
        "hpo_status": hpo_metadata.get("status", "disabled"),
        "hpo_trial_count": hpo_metadata.get("trial_count", 0),
    },
    "hpo_metadata": hpo_metadata,
    "hpo_trials": hpo_trials,
}
```

### Uniform HPO policy normalization
**Source:** `survarena/benchmark/tuning.py` (lines 63-82, 245-249)  
**Apply to:** benchmark-wide budget policy coercion before mode execution and explicit requested budget propagation
```python
cfg["max_trials"] = max(int(cfg.get("max_trials", 20)), 1)
cfg["timeout_seconds"] = None if timeout_seconds is None else max(float(timeout_seconds), 0.0)
...
study.optimize(
    _objective,
    n_trials=int(resolved_hpo["max_trials"]),
    timeout=resolved_hpo["timeout_seconds"],
)
```

### Parity gate insertion point before comparative claims
**Source:** `survarena/benchmark/runner.py` (lines 567-584) and `survarena/logging/export.py` (lines 212-219)  
**Apply to:** filter comparison-ineligible rows before `export_leaderboard`, `pairwise_win_rate`, `pairwise_significance`, and related summary exports
```python
frame = export_fold_results(...)
seed_summary = export_seed_summary(..., frame, ...)
leaderboard = export_leaderboard(..., seed_summary, ...)
export_manuscript_comparison(..., leaderboard, fold_results=frame, ...)
```

### Explicit status surfaces instead of silent fallback
**Source:** `survarena/benchmark/tuning.py` (lines 197-210, 252-261) and `survarena/benchmark/runner.py` (lines 173-177, 281-283)  
**Apply to:** parity and budget governance failures (`disabled`, `optuna_missing`, `no_valid_trial`, `failed`) that must remain auditable
```python
if not resolved_hpo["enabled"]:
    return default_result
...
default_result["hpo_metadata"]["status"] = "optuna_missing"
...
"hpo_status": hpo_metadata.get("status", "disabled"),
"hpo_trial_count": hpo_metadata.get("trial_count", 0),
```

### Test pattern: monkeypatch + deterministic split + artifact assertions
**Source:** `tests/test_compare_api.py` and `tests/test_benchmark_runner.py`  
**Apply to:** new dual-mode governance tests covering no-HPO->HPO ordering, parity gating, and requested-vs-realized budget reporting
```python
def test_xxx(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("survarena.api.compare.evaluate_split", fake_evaluate_split)
    ...
    with pytest.raises(ValueError, match="..."):
        ...
    assert (output_dir / "...").exists()
```

## No Analog Found

None. All identified phase files have close analogs in the current codebase.

## Metadata

**Analog search scope:** `survarena/benchmark/`, `survarena/logging/`, `survarena/api/`, `tests/`  
**Files scanned:** 10  
**Pattern extraction date:** 2026-04-23
