# Phase 1: Deterministic Execution Foundation - Pattern Map

**Mapped:** 2026-04-23  
**Files analyzed:** 9  
**Analogs found:** 9 / 9

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `survarena/run_benchmark.py` | config | request-response | `survarena/run_benchmark.py` | exact |
| `survarena/benchmark/runner.py` | service | batch | `survarena/benchmark/runner.py` | exact |
| `survarena/data/splitters.py` | utility | transform | `survarena/data/splitters.py` | exact |
| `survarena/logging/export.py` | utility | file-I/O | `survarena/logging/export.py` | exact |
| `configs/benchmark/smoke_all_models_no_hpo.yaml` | config | batch | `configs/benchmark/standard_v1.yaml` | role-match |
| `configs/benchmark/standard_v1.yaml` | config | batch | `configs/benchmark/manuscript_v1.yaml` | role-match |
| `configs/benchmark/manuscript_v1.yaml` | config | batch | `configs/benchmark/standard_v1.yaml` | role-match |
| `tests/test_benchmark_runner.py` | test | transform + batch | `tests/test_robustness_tracks.py` (splits) / `tests/test_compare_api.py` (resume) | role-match + role+flow |

## Pattern Assignments

### `survarena/run_benchmark.py` (config, request-response)

**Analog:** `survarena/run_benchmark.py`

**Imports + thin entrypoint pattern** (lines 3-8):
```python
import argparse
from pathlib import Path

from survarena.benchmark.runner import run_benchmark
from survarena.config import read_yaml
```

**CLI argument contract pattern** (lines 12-25):
```python
parser.add_argument("--benchmark-config", type=str, default="configs/benchmark/standard_v1.yaml")
parser.add_argument("--resume", action="store_true", help="Resume from an existing output directory if available.")
parser.add_argument("--max-retries", type=int, default=0, help="Retry failed runs this many times.")
parser.add_argument("--dry-run", action="store_true", help="Validate setup without fitting models.")
```

**Validation + delegate pattern** (lines 31-54):
```python
try:
    benchmark_cfg = read_yaml(repo_root / args.benchmark_config)
except ModuleNotFoundError as exc:
    if args.dry_run:
        print("Dry run completed with missing optional dependency.")
        print(f"missing_module={exc.name}")
        print("Install the required package extras before full benchmark execution.")
        return
    raise

run_benchmark(
    repo_root=repo_root,
    benchmark_cfg=benchmark_cfg,
    resume=bool(args.resume),
    max_retries=max(int(args.max_retries), 0),
)
```

---

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

**Resume-eligibility seed key pattern** (lines 417-433):
```python
completed_keys: set[tuple[str, str, str, int]] = set()
if resume:
    existing_fold_results = experiment_dir / f"{benchmark_id}_fold_results.csv"
    if existing_fold_results.exists():
        import pandas as pd

        existing = pd.read_csv(existing_fold_results)
        for row in existing.to_dict(orient="records"):
            if row.get("status") == "success":
                completed_keys.add(
                    (str(row.get("dataset_id")), str(row.get("method_id")), str(row.get("split_id")), int(row.get("seed", 0)))
                )
```

**Retry loop + structured attempt tracking pattern** (lines 514-565):
```python
attempt = 0
while True:
    record = evaluate_split(...)
    run_payload = record.pop("run_payload")
    run_payload["metrics"]["retry_attempt"] = int(attempt)
    run_records.append(run_payload)
    record["retry_attempt"] = int(attempt)
    all_records.append(record)
    if record["status"] == "success" or attempt >= max_retries:
        print(f"[{record['status']}] ...")
        break
    attempt += 1
```

**Failure payload pattern in split evaluation** (lines 212-251):
```python
except Exception as exc:  # noqa: BLE001
    tb_str = traceback.format_exc()
    run_payload = {
        "manifest": manifest.to_dict(),
        "metrics": {
            "status": "failed",
            "failure_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        "failure": {"traceback": tb_str},
    }
```

---

### `survarena/data/splitters.py` (utility, transform)

**Analog:** `survarena/data/splitters.py`

**Deterministic manifest payload pattern** (lines 54-89):
```python
def _expected_split_manifest_payload(...):
    payload: dict[str, object] = {
        "version": _SPLIT_MANIFEST_VERSION,
        "split_strategy": split_strategy,
        "n_samples": int(n_samples),
        "event_fingerprint": _event_fingerprint(event),
        "event_rate": float(np.mean(event)),
        "seeds": [int(seed) for seed in seeds],
    }
    if split_strategy == "repeated_nested_cv":
        payload.update({"outer_folds": int(outer_folds), "outer_repeats": int(outer_repeats), "seed_policy": "one_seed_per_repeat"})
    elif split_strategy == "fixed_split":
        payload.update({"outer_folds": None, "outer_repeats": None, "seed_policy": "single_fixed_split"})
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")
    return payload
```

**Manifest reuse gate pattern** (lines 281-289):
```python
manifest_path = _split_manifest_path(root, task_id)
if manifest_path.exists():
    manifest = read_split_manifest(manifest_path)
    if manifest.get("manifest_payload") == manifest_payload:
        split_ids = [str(split_id) for split_id in manifest.get("split_ids", [])]
        loaded_splits = [read_split(_split_file_path(root, task_id, split_id)) for split_id in split_ids]
        _validate_split_integrity(loaded_splits, n_samples)
        _validate_event_stratification(loaded_splits, event)
        return loaded_splits
```

**Integrity + stratification hard-fail validation pattern** (lines 216-271):
```python
def _validate_split_integrity(...):
    if split.split_id in seen_split_ids:
        raise ValueError(f"Duplicate split_id detected: {split.split_id}")
    ...
    if np.intersect1d(train_idx, test_idx).size > 0:
        raise ValueError(f"Train/test overlap detected for {split.split_id}")

def _validate_event_stratification(...):
    if abs(train_rate - overall_rate) > tolerance:
        raise ValueError(...)
```

---

### `survarena/logging/export.py` (utility, file-I/O)

**Analog:** `survarena/logging/export.py`

**Ledger normalization + schema versioning pattern** (lines 306-313):
```python
created_at = datetime.now().isoformat(timespec="seconds")
normalized_records = [
    {
        "schema_version": RUN_LEDGER_SCHEMA_VERSION,
        **record,
    }
    for record in run_records
]
```

**Dual-output (full + compact) export pattern** (lines 315-325, 336-349):
```python
output = output_dir / f"{benchmark_id}_run_records.jsonl.gz"
compact_output = output_dir / f"{benchmark_id}_run_records_compact.jsonl.gz"
write_jsonl_gz(output, normalized_records)
...
compact_records.append(compact)
write_jsonl_gz(compact_output, compact_records)
```

**Index metadata contract pattern** (lines 350-391):
```python
write_json(index_output, {"schema_version": RUN_LEDGER_SCHEMA_VERSION, "record_sections": [...]})
write_json(
    compact_index_output,
    {
        "schema_version": RUN_LEDGER_COMPACT_SCHEMA_VERSION,
        "manifest_shared": shared_manifest,
        "record_sections": [...],
    },
)
```

---

### `configs/benchmark/smoke_all_models_no_hpo.yaml` (config, batch)

**Analog:** `configs/benchmark/standard_v1.yaml`

**Profile contract pattern** (smoke lines 1-9, standard lines 1-9):
```yaml
benchmark_id: smoke_all_models_no_hpo
split_strategy: repeated_nested_cv
outer_folds: 2
outer_repeats: 1
seeds: [11]
primary_metric: uno_c
profile: smoke
```

**Execution control section pattern** (smoke lines 25-38):
```yaml
hpo:
  enabled: false
  max_trials: 4
  timeout_seconds: 120
decision_curve:
  thresholds: [0.2]
robustness:
  enabled: false
time_horizons_quantiles: [0.25, 0.5, 0.75]
```

---

### `configs/benchmark/standard_v1.yaml` (config, batch)

**Analog:** `configs/benchmark/manuscript_v1.yaml`

**Research-tier config shape pattern** (standard lines 1-9):
```yaml
benchmark_id: standard_v1
split_strategy: repeated_nested_cv
outer_folds: 5
outer_repeats: 3
seeds: [11, 22, 33, 44, 55]
primary_metric: uno_c
profile: research
```

**HPO-enabled benchmark policy pattern** (standard lines 22-28):
```yaml
hpo:
  enabled: true
  max_trials: 30
  timeout_seconds: 1800
  sampler: tpe
  pruner: median
```

---

### `configs/benchmark/manuscript_v1.yaml` (config, batch)

**Analog:** `configs/benchmark/standard_v1.yaml`

**Publication-grade profile defaults pattern** (manuscript lines 1-15):
```yaml
benchmark_id: manuscript_v1
split_strategy: repeated_nested_cv
outer_folds: 5
outer_repeats: 3
seeds: [11, 22, 33, 44, 55]
primary_metric: uno_c
secondary_metrics:
  - harrell_c
  - ibs
  - td_auc
  - brier
  - calibration
  - net_benefit
```

---

### `tests/test_benchmark_runner.py` (test, transform + batch)

**Scope:** `tests/test_benchmark_runner.py` groups deterministic profile and split-manifest tests (with `tests/test_robustness_tracks.py`-style split assertions) and mocked resume/EXEC-04 tests (with `tests/test_compare_api.py`-style run payloads).

**Determinism (profile + splits) — analog:** `tests/test_robustness_tracks.py` (for split fixture + deterministic transform assertions), `tests/test_io_config.py` (for fail-fast `ValueError` assertions)

**Split fixture pattern** (`test_robustness_tracks.py` lines 10-18):
```python
def _split() -> SplitDefinition:
    return SplitDefinition(
        split_id="s0",
        seed=11,
        repeat=0,
        fold=0,
        train_idx=np.asarray([0, 1, 2, 3], dtype=int),
        test_idx=np.asarray([4, 5], dtype=int),
    )
```

**Deterministic mutation assertion pattern** (`test_robustness_tracks.py` lines 36-39):
```python
out = apply_robustness_track(X, track=track, split=_split(), seed=11)
assert np.allclose(out.iloc[:4].to_numpy(), X.iloc[:4].to_numpy())
assert not np.allclose(out.iloc[4:].to_numpy(), X.iloc[4:].to_numpy())
```

**Hard-fail validation assertion pattern** (`test_io_config.py` lines 39-40):
```python
with pytest.raises(ValueError, match="Unsupported file format"):
    read_tabular_data(path)
```

**Resume / EXEC-04 (mocked runner) — analog:** `tests/test_compare_api.py`

**Monkeypatch orchestration boundary pattern** (`test_benchmark_runner.py` resume helpers, mirroring `test_compare_api.py` lines 31-33, 61):
```python
monkeypatch.setattr("survarena.api.compare.load_or_create_splits", lambda **kwargs: [split])
...
monkeypatch.setattr("survarena.api.compare.evaluate_split", fake_evaluate_split)
```

**Structured run payload test double pattern** (lines 34-39, 58):
```python
return {
    "run_payload": {
        "manifest": {"run_id": "toy_coxph_fixed_split_0_seed11"},
        "metrics": {"status": "success"},
        "failure": None,
    },
    ...
    "status": "success",
}
```

**Artifact-presence assertions pattern** (lines 77-95):
```python
assert (output_dir / "experiment_manifest.json").exists()
assert (output_dir / "user_compare_fixed_fold_results.csv").exists()
assert (output_dir / "user_compare_fixed_run_records.jsonl.gz").exists()
assert (output_dir / "user_compare_fixed_run_records_compact_index.json").exists()
```

## Shared Patterns

### Fail-fast validation with explicit errors
**Source:** `survarena/data/splitters.py` (lines 138-145, 175-176, 218-243, 255-271)  
**Apply to:** split-contract and profile-governance checks in runner/splitters
```python
if len(seeds) < repeats:
    raise ValueError(f"Need at least {repeats} seeds for repeated nested CV, but received {len(seeds)}.")
...
if np.intersect1d(train_idx, test_idx).size > 0:
    raise ValueError(f"Train/test overlap detected for {split.split_id}")
```

### Resume filtering via key set + status gate
**Source:** `survarena/benchmark/runner.py` (lines 417-433, 499-501)  
**Apply to:** resume eligibility logic
```python
if row.get("status") == "success":
    completed_keys.add((dataset_id, method_id, split_id, seed))
...
if key in completed_keys:
    continue
```

### Structured failure records instead of hard crash
**Source:** `survarena/benchmark/runner.py` (lines 212-251)  
**Apply to:** per-split evaluation failure and retry bookkeeping
```python
except Exception as exc:  # noqa: BLE001
    run_payload = {
        "manifest": manifest.to_dict(),
        "metrics": {"status": "failed", "failure_type": type(exc).__name__},
        "failure": {"traceback": traceback.format_exc()},
    }
```

### File export through logging tracker helpers
**Source:** `survarena/logging/export.py` (lines 325, 349-391)  
**Apply to:** any new run-ledger/resume metadata persistence
```python
write_jsonl_gz(output, normalized_records)
write_json(index_output, {...})
write_json(compact_index_output, {...})
```

### Test style: monkeypatch + tmp_path + focused asserts
**Source:** `tests/test_compare_api.py`, `tests/test_io_config.py`, `tests/test_robustness_tracks.py`  
**Apply to:** new deterministic/resume tests
```python
def test_xxx(tmp_path: Path, monkeypatch) -> None:
    ...
    with pytest.raises(ValueError, match="..."):
        ...
```

## No Analog Found

None. All identified phase files have close analogs in current code.

## Metadata

**Analog search scope:** `survarena/benchmark/`, `survarena/data/`, `survarena/logging/`, `survarena/`, `configs/benchmark/`, `tests/`  
**Files scanned:** 13  
**Pattern extraction date:** 2026-04-23
