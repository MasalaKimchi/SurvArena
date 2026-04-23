---
phase: 01-deterministic-execution-foundation
reviewed: 2026-04-23T23:03:31Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - survarena/benchmark/runner.py
  - survarena/data/splitters.py
  - survarena/logging/export.py
  - survarena/run_benchmark.py
  - tests/test_benchmark_determinism.py
  - tests/test_benchmark_resume.py
  - configs/benchmark/standard_v1.yaml
  - configs/benchmark/manuscript_v1.yaml
findings:
  critical: 0
  warning: 2
  info: 0
  total: 2
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-04-23T23:03:31Z  
**Depth:** standard  
**Files Reviewed:** 8  
**Status:** issues_found

## Summary

Reviewed benchmark orchestration, deterministic split management, export logic, CLI entrypoint, targeted tests, and profile configs for Phase 01 scope. No critical security issues were found, but there are two warning-level correctness risks that can cause confusing failure modes or deterministic-contract drift in edge cases.

## Warnings

### WR-01: Unknown method IDs fail before explicit registry validation

**File:** `survarena/benchmark/runner.py:531`  
**Issue:** Method config files are loaded for all configured methods before checking whether each method is registered (`method_cfg_cache` is created at line 531, while registry validation happens later at lines 613-614). If a method ID is invalid, execution can fail with a file-read/config error instead of the intended clear "Unknown method_id" validation path, which weakens diagnostics and can mask root cause.  
**Fix:**
```python
# Validate method IDs before reading per-method config files.
registered_methods = set(registered_method_ids())
unknown_methods = [m for m in methods if m not in registered_methods]
if unknown_methods:
    raise ValueError(f"Unknown method_id(s) {unknown_methods}. Registered: {sorted(registered_methods)}")

method_cfg_cache = {
    method_id: read_yaml(repo_root / "configs" / "methods" / f"{method_id}.yaml")
    for method_id in methods
}
```

### WR-02: Split manifest fingerprint can collide for non-binary event labels

**File:** `survarena/data/splitters.py:49-51`  
**Issue:** `_event_fingerprint()` coerces labels to `np.int8` before hashing. If upstream event values are outside int8 range or not normalized to {0,1}, distinct event arrays can map to the same byte representation after truncation/overflow. That can incorrectly treat changed labels as "same manifest payload", violating deterministic split contract checks in edge cases.  
**Fix:**
```python
def _event_fingerprint(event: np.ndarray) -> str:
    arr = np.asarray(event)
    if not np.isin(arr, [0, 1, False, True]).all():
        raise ValueError("Event labels must be binary (0/1) for deterministic split fingerprinting.")
    encoded = arr.astype(np.uint8, copy=False).tobytes()
    return sha256(encoded).hexdigest()
```

---

_Reviewed: 2026-04-23T23:03:31Z_  
_Reviewer: Claude (gsd-code-reviewer)_  
_Depth: standard_
