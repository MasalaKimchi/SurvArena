---
phase: 01-deterministic-execution-foundation
reviewed: 2026-04-24T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - survarena/benchmark/runner.py
  - survarena/data/splitters.py
  - survarena/run_benchmark.py
  - survarena/logging/export.py
  - tests/test_benchmark_determinism.py
  - tests/test_benchmark_resume.py
  - configs/benchmark/standard_v1.yaml
findings:
  critical: 0
  warning: 2
  info: 0
  total: 2
status: issues_found
---

# Phase 01: Code Review Report

**Reviewed:** 2026-04-24T00:00:00Z  
**Depth:** standard  
**Files reviewed:** 7 (scoped from `01-01-SUMMARY.md` and `01-02-SUMMARY.md` `key-files`)  
**Status:** issues_found

## Summary

Re-reviewed benchmark orchestration, split governance, CLI entry, export normalization, Phase 01 tests, and `standard_v1` config. **No new issues** since the last Phase 01 report; the two prior warning-level items remain **unresolved** in current `main` and are restated with updated line anchors.

## Warnings

### WR-01: Unknown method IDs fail after method YAML load, not before

**File:** `survarena/benchmark/runner.py` (non–`dry_run` path: `method_cfg_cache` at ~L550, registry check at ~L637–L639)  
**Issue:** Per-method configs are read for every `method_id` in one dict comprehension before the inner loop validates `method_id in registered_methods`. A typo in `methods` that matches no YAML under `configs/methods/` fails with a file or parse error; a name that is not registered but *does* match a file could proceed until evaluation. The intended `Unknown method_id` `ValueError` is only produced inside the dataset/method loop, after dataset and split work has started.  
**Remediation:** Build `method_cfg_cache` only after `unknown_methods = [m for m in methods if m not in registered_methods]` and `if unknown_methods: raise ValueError(...)`; then load YAML for known IDs only (or validate registry membership before any `read_yaml`).

### WR-02: Event fingerprint can collide when labels are not strict binary 0/1

**File:** `survarena/data/splitters.py` — `_event_fingerprint` (~L49–L51)  
**Issue:** Fingerprinting uses `np.asarray(event, dtype=np.int8).tobytes()`. Values outside the intended binary set can truncate or overflow in ways that make distinct label vectors hash the same, weakening manifest mismatch detection. Survival pipelines typically use {0,1} labels; the risk is mainly contract clarity for non-binary or non-standard encodings.  
**Remediation:** Assert binary labels (or hash raw bytes at full precision) before hashing; see previous review for a concrete hardening pattern.

## Info

None.

---

_Reviewer: GSD code-review workflow (standard depth)_  
_Scope: SUMMARY `key-files` only (no extra configs)._
