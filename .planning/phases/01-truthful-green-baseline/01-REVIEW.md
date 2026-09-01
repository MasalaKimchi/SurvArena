---
phase: 01-truthful-green-baseline
reviewed: 2026-09-01T04:08:52Z
depth: standard
files_reviewed: 28
files_reviewed_list:
  - .github/workflows/benchmark-smoke.yml
  - .github/workflows/ci.yml
  - PROJECT_STATE.md
  - README.md
  - docs/ci_and_reproducibility.md
  - docs/test_status.md
  - pyproject.toml
  - scripts/audit_manuscript_publishability.py
  - survarena/__init__.py
  - survarena/benchmark/hpo_config.py
  - survarena/benchmark/resume.py
  - survarena/benchmark/runner.py
  - survarena/core/__init__.py
  - survarena/core/models/__init__.py
  - survarena/core/models/contract.py
  - survarena/core/protocol/__init__.py
  - survarena/core/protocol/spec.py
  - survarena/core/results/__init__.py
  - survarena/core/results/schema.py
  - survarena/core/results/store.py
  - survarena/data/splitters.py
  - survarena/methods/base.py
  - tests/test_benchmark.py
  - tests/test_core_models.py
  - tests/test_core_protocol.py
  - tests/test_core_results.py
  - tests/test_manuscript_report.py
  - tests/test_phase1_identity.py
findings:
  critical: 0
  warning: 6
  info: 1
  total: 7
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-09-01T04:08:52Z
**Depth:** standard
**Files Reviewed:** 28
**Status:** issues_found

## Summary

The Phase 1 baseline is materially stronger and its verification claims are appropriately fail-closed. One shipping blocker was found during review: dual no-HPO/HPO runs shared a run/manifest ID even though their artifact paths differed. That defect was fixed in `7b3c7fd`, with arm-qualified identity and regression coverage, before this report was finalized.

The remaining findings are real but align with work explicitly assigned to later v2 phases. They prevent SurvArena from being called a comprehensive or citable benchmark today; they do not invalidate Phase 1's narrower claim of a truthful green development baseline.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: A timed-out fit continues running and can keep the worker alive

**File:** `survarena/benchmark/runner.py:347-367`

**Issue:** `Future.result(timeout=...)` stops waiting, but `shutdown(wait=False)` cannot terminate a running thread. The fit continues mutating model/RNG state, consuming CPU/GPU/RAM, and Python waits for non-daemon executor threads during interpreter shutdown. A genuinely hung native fit can therefore outlive its failure row or stall the process, so this is not a hard wall-clock budget.

**Fix:** Execute each fit in a spawned subprocess/process group, return results through IPC, and terminate then kill the complete descendant tree on timeout. Record requested and observed termination telemetry. This is the Phase 3 killable-execution boundary.

### WR-02: Group safety stops at the outer split

**File:** `survarena/benchmark/runner.py:257-304`

**Issue:** The final-fit validation holdout uses row-level `train_test_split`, and the HPO cache is prepared without subject groups. For repeated-subject datasets, a subject can appear in both inner train and validation data even though the outer split is group-disjoint. That leaks subject-specific information into early stopping and hyperparameter selection.

**Fix:** Carry groups in `BenchmarkRunUnit`, construct group-disjoint inner folds and holdouts, and assert zero group overlap at every outer, inner, and final-fit boundary. Include adversarial repeated-subject fixtures.

### WR-03: The committed adapters do not declare the new capability contract

**File:** `survarena/methods/base.py:20-51`

**Issue:** The committed tree defines `consumes_validation=False` on the base class, but no concrete adapter in `HEAD` overrides it. Consequently, the new runner validation branch is dead for real adapters in a clean checkout, including early-stopping neural and boosting methods. The contract tests only exercise dummy classes.

**Fix:** Give every registered adapter an explicit capability declaration and add a registry-wide conformance test that checks declaration completeness, fit signature, prediction shapes, survival monotonicity, determinism, and early-stopping/refit semantics.

### WR-04: Strict protocol parsing silently turns string `"false"` into `True`

**File:** `survarena/core/protocol/spec.py:161-206,237-255`

**Issue:** Boolean fields are parsed with Python `bool(value)`. Non-empty strings such as `"false"`, `"0"`, and `"no"` become `True`, and `ProtocolSpec.validate()` does not validate nested boolean/numeric types. `from_mapping(..., strict=True)` therefore accepts a configuration whose execution semantics are the opposite of what its text says.

**Fix:** Preserve invalid raw values on the lenient path and add typed nested validation, or use an explicit boolean parser accepting only actual booleans (and, if desired, a documented exact string set). Add strict negative tests for every nested scalar field.

### WR-05: Result conflicts are silently replaced

**File:** `survarena/core/results/store.py:175-245`

**Issue:** `append()` intentionally uses last-write-wins deduplication plus `INSERT OR REPLACE`. Two shards can emit the same natural key with different metrics, dataset/method versions, environment fingerprints, or status, and the later append silently destroys the earlier evidence. That is unsafe for a canonical benchmark collection.

**Fix:** Make identical re-appends idempotent, but reject non-identical rows for an existing natural key with a structured conflict error and audit record. Include recipe, dataset, split, method, and environment digests in the comparison. Phase 4 should make this store authoritative only after conflict rejection exists.

### WR-06: Split content fingerprints can collide after lossy normalization

**File:** `survarena/data/splitters.py:66-109`

**Issue:** DataFrame numeric columns are coerced to `float64`, so distinct `int64` values above `2**53` can hash identically. Object values are concatenated with unescaped sentinel/separator characters and lose original types, so different sequences can also produce the same byte stream. A collision can reuse positional split indices for changed data.

**Fix:** Hash a length-prefixed, type-tagged canonical encoding that preserves each column's dtype and exact values; include the group vector explicitly. Add collision-oriented tests for large integers, delimiter-containing strings, missing values, categorical values, and changed group assignments.

## Info

### IN-01: CI actions are version-tagged rather than digest-pinned

**File:** `.github/workflows/ci.yml:28-29`

**Issue:** `actions/checkout@v4`, `actions/setup-python@v5`, and the artifact action use mutable major-version tags. This is acceptable for the current development baseline but is not a frozen release supply chain.

**Fix:** Pin action commit SHAs and record/update them through an audited dependency process in Phase 6.

---

_Reviewed: 2026-09-01T04:08:52Z_
_Reviewer: Codex (inline gsd-code-reviewer fallback)_
_Depth: standard_
