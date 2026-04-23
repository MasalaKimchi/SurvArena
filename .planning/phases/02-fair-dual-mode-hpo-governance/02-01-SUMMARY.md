---
phase: 02-fair-dual-mode-hpo-governance
plan: 01
subsystem: testing
tags: [fairness, hpo, parity, governance, pytest]
requires:
  - phase: 01-deterministic-execution-foundation
    provides: deterministic profile and split contract baseline used by new fairness tests
provides:
  - dual-mode parity governance test contracts for runner-ledger behavior
  - hpo budget policy normalization contract tests for parser output semantics
affects: [survarena/benchmark/runner.py, survarena/benchmark/tuning.py, phase-02-implementation]
tech-stack:
  added: []
  patterns: [monkeypatch orchestration doubles, contract-first RED test gating]
key-files:
  created:
    - tests/test_dual_mode_hpo_governance.py
    - tests/test_hpo_config.py
  modified: []
key-decisions:
  - "Phase 02-01 remains RED-focused: encode fairness and budget governance in tests before runtime implementation changes."
  - "Dual-mode tests assert runner-level payload contracts using monkeypatch seams instead of expensive full model training."
patterns-established:
  - "Contract tests for run-ledger governance should target run_benchmark orchestration seams and exported run-record rows."
  - "HPO config contract tests should assert key presence and coercion behavior for deterministic policy stability."
requirements-completed: [EXEC-02, EXEC-03]
duration: 2 min
completed: 2026-04-23
---

# Phase 2 Plan 01: Fair Dual-Mode HPO Governance Summary

**Dual-mode parity governance and uniform HPO budget semantics are now encoded as executable contracts, with RED failures proving current runner gaps before Phase 02 runtime changes.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-23T22:54:28Z
- **Completed:** 2026-04-23T22:56:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Added `tests/test_dual_mode_hpo_governance.py` with explicit contracts for `hpo_mode`, `parity_key`, deterministic no-HPO then HPO ordering, and `parity_eligible` gating.
- Extended `tests/test_hpo_config.py` with governance contracts for normalized policy keys and combined `max_trials` plus `timeout_seconds` semantics.
- Verified RED behavior is meaningful: plan-level verification fails on missing `hpo_mode` in runner payloads, creating a clear implementation target for next plans.

## Task Commits

Each task was committed atomically:

1. **Task 1: Author dual-mode pairing and parity-gate contract tests** - `31b7121` (test)
2. **Task 2: Extend HPO config tests for uniform budget policy contract** - `21ebcc2` (test)

**Plan metadata:** Included in final docs commit for this plan summary.

## Files Created/Modified

- `tests/test_dual_mode_hpo_governance.py` - New RED contract suite for dual-mode parity and ineligibility governance.
- `tests/test_hpo_config.py` - New budget policy contract tests for key normalization and deterministic coercion.

## Decisions Made

- Kept this plan as a test-contract phase (RED-first) to lock behavior before runtime implementation modifications.
- Used lightweight monkeypatch-driven doubles around runner orchestration to isolate fairness/governance semantics without introducing integration-runtime flakiness.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed run-ledger capture in test harness**
- **Found during:** Task 1 (Author dual-mode pairing and parity-gate contract tests)
- **Issue:** Mocked `export_run_ledger` attempted to read `run_records` from kwargs, but function call provided positional args.
- **Fix:** Updated the mock to capture positional run-record argument (`_args[1]`), enabling intended RED assertion path.
- **Files modified:** `tests/test_dual_mode_hpo_governance.py`
- **Verification:** `pytest tests/test_dual_mode_hpo_governance.py -x` fails on expected missing governance field (`hpo_mode`) rather than harness error.
- **Committed in:** `31b7121`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Deviation was strictly harness correctness; no scope creep and no requirement changes.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None.

## Next Phase Readiness

- Runtime implementation is now unblocked with explicit failing contracts for D-01, D-03, D-04, and EXEC-03 parser governance.
- Next plan can implement runner/tuning/export behavior to satisfy these tests and convert RED to GREEN.

## Verification Results

- `pytest tests/test_dual_mode_hpo_governance.py -x` -> **FAIL (expected RED)** on missing `metrics.hpo_mode`.
- `pytest tests/test_hpo_config.py -x` -> **PASS** (4 passed).
- `pytest tests/test_dual_mode_hpo_governance.py tests/test_hpo_config.py -x` -> **FAIL (expected RED)** at first dual-mode contract.

## Self-Check: PASSED

- Summary file exists at `.planning/phases/02-fair-dual-mode-hpo-governance/02-01-SUMMARY.md`.
- Task commit `31b7121` found in git history.
- Task commit `21ebcc2` found in git history.

---
*Phase: 02-fair-dual-mode-hpo-governance*
*Completed: 2026-04-23*
