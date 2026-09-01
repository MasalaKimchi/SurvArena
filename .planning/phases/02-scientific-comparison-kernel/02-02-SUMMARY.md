---
phase: 02-scientific-comparison-kernel
plan: "02"
subsystem: dataset-level-inference
tags: [wilcoxon, bootstrap, friedman, nemenyi, common-support]
requires: [02-01]
provides:
  - Dataset-level paired effects and inferential sample sizes
  - Equal-weight dataset bootstrap estimates and provenance
  - Complete-block Friedman and guarded Nemenyi summaries
affects: [02-03, phase-4-canonical-results, phase-6-reproduction]
tech-stack:
  added: []
  patterns: [dataset-as-inferential-unit, exact-cell-before-aggregation, explicit-assumption-status]
key-files:
  modified:
    - survarena/evaluation/_significance.py
    - tests/test_scientific_comparison.py
key-decisions:
  - Exact matched cells are reduced to one paired delta per dataset before Wilcoxon testing.
  - Headline bootstrap point estimates and draws weight dataset means equally regardless of fold count.
  - Multiple-method inference uses only complete dataset-method blocks and exposes when assumptions prevent post-hoc claims.
requirements-completed: [STAT-03, STAT-04, STAT-05, STAT-07]
duration: 5 min
completed: 2026-09-01
---

# Phase 2 Plan 02: Dataset-Level Inference Summary

Pairwise tests, uncertainty intervals, and multiple-method summaries now use datasets—not pooled folds—as their experimental units, with exact support and assumption metadata carried into every output.

## Accomplishments

- Pairwise significance performs an exact one-to-one cell join, averages paired deltas within dataset, and reports dataset count, matched-cell count, mean/median effect, wins/ties/losses, and explicit not-testable reasons.
- Repeating folds inside one dataset cannot change a pairwise effect or p-value, and one dataset cannot manufacture a significance claim.
- Bootstrap point estimates and resamples operate on per-dataset means; outputs record dataset/cell counts, support digest, policy, and random seed.
- Critical-difference summaries consume only the complete rectangular block and report total, included, and excluded dataset counts plus Friedman/post-hoc assumption status.

## Task Commits

| Task | Commit | Description |
|---|---|---|
| RED | `82bb225` | Add dataset-inference and incomplete-block counterexamples |
| 1–3 | `d2efec0` | Implement dataset-level pairwise, bootstrap, and complete-block inference |
| 3 | `ff566fc` | Record bootstrap median and seed provenance |

## Verification

- Scientific-comparison plus existing evaluation suites: **34 passed**.
- Bootstrap-specific regression: **1 passed**.
- Targeted Ruff and `git diff --check`: **passed**.

## Deviations from Plan

### [Rule 3 - Scope preservation] Keep new inference fixtures in the isolated Phase 2 test module

- `tests/test_evaluation.py` contains unrelated uncommitted user work, so the new acceptance fixtures remain in `tests/test_scientific_comparison.py` and the existing suite is still executed as a compatibility gate.

### [Rule 2 - Backward-compatible assumption reporting] Retain a descriptive CD for small K/N

- Existing consumers expect a numeric critical-difference field. The implementation retains that descriptive value but sets `posthoc_eligible=false` and an explicit `assumption_status`, preventing it from being interpreted as a valid post-hoc claim.

**Total deviations:** 2 scope/compatibility deviations. **Impact:** Planned statistical validity is preserved without staging unrelated work or breaking descriptive consumers.

## Next Plan Readiness

Ready for Plan 02-03: prediction validation, fixed evaluation horizons, censoring-support metadata, and independent metric-reference tests.

## Self-Check: PASSED

- Pseudoreplication and unequal-fold counterexamples pass.
- Incomplete multiple-method blocks are excluded and disclosed.
- Source and tests are committed atomically.
