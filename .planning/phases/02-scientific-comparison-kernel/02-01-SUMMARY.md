---
phase: 02-scientific-comparison-kernel
plan: "01"
subsystem: scientific-comparison-population
tags: [pairing, common-support, ranking, elo, reliability]
requires: [01-truthful-green-baseline]
provides:
  - Exact natural-cell validation and complete common-support populations
  - Dataset-level ranks and matched win rates
  - Equal-weight dataset Elo ratings and separate coverage diagnostics
affects: [02-02, 02-03, phase-4-canonical-results]
tech-stack:
  added: []
  patterns: [fail-closed-cell-identity, complete-common-support, dataset-first-aggregation]
key-files:
  created:
    - survarena/evaluation/_comparison.py
    - survarena/evaluation/_eligibility.py
    - tests/test_scientific_comparison.py
  modified:
    - survarena/evaluation/_ranking.py
    - survarena/evaluation/_ratings.py
    - survarena/evaluation/statistics.py
key-decisions:
  - Headline comparison defaults to complete common support; available-case analysis requires an explicit diagnostic policy.
  - Duplicate method/cell identities are errors and pairwise wins use exact one-to-one key joins.
  - Folds and seeds are aggregated within dataset before ranks or Elo matches so datasets receive equal weight.
requirements-completed: [STAT-01, STAT-02, STAT-04, STAT-06, STAT-07]
duration: 6 min
completed: 2026-09-01
---

# Phase 2 Plan 01: Exact Comparison Population Summary

Exact cell joins, complete common support, dataset-first ranks/ratings, and machine-queryable reliability outcomes now replace Cartesian comparison and pooled-fold behavior.

## Accomplishments

- Added one comparison-population contract that validates identity uniqueness, finite eligible metrics, method rosters, expected cells, and complete dataset support.
- Replaced within-dataset score outer products with exact one-to-one cell joins; the two-split regression reports two comparisons rather than four.
- Ranks now operate on one score per dataset/method and stay on the 1..K scale.
- Elo point estimates and bootstrap populations use one match per dataset/method pair; fold replication cannot increase a dataset's weight.
- Coverage separates attempts, success, failure, timeout, invalid prediction, fallback, eligible, missing, and ineligible cells.

## Task Commits

| Task | Commit | Description |
|---|---|---|
| RED | `6429aa3` | Add failing comparison-population counterexamples |
| 1 | `0b285bb` | Define exact comparison population and eligibility contract |
| 2 | `bcdd8d9` | Rank datasets and compare exact matched cells |
| 3 | `3d3206c`, `2a179b4` | Weight Elo by dataset and expose support diagnostics |

## Verification

- Focused scientific comparison suite: **6 passed**.
- Existing evaluation plus Elo identity suites: **29 passed**.
- Combined affected suite: **35 passed**.
- Targeted Ruff: **passed**.

## Deviations from Plan

### [Rule 3 - Scope preservation] Isolate new counterexamples from a dirty shared test file

- `tests/test_evaluation.py` already contained broad uncommitted user work. Added `tests/test_scientific_comparison.py` for the new RED/acceptance fixtures so unrelated edits were neither overwritten nor staged.

### [Rule 3 - Historical identity] Ignore unrelated legacy `02-01` commits

- Git history contains an older milestone's `02-01` governance commits, but their paths and dates predate this plan. The safe-resume check found no partial current-plan implementation, so execution continued without altering those commits.

**Total deviations:** 2 scope/recovery deviations. **Impact:** Scientific scope is complete and pre-existing work remains preserved.

## Next Plan Readiness

Ready for Plan 02-02: dataset-level pairwise inference, equal-weight bootstrap, and complete-block Friedman/Nemenyi can consume the authoritative comparison population.

## Self-Check: PASSED

- Every Plan 02-01 acceptance criterion is covered by an executed regression fixture.
- Source and tests are committed atomically.
- No historical benchmark result was regenerated or promoted.
