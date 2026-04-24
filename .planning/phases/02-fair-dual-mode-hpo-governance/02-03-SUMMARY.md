---
phase: 02-fair-dual-mode-hpo-governance
plan: 03
subsystem: export-and-compare
tags: [parity-gate, compare-api, budget-governance, reporting]
requires:
  - phase: 02-02
    provides: dual-mode runner rows with parity metadata and requested/realized budget fields
provides:
  - parity-gated comparative exports for fairness claims
  - compare API propagation of canonical dual-mode governance fields
  - regression coverage for parity filtering and budget visibility
affects: [survarena/logging/export.py, survarena/api/compare.py, tests/test_compare_api.py]
tech-stack:
  added: []
  patterns:
    - "comparative outputs derive from parity-eligible rows only"
    - "single canonical artifact tree with explicit mode/governance columns"
key-files:
  created: []
  modified:
    - survarena/logging/export.py
    - survarena/api/compare.py
    - tests/test_compare_api.py
key-decisions:
  - "Parity-ineligible rows remain in raw fold/run ledgers but are excluded from rank/significance/ELO claims."
  - "Canonical governance fields are surfaced as requested_* plus realized_trial_count across compare/export flows."
patterns-established:
  - "Compare API executes deterministic no_hpo then hpo passes for each split/method pair."
  - "Manuscript comparison consumes parity-gated source frames before comparative aggregations."
requirements-completed: [EXEC-02, EXEC-03]
duration: 18 min
completed: 2026-04-24
---

# Phase 2 Plan 03: Parity-Gated Comparative Export Summary

**Comparative benchmark claims now exclude parity-ineligible rows while preserving one canonical artifact set with explicit dual-mode and budget-governance metadata.**

## Accomplishments

- Implemented parity gating in `export_manuscript_comparison` so rank, pairwise, significance, CD, CI, and ELO outputs derive only from eligible dual-mode rows.
- Upgraded `compare_survival_models` to run no-HPO and HPO deterministically per split/method, stamping `hpo_mode`, `parity_key`, and requested-vs-realized budget fields.
- Added regression tests proving parity-ineligible rows are excluded from comparative summaries and budget governance fields are present in outputs.

## Task Commits

1. **Task 1: Apply hard parity gate before comparative summary computations** — `3a7219c`
2. **Task 2: Keep canonical artifact set with explicit mode/governance labeling** — `b5d1bad`
3. **Task 3: Add parity/budget regression coverage** — `808b655`

## Deviations from Plan

- The executor subagent could not be used for this wave due runtime API limit errors.
- Executed the same planned changes inline in the orchestrator and preserved atomic task commits.

## Verification Results

- `pytest tests/test_compare_api.py tests/test_dual_mode_hpo_governance.py -x --tb=short` -> **PASS**
- `pytest -x -q --tb=short` -> **PASS** (125 passed, 6 skipped)

## Self-Check: PASSED

- Summary file exists at `.planning/phases/02-fair-dual-mode-hpo-governance/02-03-SUMMARY.md`.
- Task commits are present in git history: `3a7219c`, `b5d1bad`, `808b655`.

---
*Phase: 02-fair-dual-mode-hpo-governance*
*Completed: 2026-04-24*
