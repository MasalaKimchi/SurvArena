---
phase: 02-scientific-comparison-kernel
plan: "03"
subsystem: survival-metric-contracts
tags: [prediction-validation, ipcw, fixed-horizons, differential-testing, runner]
requires: [02-01, 02-02]
provides:
  - Reason-coded fail-closed validation for risk and survival predictions
  - Fixed dataset horizons with explicit censoring-support eligibility
  - Independently checked survival metrics and censored D-calibration
  - Runner enforcement for invalid and risk-only model outputs
affects: [phase-3-execution-safety, phase-4-canonical-results, phase-5-pilot]
tech-stack:
  added: []
  patterns: [validate-before-score, fixed-estimand-provenance, independent-differential-reference]
key-files:
  created:
    - survarena/evaluation/predictions.py
    - tests/test_metric_contracts.py
  modified:
    - survarena/evaluation/metrics.py
    - survarena/benchmark/runner.py
    - survarena/core/results/schema.py
    - survarena/methods/base.py
    - tests/test_evaluation.py
key-decisions:
  - Invalid prediction tensors are never repaired; they fail with stable machine-readable reason codes.
  - Requested horizons are dataset-level constants and unsupported values remain requested but unscored.
  - IPCW support follows the fitted censoring distribution and every output records support/window provenance.
  - Risk-only methods keep risk metrics while full-distribution eligibility and metrics remain false/missing.
requirements-completed: [METR-01, METR-02, METR-03, METR-04, METR-05]
duration: 10 min
completed: 2026-09-01
---

# Phase 2 Plan 03: Prediction and Metric Contract Summary

The benchmark now validates every prediction bundle before scoring, fixes requested horizons at dataset scope, exposes censoring support without silent clipping, and differentially checks release metrics against an independent implementation.

## Accomplishments

- Added reason-coded validation for risk shape/count/finiteness and survival shape/count/grid/finiteness/bounds/monotonicity/capability.
- Replaced maximum-event-time truncation with censoring-distribution support and explicit per-horizon eligibility/reason fields.
- Preserved requested horizons exactly; unsupported Brier/AUC/calibration values are NaN rather than scores at substituted times.
- Added censored D-calibration and independent scikit-survival references for Harrell C, Uno C, Brier, IBS, and cumulative/dynamic AUC, plus an independent SciPy calibration optimizer.
- Computes dataset horizons once before run-unit scheduling; every fold/method receives the same tuple.
- Invalid predictions produce failed, comparison-ineligible rows with `invalid_prediction:<reason>` evidence; risk-only methods retain Harrell/Uno but cannot emit curve metrics.
- Extended the immutable result adapter to preserve compact metric-support provenance.

## Task Commits

| Task | Commit | Description |
|---|---|---|
| RED | `eb6bdf7` | Add failing prediction-contract fixtures |
| 1 | `f9937f9` | Implement fail-closed prediction validation |
| 2 | `eb224b1` | Correct IPCW support, fixed horizons, D-calibration, and independent references |
| 3 | `ff860a5` | Enforce validation, fixed horizons, capabilities, and provenance in the runner |

## Verification

- Phase-affected integration suite: **116 passed**.
- Full repository suite: **290 passed, 6 skipped**.
- Workspace Ruff, compileall, and bounded mypy (11 source files): **passed**.
- Independent differential tolerance: **1e-6** for Harrell C, Uno C, Brier, IBS, and time-dependent AUC.
- WHAS500/CoxPH semantic runner smoke: **5/5 successful**, fixed horizons `(21.0, 166.0, 613.0)`, finite Uno/Harrell in every fold, full-distribution eligible in every fold.
- Historical split cache protection: CLI refused to overwrite legacy manifests missing fingerprints; semantic smoke used a temporary split root.

## Deviations from Plan

### [Rule 3 - Scope preservation] Isolate new contract/reference fixtures

- New runner and metric fixtures live in `tests/test_metric_contracts.py` so broad uncommitted user additions in `tests/test_benchmark.py` and `tests/test_evaluation.py` remain unstaged. Only the obsolete clipped-horizon assertion was partially staged from the existing evaluation file.

### [Rule 2 - Capability bridge] Add a lightweight adapter flag without expanding the core capability dataclass

- Existing adapters already implement survival prediction and core capability tests freeze the current six-field record. `BaseSurvivalMethod.supports_survival_distribution` provides the runner gate now; a future schema migration can merge it into the core contract without breaking Phase 2 behavior.

**Total deviations:** 2 compatibility/scope deviations. **Impact:** All planned metric and runner truths are enforced with no historical split/result mutation.

## Next Phase Readiness

Phase 2 implementation is complete. Phase 3 can now replace thread timeouts, harden group-aware nested validation, enforce conflict-safe persistence, and finish execution/data safety on top of a scientifically valid scoring kernel.

## Self-Check: PASSED

- Invalid outputs cannot be scored.
- Requested horizons cannot silently move.
- Independent references cover every release metric family.
- Source/tests are committed and the full suite is green.
