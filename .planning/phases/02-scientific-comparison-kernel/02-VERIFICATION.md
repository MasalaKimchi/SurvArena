---
phase: 02-scientific-comparison-kernel
verified: 2026-09-01
status: passed
score: 12/12
---

# Phase 2 Verification

## Result

PASSED — all 12 Phase 2 requirements are implemented and have executable evidence.

## Requirement Evidence

| Requirement | Status | Evidence |
|---|---|---|
| STAT-01 | PASS | Exact one-to-one natural-cell join; Cartesian counterexample passes. |
| STAT-02 | PASS | Dataset/method aggregation precedes 1..K ranks. |
| STAT-03 | PASS | Wilcoxon consumes one paired delta per dataset and reports `n_datasets`. |
| STAT-04 | PASS | Complete common-support population is the headline default. |
| STAT-05 | PASS | Friedman/Nemenyi uses a rectangular block; CD is missing unless assumptions and omnibus pass. |
| STAT-06 | PASS | Coverage separately reports attempts, success, failure, timeout, fallback, invalid, and eligible cells. |
| STAT-07 | PASS | Rank, win-rate, bootstrap, effects, and Elo use equal dataset weighting. |
| METR-01 | PASS | Torchsurv outputs match scikit-survival references within 1e-6. |
| METR-02 | PASS | Result rows include requested/used horizon, IPCW support, window, eligibility, reason, and policy. |
| METR-03 | PASS | Shape/finiteness/grid/bounds/monotonicity/capability validator is reason-coded and fail-closed. |
| METR-04 | PASS | Fixed horizons never clip; calibration matches a SciPy optimizer; censored D-calibration is emitted for valid curves. |
| METR-05 | PASS | Risk-only path emits risk metrics only; invalid/full-distribution-ineligible outputs cannot enter curve metrics. |

## Automated Evidence

- Clean committed snapshot: **286 passed, 6 skipped**.
- Final workspace integration snapshot (includes unrelated user tests): **293 passed, 6 skipped**; affected post-review suites: **103 passed** plus dedicated post-hoc fixtures.
- Clean snapshot Ruff: **passed**.
- Clean snapshot compileall: **passed**.
- Clean snapshot bounded mypy: **11 source files, no issues**.
- WHAS500/CoxPH semantic smoke: **5/5 successful**, fixed horizons `(21.0, 166.0, 613.0)`, finite Harrell/Uno, all full-distribution eligible.

## Independent Reference Coverage

- Harrell concordance: scikit-survival, including tied risks.
- Uno concordance: scikit-survival, including heavy censoring.
- Brier and integrated Brier: scikit-survival.
- Cumulative/dynamic AUC: scikit-survival.
- Calibration intercept/slope: independent SciPy BFGS optimizer.
- D-calibration: censored redistribution plus chi-square contract fixtures.

## Manual/Operational Evidence

- The CLI refused to overwrite legacy split caches whose manifests lacked the strengthened time/feature fingerprints.
- The semantic smoke used an isolated `/tmp` split root; no historical result matrix or split cache was regenerated.
- Exact support identities are hashed, so different cell populations cannot share a support digest.

## Release Boundary

Phase 2 is verified. The overall benchmark is still **not citable/publishable** until Phases 3–6 close execution safety, canonical storage/reporting, representative pilot, and clean release reproduction.
