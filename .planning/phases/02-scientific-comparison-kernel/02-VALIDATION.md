---
phase: 2
slug: scientific-comparison-kernel
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-01
---

# Phase 2 — Validation Strategy

## Test Infrastructure

| Property | Value |
|---|---|
| Framework | pytest, Ruff, bounded mypy |
| Quick command | `.venv/bin/python -m pytest -q tests/test_evaluation.py` |
| Full command | `.venv/bin/python -m pytest -q` |
| Independent metric reference | declared scikit-survival dependency |

## Sampling Rate

- After each task: run the named constructed counterexample or differential fixture.
- After each plan: run all `tests/test_evaluation.py` plus affected benchmark/API tests and Ruff.
- At phase completion: full pytest, Ruff, bounded mypy, compileall, and semantic smoke.

## Per-Task Verification Map

| Task | Requirement | Threat | Automated proof |
|---|---|---|---|
| 02-01-01 | STAT-01, STAT-04, STAT-06 | T-01-01, T-01-02 | duplicate/missing/failed-cell support fixtures |
| 02-01-02 | STAT-01, STAT-02, STAT-07 | T-01-03 | exact two-cell win/rank counterexample |
| 02-01-03 | STAT-06, STAT-07 | T-01-04 | equal-dataset-weight Elo/reliability fixtures |
| 02-02-01 | STAT-03, STAT-07 | T-02-01 | dataset-level Wilcoxon sample-size fixture |
| 02-02-02 | STAT-04, STAT-05 | T-02-02 | incomplete-block rejection and complete-block CD fixture |
| 02-02-03 | STAT-06, STAT-07 | T-02-03 | dataset bootstrap and explicit support metadata fixture |
| 02-03-01 | METR-03, METR-05 | T-03-01 | adversarial prediction bundle fixtures |
| 02-03-02 | METR-01, METR-02, METR-04 | T-03-02, T-03-03 | sksurv differential and calibration/D-calibration fixtures |
| 02-03-03 | METR-02, METR-03, METR-05 | T-03-04 | runner support/provenance and invalid-output tests |

## Wave 0 Requirements

- [x] scikit-survival, SciPy, torchsurv, pytest, and NumPy are already declared and installed.
- [x] Existing evaluation and benchmark fixtures provide the integration seams.
- [x] Phase 1 clean-snapshot and semantic-smoke gates are green.

## Manual Verification

None. Scientific correctness is covered by deterministic counterexamples, independent references, and machine-readable support assertions.

## Sign-Off

- [x] Every task has an automated check.
- [x] No three-task validation gap exists.
- [x] Independent reference coverage is required, not optional/skipped.
- [x] No historical benchmark matrix is treated as validation evidence.

**Approval:** approved 2026-09-01
