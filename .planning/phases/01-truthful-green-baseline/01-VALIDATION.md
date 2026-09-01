---
phase: 1
slug: truthful-green-baseline
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-08-31
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x, Ruff 0.15.x, mypy 2.3.1 |
| **Config file** | `pyproject.toml` |
| **Quick run command** | `.venv/bin/python -m pytest -q <targeted-node-ids>` |
| **Full suite command** | `.venv/bin/python -m pytest -q` |
| **Estimated runtime** | Targeted 5–20 seconds; current full suite about 31 seconds |

## Sampling Rate

- **After every task commit:** Run the task's targeted pytest/lint/type command.
- **After every plan wave:** Run affected suites plus Ruff and the declared mypy scope.
- **Before phase verification:** Full suite, compileall, strict audit, and clean-root CoxPH smoke must be green.
- **Max feedback latency:** 60 seconds for per-task automated feedback.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | BASE-01 | T-01-01 | Test changes cannot hide failed benchmark units | unit | `.venv/bin/python -m pytest -q tests/test_benchmark.py::test_model_artifact_request_fails_if_fitted_state_is_not_pickleable tests/test_benchmark.py::test_exec04_resume_preserves_successful_outputs` | ✅ | ✅ passed |
| 1-01-02 | 01 | 1 | BASE-01 | T-01-02 | Process behavior is tested without shared-memory assumptions | unit/integration | `.venv/bin/python -m pytest -q tests/test_benchmark.py::test_benchmark_execution_uses_configured_n_jobs` | ✅ | ✅ passed |
| 1-01-03 | 01 | 1 | BASE-01 | Optional dependency branches are deterministic | unit | `.venv/bin/python -m pytest -q tests/test_core_results.py` | ✅ | ✅ passed |
| 1-01-04 | 01 | 1 | BASE-01 | Type gate is scoped and repeatable | static | `.venv/bin/python -m mypy <declared-scope>` | ✅ | ✅ passed |
| 1-02-01 | 02 | 1 | BASE-02 | Strict audit cannot bypass blockers through renderer failure | unit | `.venv/bin/python -m pytest -q tests/test_manuscript_report.py` | ✅ | ✅ passed |
| 1-02-02 | 02 | 1 | BASE-03 | Cache changes remain explicit and fail closed | unit | `.venv/bin/python -m pytest -q tests/test_benchmark.py -k 'manifest_mismatch'` | ✅ | ✅ passed |
| 1-03-01 | 03 | 2 | BASE-04 | Documentation cannot label invalidated evidence current | source/command | `.venv/bin/python scripts/audit_manuscript_publishability.py --strict` | ✅ | ✅ passed (expected exit 2) |
| 1-03-02 | 03 | 2 | BASE-01 | All baseline gates pass together | full/E2E | `.venv/bin/python -m pytest -q && .venv/bin/python -m compileall -q survarena` | ✅ | ✅ passed |

## Wave 0 Requirements

- [x] Add `mypy==2.3.1` to the development extra and install the updated editable development environment.
- [x] Add `[tool.mypy]` configuration and a documented module allow-list.
- [x] Add or extend audit/cache fixtures as required by Plans 01-02 and 01-03.

## Manual-Only Verifications

All phase behaviors have automated verification. Final documentation wording receives source review but must be backed by the captured command results.

## Validation Sign-Off

- [x] All tasks have automated verify commands or Wave 0 dependencies.
- [x] Sampling continuity has no three consecutive tasks without automated verification.
- [x] Wave 0 identifies all missing tooling/fixtures.
- [x] No watch-mode flags are used.
- [x] Per-task feedback target is under 60 seconds.
- [x] `nyquist_compliant: true` is set.

**Approval:** approved 2026-08-31
