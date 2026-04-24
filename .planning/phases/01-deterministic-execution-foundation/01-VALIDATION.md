---
phase: 1
slug: deterministic-execution-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-23
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | pyproject.toml |
| **Quick run command** | `pytest tests/test_benchmark_runner.py -x` |
| **Full suite command** | `pytest` |
| **Estimated runtime** | ~180 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_benchmark_runner.py -x`
- **After every plan wave:** Run `pytest`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 0 | EXEC-01 | T-1-01 | Deterministic profile/split contract rejects invalid manifest reuse | unit/integration | `pytest tests/test_benchmark_runner.py -k "not exec04" -x` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 0 | EXEC-04 | T-1-02 | Resume accepts only valid success outputs and preserves structured failure records | unit/integration | `pytest tests/test_benchmark_runner.py -k "exec04" -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_benchmark_runner.py` — EXEC-01 (non-`exec04` tests) and EXEC-04 (`test_exec04_*` tests)
- [ ] `python -m pip install -e ".[dev]"` — ensure local pytest/ruff versions align with project constraints

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Resume from a killed long-running benchmark process in real project environment | EXEC-04 | Requires real interruption/restart behavior beyond unit harness | Start benchmark, interrupt process, re-run with `--resume`, confirm prior successful keys skipped and failures retried only within budget |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
