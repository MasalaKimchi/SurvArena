---
phase: 02
slug: fair-dual-mode-hpo-governance
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-23
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.x |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `pytest tests/test_hpo_config.py` |
| **Full suite command** | `pytest` |
| **Estimated runtime** | ~180 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_hpo_config.py`
- **After every plan wave:** Run `pytest`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | EXEC-02 | — | Dual-mode run records must preserve strict dataset/split/seed pairing without mode mixing | integration | `pytest tests/test_hpo_config.py` | ✅ | ⬜ pending |
| 02-01-02 | 01 | 1 | EXEC-03 | — | Requested and realized HPO budget fields must be emitted per run and auditable | unit | `pytest tests/test_hpo_config.py` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_hpo_config.py` — extend assertions for dual-mode governance and budget telemetry.
- [ ] `tests/test_compare_api.py` — add parity/eligibility coverage for dual-mode comparative summaries.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| End-to-end benchmark fairness narrative in exported artifacts | EXEC-02, EXEC-03 | Requires holistic inspection of generated experiment manifests and summary tables | Run a benchmark profile with dual-mode enabled, then inspect mode-labeled run records and parity-gated comparative summaries in `results/summary/exp_*`. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 180s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
