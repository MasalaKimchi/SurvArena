# Phase 1: Deterministic Execution Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-23
**Phase:** 01-deterministic-execution-foundation
**Areas discussed:** Benchmark methodology positioning, Profile contract, Determinism policy, Resume policy

---

## Benchmark methodology positioning

| Option | Description | Selected |
|--------|-------------|----------|
| TabArena-inspired methodology with SurvArena protocol canonical | Align philosophy to TabArena while keeping SurvArena definitions as implementation source of truth | ✓ |
| Mirror TabArena protocol directly | Follow TabArena process exactly | |
| Independent SurvArena-only methodology | No explicit TabArena framing | |

**User's choice:** TabArena-inspired methodology with SurvArena protocol canonical  
**Notes:** User explicitly confirmed this framing.

---

## Profile contract

| Option | Description | Selected |
|--------|-------------|----------|
| Strict locked profiles | Fixed split/seed/metric defaults and comparable behavior | ✓ |
| Strict baseline + explicit override mode | Allow override runs marked non-comparable | |
| Loose templates | Freely change settings without comparability guardrails | |

**User's choice:** Strict locked profiles  
**Notes:** User chose strongest comparability contract.

| Option | Description | Selected |
|--------|-------------|----------|
| Canonical intent mapping | Smoke=health check, Standard=iterative research, Manuscript=publication-grade claims | ✓ |
| Balanced mapping | Manuscript mainly broadens coverage with similar rigor to standard | |
| Custom mapping | User-defined semantics | |

**User's choice:** Canonical intent mapping  
**Notes:** Profile purposes locked to explicit quality tiers.

---

## Determinism policy

| Option | Description | Selected |
|--------|-------------|----------|
| Hard fail on manifest mismatch | Require explicit regeneration | ✓ |
| Auto-regenerate with audit log | Regenerate automatically and log | |
| Best-effort reuse | Reuse old splits where possible | |

**User's choice:** Hard fail on manifest mismatch  
**Notes:** Reproducibility prioritized over convenience.

| Option | Description | Selected |
|--------|-------------|----------|
| Strict seed propagation | Missing seed handling is an error | ✓ |
| Strict with warning | Missing seed allowed with warning | |
| Relaxed seed handling | Best-effort seed propagation | |

**User's choice:** Strict seed propagation  
**Notes:** Determinism contract should reject unseeded stochastic behavior.

---

## Resume policy

| Option | Description | Selected |
|--------|-------------|----------|
| Success-only completion | Only `status=success` with valid required outputs counts as complete | ✓ |
| Success + partial | Allow selected partial outputs | |
| Any existing row | Existence alone marks complete | |

**User's choice:** Success-only completion  
**Notes:** Resume must avoid treating partial/invalid outputs as finished work.

| Option | Description | Selected |
|--------|-------------|----------|
| Retry by configured policy | Retry only within retry budget; keep final failures | ✓ |
| Always retry | Retry all failures on resume | |
| Never retry | Preserve failures only | |

**User's choice:** Retry by configured policy  
**Notes:** Balances resilience and reproducibility.

---

## Claude's Discretion

- Exact implementation shape for strict output-validation checks used by resume gating.
- Internal refactoring boundaries while preserving current CLI/API behavior.

## Deferred Ideas

None.
