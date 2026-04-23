# Phase 2: Fair Dual-Mode HPO Governance - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-23
**Phase:** 02-fair-dual-mode-hpo-governance
**Areas discussed:** Dual-mode run contract

---

## Dual-mode run contract

| Option | Description | Selected |
|--------|-------------|----------|
| Pair on the same dataset/split/seed unit | Strict fairness: compare modes under identical data slices | ✓ |
| Pair only at dataset level | Modes may differ by split/seed and aggregate later | |
| Allow mixed availability | Keep partial overlap when one mode is missing | |

**User's choice:** Pair on the same dataset/split/seed unit.
**Notes:** Fairness should be enforced at the finest benchmark pairing unit, not just at aggregate level.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Single artifacts with explicit mode field | Keep one canonical collection and label each row with mode | ✓ |
| Separate artifact sets by mode | Write separate no-HPO and HPO outputs | |
| HPO-only primary outputs | Keep no-HPO as optional diagnostics | |

**User's choice:** Single artifacts with explicit mode field.
**Notes:** Canonical output should stay unified, with clear mode lineage.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Hard parity gate | Comparison-ineligible unless both modes exist for a unit | ✓ |
| Soft warning + keep partial rows | Keep partial comparisons with caveats | |
| Auto-fill fallback from available mode | Fill missing side from whichever mode exists | |

**User's choice:** Hard parity gate.
**Notes:** Comparative claims should not include unmatched no-HPO/HPO units.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Deterministic sequential per pair | Run no-HPO then HPO per dataset/method/split/seed | ✓ |
| Mode-batched passes | Run all no-HPO then all HPO | |
| Free interleaving | Throughput-first interleaving of both modes | |

**User's choice:** Deterministic sequential per pair.
**Notes:** Execution order should remain reproducible and stable.

---

## Claude's Discretion

- Exact field names for mode labeling and parity eligibility flags.
- Internal control-flow decomposition in runner/export modules.

## Deferred Ideas

None.
