---
phase: 02-scientific-comparison-kernel
audited: 2026-09-01
status: passed
threats_closed: 8
threats_open_in_scope: 0
---

# Phase 2 Security and Integrity Audit

## Scope

Scientific-integrity and local execution boundaries introduced by comparison population construction, statistical inference, prediction validation, metric scoring, and runner wiring.

## Threat Results

| Threat | Severity | Mitigation | Status |
|---|---|---|---|
| Duplicate/ambiguous comparison identity creates false matches | HIGH | Unique natural key validation and one-to-one joins | CLOSED |
| Missing/failing methods improve headline performance | HIGH | Complete common support plus separate reliability view | CLOSED |
| Fold pseudoreplication creates anti-conservative p-values | HIGH | One paired aggregate per dataset | CLOSED |
| Incomplete blocks produce invalid post-hoc claims | HIGH | Rectangular support, Friedman gate, NaN CD on invalid assumptions | CLOSED |
| Malformed/nonfinite model output produces plausible metrics | HIGH | Reason-coded validate-before-score boundary | CLOSED |
| Silent horizon clipping changes the estimand | HIGH | Fixed requested horizons; unsupported values remain missing | CLOSED |
| Wrong IPCW boundary scores unsupported observations | HIGH | Fitted censoring-distribution support and recorded bounds | CLOSED |
| Risk-only/fallback output leaks into curve metrics | MEDIUM | Explicit survival capability and metric-family gating | CLOSED |

## Abuse and Failure Properties

- Prediction payloads are converted to bounded-shape NumPy arrays and validated before torch/scikit metric calls.
- Validation never sorts, clips, transposes, fills, or otherwise launders invalid model output.
- Failure evidence contains stable reason codes and no raw dataset rows.
- Support digests use SHA-256 over deterministic exact-cell identity payloads.
- Statistical helpers reject duplicate/null identities and unknown policy values.
- Historical caches/results are fail-closed and were not overwritten during verification.

## Deferred Boundaries

Killable workers/resource ceilings, group-safe nested validation, conflict-safe database writes, and serialized artifact trust remain Phase 3/4 scope. They are release blockers, but no new Phase 2 interface weakens those boundaries.

## Verdict

PASS. No open security or scientific-integrity threat remains inside the Phase 2 implementation boundary.
