---
phase: 02-scientific-comparison-kernel
reviewed: 2026-09-01
verdict: PASS
open_phase_findings: 0
fixed_findings: 3
---

# Phase 2 Code Review

## Verdict

PASS. The Phase 2 comparison and metric kernel is internally coherent, fail-closed on its scientific assumptions, and covered by adversarial counterexamples. No HIGH or MEDIUM finding remains in Phase 2 scope.

This verdict does not make the benchmark release publishable: process isolation, nested group safety, conflict-safe persistence, canonical reporting, protocol pilot evidence, and clean-environment reproduction remain Phase 3–6 blockers.

## Findings Fixed During Review

### [P1] Duplicate fixed horizons crashed otherwise valid runs

- **Cause:** tied event-time quantiles were forwarded as duplicate `new_time` values, which torchsurv rejects.
- **Fix:** calculate unique supported horizons once, then map AUC/Brier/status/weights back to the original 25/50/75 estimands without changing their requested values.
- **Evidence:** `test_duplicate_fixed_horizons_are_scored_without_changing_estimand`.
- **Commit:** `6147695`.

### [P1] Invalid Friedman/Nemenyi assumptions still exposed a numeric CD

- **Cause:** the table correctly marked `posthoc_eligible=false` but retained a numeric critical difference for compatibility, which a downstream consumer could misinterpret.
- **Fix:** emit `critical_difference=NaN` unless K/N assumptions hold and the Friedman omnibus permits post-hoc analysis.
- **Evidence:** two-method and incomplete-block fixtures fail closed; a complete four-dataset/three-method fixture emits a finite CD only after a significant omnibus.
- **Commit:** `4279703`.

### [P2] Support digest did not identify exact comparison cells

- **Cause:** the digest encoded dataset/method coverage but not split/seed/scenario cell identities.
- **Fix:** hash the sorted exact natural cell identities, roster, metric, and support policy.
- **Evidence:** changing only `split_id` changes the digest.
- **Commit:** `6147695`.

## Reviewed Invariants

- Comparison identity is unique and exact before pairing.
- Complete common support is the headline default.
- Dataset is the independent inferential and weighting unit.
- Multiple-method post-hoc analysis is rectangular-block and omnibus gated.
- Prediction validation precedes all scoring and performs no repair.
- Fixed dataset horizons remain unchanged when unsupported.
- Risk-only and invalid outputs cannot populate curve metrics.
- Every metric family has an independent differential/golden check.

## Deferred Cross-Phase Risks

| Priority | Risk | Owning phase |
|---|---|---|
| P1 | Native fit timeout is thread-based and cannot kill the task/process tree. | Phase 3 |
| P1 | Group separation is not yet enforced through every nested validation/tuning path. | Phase 3 |
| P1 | SQLite `ResultStore` does not yet persist metric-support provenance or reject conflicting reruns. | Phase 4 |
| P1 | Live execution still assembles legacy flat dictionaries before typed canonical storage. | Phase 4 |
| P2 | Survival-distribution capability is a base-adapter flag rather than a versioned core capability field. | Phase 4 |
| P1 | Historical split caches require deliberate regeneration under the strengthened fingerprints. | Phase 6 |

## Conclusion

Phase 2 logic is correct for its declared contracts. Remaining risks are explicit roadmap work, not hidden exceptions to Phase 2 claims.
