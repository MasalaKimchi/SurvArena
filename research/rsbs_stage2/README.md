# RSBS Stage 2 research track

This directory records an exploratory survival-modeling study that began with Kernel Martingale Discrepancy (KMD) and converged on a narrower, better-supported direction:

> flexible event-intensity modeling + a strictly proper reference-scaled score + held-out martingale specification tests.

## Current conclusion

- The largest gains under crossing hazards came from relaxing the proportional-hazards architecture.
- The implemented KMD V/U penalties were around `1e-6` and did not materially affect training.
- A likelihood-anchored **Reference-Scaled Bregman Survival (RSBS)** loss is the next testable objective.
- KMD is retained as a held-out specification diagnostic until covariance normalization and cross-fitting are implemented.
- Time-dependent Uno C should replace the ambiguous label `IPCW-Antolini`; it corrects Antolini's censoring bias but does not measure calibration.

## Files in this branch

- `survarena/methods/deep/rsbs_loss.py`: architecture-agnostic PyTorch loss.
- `survarena/evaluation/concordance_time_dependent.py`: transparent reference implementations of Harrell, Antolini, time-dependent Uno, and conditional pairwise IPCW concordance.
- `tests/test_rsbs_research.py`: mathematical and numerical checks.
- `docs/research_rsbs_stage2.md`: theory, benchmark findings, limitations, and literature boundary.
- `docs/rsbs_confirmatory_protocol.md`: pre-specified Stage 3 benchmark protocol.

## Recommended API use

```python
from survarena.methods.deep.rsbs_loss import hazard_from_logits, rsbs_loss

hazard = hazard_from_logits(network(features))
loss = rsbs_loss(
    hazard,
    exposure,
    events,
    reference_hazard,
    beta=2.0,
    eta=0.1,
)
```

The reference hazard must be training-only and detached from the optimized network. `beta` and `eta` must be selected inside validation folds.

## Status

Exploratory. No universal performance claim is made. The confirmatory benchmark must use matched nested tuning, paired outer folds, censoring-support diagnostics, and corrected uncertainty intervals.
