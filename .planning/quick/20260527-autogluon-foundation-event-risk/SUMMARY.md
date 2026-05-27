---
status: complete
completed: 2026-05-27
---

# Summary

Implemented AutoGluon event-risk survival adapters for TabICL, TabM, TabDPT, and RealTabPFN-V2, plus method configs and manuscript-grade foundation-only benchmark configs for clinical and genomics tracks.

## Verification

- `pytest` passed: 182 passed, 6 skipped.
- `ruff check survarena tests scripts` passed.
- `python -m compileall survarena` passed.
- Dry-runs passed for the full clinical/genomics manuscript configs and the new foundation-only clinical/genomics configs.
- `python -m pip install -e ".[foundation-tabarena]"` passed after defining direct optional dependencies rather than AutoGluon's broad `tabarena` extra.

## Benchmark Execution Notes

- The full manuscript-grade clinical/genomics foundation runs were not completed in this local interactive session.
- Initial execution showed missing optional packages for TabICL/TabM; direct `foundation-tabarena` installation resolved runtime readiness.
- AutoGluon's broad `tabarena` extra conflicts with SurvArena's `xgboost==3.2.0` pin because AutoGluon 1.5.0 declares `xgboost<3.2` for that extra.
- CPU-only smoke attempts for TabICL and TabDPT/RealTabPFN consumed substantial CPU/RAM and did not complete promptly, so they were stopped rather than leaving runaway jobs.

## Follow-up: Direct Horizon Adapters

- Replaced `tabicl_survival` and `tabdpt_survival` registry targets with direct horizon-classifier adapters using `tabicl.TabICLClassifier` and `tabdpt.TabDPTClassifier`.
- Updated method configs and manuscript overrides to use horizon parameters instead of AutoGluon `time_limit` for TabICL/TabDPT.
- Dry-runs pass for the clinical/genomics foundation configs and the full clinical manuscript config.
- A direct TabICL support smoke was still CPU-bound after roughly two minutes on a full support fold, so it was stopped; the direct path removes AutoGluon overhead but does not make TabICL cheap on CPU.
