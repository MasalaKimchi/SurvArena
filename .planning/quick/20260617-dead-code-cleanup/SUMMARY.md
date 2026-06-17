---
status: complete
---

# Quick Task 20260617: Dead Code Cleanup Summary

## Findings

- Ruff unused-name checks were already clean across `survarena`, `tests`, and `scripts`.
- AST duplicate-body/reference scans found several wrapper and duplicate-helper candidates.
- Safe edits were limited to private one-line wrappers with only internal usage:
  - Deep method seed/device pass-through helpers.
  - Benchmark tuning metric-direction pass-through helper.
  - TabPFN private probability pass-through tested only through a class-private hook.

## Files Changed

- `survarena/methods/deep/deepsurv.py`
- `survarena/methods/deep/deepsurv_moco.py`
- `survarena/methods/deep/discrete_hazard.py`
- `survarena/methods/deep/pycox_models.py`
- `survarena/benchmark/tuning.py`
- `survarena/methods/foundation/tabpfn_survival.py`
- `tests/test_foundation.py`

## Validation

- `python -m ruff check survarena/methods/deep/deepsurv.py survarena/methods/deep/deepsurv_moco.py survarena/methods/deep/pycox_models.py survarena/methods/deep/discrete_hazard.py survarena/benchmark/tuning.py` - passed
- `pytest -q tests/test_evaluation.py::test_selection_helpers_strip_runtime_only_defaults` - passed
- `pytest -q 'tests/test_methods.py::test_new_method_adapters_fit_and_emit_survival_curves[deepsurv-params15]' tests/test_methods.py::test_shared_discrete_hazard_fit_predicts_monotone_survival tests/test_methods.py::test_deepsurv_moco_fit_predict_works_without_momentum_encoder tests/test_methods.py::test_deepsurv_moco_requires_observed_events` - passed
- `python -m compileall survarena/methods/deep survarena/benchmark/tuning.py` - passed
- Standalone `logistic_hazard` PyCox smoke fit/predict - passed with a torchtuples deprecation warning
- `python -m ruff check survarena tests scripts` - passed
- `pytest` - passed

## Deferred

- Predictor budget private wrappers: tests monkeypatch these hooks, so removing them is not behavior-preserving.
- `automl.presets._foundation_runtime_status`: tests monkeypatch this hook, so it remains.
- Foundation/AutoGluon discrete-hazard prediction wrappers: public adapter behavior and artifact metadata are involved; not a cleanup-only target.
- Broad test fake deduplication: repeated helpers are scenario-local and extracting them would add churn without a clear safety gain.
- Public compatibility method IDs and legacy artifact aliases: deliberately untouched.
- `survarena/methods/foundation/pooled_hazard.py`: internally unreferenced but retained as an external compatibility import path.
