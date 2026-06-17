# Quick Task 20260617: Codebase Refactor Cleanup

**Status:** In progress
**Task:** Refactor redundant benchmark, predictor, discrete hazard, deep model, and config code while preserving public behavior.

## Tasks

1. Consolidate duplicated deep and discrete-hazard helper logic.
   - Files: `survarena/methods/deep/*`, `survarena/methods/foundation/discrete_hazard.py`, `survarena/methods/automl/mitra_survival.py`
   - Verify: `pytest tests/test_methods.py tests/test_foundation.py`

2. Reduce benchmark and predictor record/prediction duplication.
   - Files: `survarena/benchmark/runner.py`, `survarena/api/predictor.py`
   - Verify: `pytest tests/test_benchmark.py tests/test_api.py`

3. Deduplicate method config aliases without changing config IDs or registered method behavior.
   - Files: `configs/methods/*.yaml`, `survarena/config.py`
   - Verify: targeted config tests and dry-run tests.

## Constraints

- Preserve public APIs, method IDs, CLI flags, artifact shapes, and serialized fields.
- Keep compatibility aliases working.
- Avoid dependency upgrades or benchmark protocol changes.
