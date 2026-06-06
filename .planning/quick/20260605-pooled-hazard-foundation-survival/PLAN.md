---
status: in_progress
created: 2026-06-05
---

# Pooled Hazard Foundation Survival

Implement and evaluate alternatives to independent per-horizon foundation survival classifiers.

## Scope

- Add shared utilities for train-event time bins, person-time hazard rows, hazard-to-survival reconstruction, and risk aggregation.
- Add a native shared discrete-time hazard neural model.
- Add pooled person-time hazard foundation adapters for TabPFN, TabICL, TabM, and RealTabPFN.
- Register methods and add method configs for manuscript benchmark usage.
- Add focused unit tests with fake backbones/AutoGluon predictors.
- Run targeted tests and a small genomics smoke comparison; full manuscript-grade genomics runs may exceed local CPU time and should be launched from the new configs if not feasible in-turn.

## Verification

- `python -m pytest tests/test_foundation.py tests/test_methods.py`
- `ruff check survarena tests`
- small local genomics benchmark smoke run comparing existing horizon adapters against pooled hazard adapters where dependencies/runtime permit.
