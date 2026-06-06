---
status: complete
completed: 2026-06-06
---

# Pooled Hazard Foundation Survival

Implemented canonical pooled discrete-time hazard foundation adapters under
`*_discrete_hazard_survival` method IDs, replacing draft pooled-hazard method
IDs in registry, catalog, configs, tests, and manuscript-facing docs.

## Verification

- `python -m pytest tests/test_foundation.py tests/test_methods.py tests/test_benchmark.py -q`
- `python -m pytest tests/test_evaluation.py tests/test_api.py tests/test_data.py -q`
- `ruff check survarena tests scripts`
- `python -m survarena.run_benchmark --config configs/benchmark/foundation_discrete_hazard_smoke.yaml --dry-run`
