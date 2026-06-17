---
status: complete
completed: 2026-06-14
---

# Discrete-Hazard Foundation Default Summary

## Outcome

Canonical foundation method IDs now use pooled discrete-time hazard survival by default:

- `tabpfn_survival`
- `tabicl_survival`
- `tabm_survival`
- `realtabpfn_survival`

The explicit `_discrete_hazard_survival` method IDs remain registered as compatibility aliases, but maintained benchmark
configs and the predictor foundation catalog advertise the canonical IDs only.

## Implementation Notes

- Removed the old independent-horizon foundation implementations from TabPFN, TabICL, TABM, and RealTabPFN paths.
- Removed the legacy direct-horizon compatibility module; explicit alias method IDs remain registered for config/runtime
  compatibility.
- Removed the stale validation-diagnostic hook that searched for horizon-probability methods no adapter implements now.
- Shared TabPFN backbone/Kaplan-Meier helpers through `survarena/methods/foundation/tabpfn_backbone.py`.
- Routed canonical registry entries and configs to pooled discrete-time hazard parameters.
- Updated manuscript benchmark configs, method YAMLs, and docs to describe the default foundation contract.
- Updated tests to assert canonical defaults, compatibility alias registration, and catalog non-advertisement of aliases.

## Verification

- `python -m ruff check survarena tests scripts configs`
- `python -m pytest -q` (`207 passed, 6 skipped`)
- `python -m survarena.run_benchmark --config configs/benchmark/foundation_discrete_hazard_smoke.yaml --dry-run`
- dry-runs for all 9 changed benchmark configs
- `./scripts/validate_benchmark_protocol.sh`
- `python -m compileall survarena`
- `code-review-graph` incremental update and minimal change review

## Notes

Historical benchmark results were not rewritten. Existing evidence bundles that predate this default switch should be
refreshed before being treated as final manuscript evidence for the foundation track.
