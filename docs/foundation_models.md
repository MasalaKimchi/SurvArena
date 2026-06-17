# Foundation Model Roadmap

Living roadmap for optional tabular foundation adapters (not a blocking issue list).

Last reviewed against manuscript config and extras: 2026-06-14.

## Current State

- default foundation adapters: `tabpfn_survival`, `tabicl_survival`,
  `tabm_survival`, and `realtabpfn_survival`
- compatibility aliases: `tabpfn_discrete_hazard_survival`,
  `tabicl_discrete_hazard_survival`, `tabm_discrete_hazard_survival`, and
  `realtabpfn_discrete_hazard_survival`
- additional foundation adapter: `mitra_survival_frozen`
- runtime inspection: `survarena foundation-check`
- CLI access: `--foundation`
- predictor access: `presets="foundation"`, `presets="all"`, or
  `enable_foundation_models=True`
- benchmark access: `configs/benchmark/manuscript_v1.yaml` is the
  manuscript-scope no-HPO benchmark for native plus frozen/bounded
  discrete-hazard foundation adapters
- current skip rules: low-event data, unsupported feature types, or dataset shape beyond backbone hints
- dependency extras: `foundation-tabpfn`, `foundation-tabarena`, `foundation-mitra`, or `foundation`

TabPFN note:

- accept the gated checkpoint terms at [Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- authenticate with `hf auth login` or `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`

## Survival Adaptation Pattern

SurvArena treats tabular foundation models as frozen tabular learners by
default, not as full survival models that must be fine-tuned end to end. The
maintained foundation adapters use a pooled discrete-time hazard contract:

1. fit or load the tabular backbone under `backbone_training: frozen`
2. build patient-interval rows from the benchmark training split only
3. train one binary classifier for conditional interval event hazards
4. reconstruct survival curves by cumulative products of predicted conditional
   survival probabilities
5. emit the same `predict_risk` and `predict_survival` outputs as native methods

This keeps focused validation runs practical and preserves the shared-split benchmark
contract.

Adapter-specific details:

- `tabpfn_survival`, `tabicl_survival`, `tabm_survival`, and
  `realtabpfn_survival` are censored-aware pooled discrete-time hazard
  adapters. They train one classifier on patient-interval rows among subjects
  known to be at risk at interval start, then reconstruct survival by
  cumulative products of predicted conditional survival probabilities.
- `tabpfn_discrete_hazard_survival`, `tabicl_discrete_hazard_survival`,
  `tabm_discrete_hazard_survival`, and `realtabpfn_discrete_hazard_survival`
  remain registered compatibility aliases for older configs and evidence
  collection scripts. New maintained configs should use the canonical IDs
  without the `_discrete_hazard_survival` suffix.
- `mitra_survival_frozen` remains available but is excluded from the manuscript
  no-HPO track because local RAM/CPU use can exceed the conventional model
  wall-clock budget. It still uses the AutoGluon event-risk plus Breslow
  survival adapter because Mitra is exposed as an event-risk tabular backbone.

See [`discrete_time_hazard_adapter.md`](discrete_time_hazard_adapter.md) for
the mathematical contract. The benchmark requirement is that each head is
trained only on training-side data and returns risk scores plus survival
probabilities at requested evaluation times.

## Next Steps

- better preprocessing for datetime, text, and high-cardinality categorical data
- richer survival heads and fine-tuning controls
- broader backbone coverage once the adapter contract is stable

Foundation models stay optional members of the same `SurvivalPredictor`
workflow rather than a separate product surface.
