# Foundation Model Roadmap

Living roadmap for optional tabular foundation adapters (not a blocking issue list).

Last reviewed against manuscript config and extras: 2026-05-24.

## Current State

- implemented adapters: `tabpfn_survival`, `mitra_survival_frozen`
- catalog-only candidates: `tabicl_survival`, `tabdpt_survival`, `realtabpfn_survival`
- runtime inspection: `survarena foundation-check`
- CLI access: `--foundation`
- predictor access: `presets="foundation"`, `presets="all"`, or
  `enable_foundation_models=True`
- benchmark access: `configs/benchmark/manuscript_v1.yaml` is the
  manuscript-scope no-HPO benchmark for native plus frozen/bounded foundation
  adapters
- current skip rules: low-event data, unsupported feature types, or dataset shape beyond backbone hints
- dependency extras: `foundation-tabpfn`, `foundation-mitra`, or `foundation`

TabPFN note:

- accept the gated checkpoint terms at [Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- authenticate with `hf auth login` or `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`

## Survival Adaptation Pattern

SurvArena treats tabular foundation models as frozen tabular learners by
default, not as full survival models that must be fine-tuned end to end. The
current adapters use a horizon/event-risk contract:

1. fit or load the tabular backbone under `backbone_training: frozen`
2. train TabPFN horizon classifiers or Mitra event-risk learners on the benchmark training split only
3. exclude rows whose event status is unknown at a TabPFN horizon
4. reconstruct or calibrate survival curves from training-side estimates
5. emit the same `predict_risk` and `predict_survival` outputs as native methods

This keeps focused validation runs practical and preserves the shared-split benchmark
contract.

Adapter-specific details:

- `tabpfn_survival` is a censored-aware TabPFN horizon adapter.
  It trains one frozen TabPFN classifier per event-time horizon using only rows
  with known event status at that horizon, falls back to Kaplan-Meier event
  probabilities when a horizon is under-supported, and reconstructs monotone
  survival curves from cumulative event probabilities.
- `mitra_survival_frozen` uses AutoGluon Tabular's `MITRA` model as a binary event-risk
  learner with `fine_tune=false` by default, then calibrates survival curves
  with the shared Breslow baseline survival adapter. The default SurvArena
  dependency set pins `torch==2.6.0`, matching the current Mitra compatibility
  path.
  Full-backbone fine-tuning is intentionally excluded from the manuscript track
  because CPU-only runs can exceed the
  conventional model wall-clock budget.

Other survival heads can be added behind the same interface: discrete-time
hazard heads, AFT heads, DeepHit-style competing-risk heads, or calibrated
stacking heads. The benchmark requirement is that each head is trained only on
training-side data and returns risk scores plus survival probabilities at
requested evaluation times.

## Next Steps

- better preprocessing for datetime, text, and high-cardinality categorical data
- richer survival heads and fine-tuning controls
- broader backbone coverage once the adapter contract is stable

Foundation models stay optional members of the same `SurvivalPredictor`
workflow rather than a separate product surface.
