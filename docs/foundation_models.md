# Foundation Model Roadmap

Living roadmap for optional tabular foundation adapters (not a blocking issue list).

## Current State

- implemented adapters: `tabpfn_survival`, `mitra_survival`
- catalog-only candidates: `tabicl_survival`, `tabdpt_survival`, `realtabpfn_survival`
- runtime inspection: `survarena foundation-check`
- CLI access: `--foundation`
- predictor access: `presets="foundation"`, `presets="all"`, or
  `enable_foundation_models=True`
- benchmark access: `configs/benchmark/foundation_tabpfn_frozen_smoke.yaml`
  provides a bounded no-HPO frozen TabPFN smoke path, and
  `configs/benchmark/mitra_survival_no_hpo_smoke.yaml` provides a bounded
  no-HPO Mitra smoke path; keep foundation adapters out of manuscript and local
  feasible HPO claims until the evidence bundle is promoted
- current skip rules: low-event data, unsupported feature types, or dataset shape beyond backbone hints

TabPFN note:

- accept the gated checkpoint terms at [Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- authenticate with `hf auth login` or `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`

## Survival Adaptation Pattern

SurvArena treats tabular foundation models as pretrained feature extractors by
default, not as full survival models that must be fine-tuned end to end. The
current adapters use a two-stage contract:

1. fit or load the tabular backbone under `backbone_training: frozen`
2. transform each row into a compact representation or surrogate prediction
3. train a lightweight survival head on the benchmark training split only
4. calibrate survival curves from the fitted Cox-style baseline hazard
5. emit the same `predict_risk` and `predict_survival` outputs as native methods

This keeps smoke runs practical and preserves the shared-split benchmark
contract.

Adapter-specific details:

- `tabpfn_survival` uses frozen TabPFN preprocessing/embeddings and trains an MLP
  Cox head; `n_estimators_final_inference` controls inference ensemble breadth.
- `mitra_survival` uses AutoGluon Tabular's `MITRA` model as a binary event-risk
  learner with `fine_tune=false` by default, then calibrates survival curves
  with the shared Breslow baseline survival adapter. AutoGluon's Mitra extra
  requires `torch>=2.6`, which the default SurvArena dependency set now pins
  through `torch==2.6.0`.

Other survival heads can be added behind the same interface: discrete-time
hazard heads, AFT heads, DeepHit-style competing-risk heads, or calibrated
stacking heads. The benchmark requirement is that each head is trained only on
training-side data and returns risk scores plus survival probabilities at
requested evaluation times.

## Next Steps

- better preprocessing for datetime, text, and high-cardinality categorical data
- richer survival heads and fine-tuning controls
- richer foundation-specific artifact fields
- broader backbone coverage once the adapter contract is stable

Foundation models stay optional members of the same `SurvivalPredictor`
workflow rather than a separate product surface.
