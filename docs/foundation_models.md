# Foundation Model Roadmap

Living roadmap for optional tabular foundation adapters (not a blocking issue list).

## Current State

- implemented adapters: `tabpfn_survival`, `mitra_survival`
- catalog-only candidates: `tabicl_survival`, `tabdpt_survival`, `realtabpfn_survival`
- runtime inspection: `survarena foundation-check`
- predictor access: `presets="foundation"`, `presets="all"`, or `enable_foundation_models=True`
- current skip rules: low-event data, unsupported feature types, or dataset shape beyond backbone hints

TabPFN note:

- accept the gated checkpoint terms at [Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- authenticate with `hf auth login` or `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`

## Next Steps

- better preprocessing for datetime, text, and high-cardinality categorical data
- richer survival heads and fine-tuning controls
- clearer artifact logging and leaderboard metadata
- broader backbone coverage once the adapter contract is stable

Foundation models stay optional members of the same `SurvivalPredictor`
workflow rather than a separate product surface.
