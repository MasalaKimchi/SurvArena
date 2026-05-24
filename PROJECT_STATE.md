# SurvArena Project State

SurvArena is being narrowed to a manuscript-grade benchmark toolkit for
right-censored tabular survival analysis. The retained benchmark contract is a
single config-driven evidence path:

- config: `configs/benchmark/manuscript_v1.yaml`
- datasets: `support`, `metabric`, `aids`, `gbsg2`, `flchain`, `whas500`
- methods: the native manuscript portfolio plus `tabpfn_survival` and
  `mitra_survival_frozen`
- mode: `no_hpo`
- geometry: 5 folds x 3 repeats per dataset/method pair
- artifact policy: compact `core_csv` outputs and one retained manuscript Elo
  evidence bundle under `results/manuscript_elo/`

Retired benchmark surfaces include smoke, standard, local-HPO, cloud-HPO,
foundation-only, KKBox, NWTCO, XGBSE, and separate foundation Elo configs.
Those paths should not be cited as maintained evidence or used as defaults.

## Current Evidence

The retained native manuscript evidence bundle is complete for the previous
23-method native-only matrix: 2,070 successful fold rows across 6 datasets,
23 methods, and 15 splits per dataset/method pair.

After adding `tabpfn_survival` and `mitra_survival_frozen` to the manuscript
config, the unified manuscript matrix expects 2,250 fold rows. The two
foundation methods still need manuscript-grade runs before native+foundation
claims are complete.

## Remaining Work

- run the two foundation methods across the six manuscript datasets
- rebuild `results/manuscript_elo/` with the unified 25-method matrix
- regenerate README/doc figures from the unified artifact bundle
- keep code-review-graph updated after cleanup passes
- avoid reintroducing parallel benchmark YAMLs unless the protocol itself
  changes enough to justify a new maintained track
