# SurvArena Project State

SurvArena is being narrowed to a manuscript-grade benchmark toolkit for
right-censored tabular survival analysis. The retained benchmark contract is a
single config-driven evidence path:

- config: `configs/benchmark/manuscript_v1.yaml`
- datasets: `support`, `metabric`, `nwtco`, `aids`, `gbsg2`, `flchain`, `whas500`
- methods: the native manuscript portfolio plus discrete-hazard foundation
  adapters `tabpfn_survival`, `tabicl_survival`, `tabm_survival`, and
  `realtabpfn_survival`
- mode: `no_hpo`
- geometry: 5 folds x 3 repeats per dataset/method pair
- artifact policy: compact CSV outputs and one retained manuscript Elo/report
  evidence bundle under `results/manuscript_grade/clinical_no_hpo/elo/`

Retired benchmark surfaces include smoke, standard, local-HPO, cloud-HPO,
foundation-only, KKBox, XGBSE, and separate foundation Elo configs.
Those paths should not be cited as maintained evidence or used as defaults.

## Current Evidence

The retained clinical manuscript no-HPO evidence bundle is complete for the
pre-discrete-hazard 27-method matrix: 2,835 successful fold rows across 7
datasets, 27 methods, and 15 splits per dataset/method pair. New manuscript
foundation runs should refresh this evidence with the canonical discrete-hazard
foundation adapters.

The stricter publication-readiness gate lives in
`docs/manuscript_publishability.md` and can be regenerated with
`python scripts/audit_manuscript_publishability.py`. The current audit verdict
is not publication-ready as a final manuscript bundle because clinical HPO is
missing, genomics remains partial, and canonical discrete-hazard foundation
evidence is incomplete or requires a locked provenance bridge from alias-run
artifacts.

The retained Elo/reporting bundle now contains metric-specific Elo ladders,
paired win-rate tables, rank summaries, coverage summaries, method summaries,
figures, and `metric_suite_index.csv`. Calibration reporting uses absolute-error
metrics; raw slope/intercept diagnostics and old decision-curve aliases are no
longer retained in exported metric bundles.

## Remaining Work

- keep the metric-suite Elo bundle synchronized with any new manuscript result
  artifacts
- run `python scripts/audit_manuscript_publishability.py --strict` before
  treating local artifacts as publication-ready
- rerun calibration-sensitive foundation checks after adapter changes
- keep code-review-graph updated after cleanup passes
- avoid reintroducing parallel benchmark YAMLs unless the protocol itself
  changes enough to justify a new maintained track
