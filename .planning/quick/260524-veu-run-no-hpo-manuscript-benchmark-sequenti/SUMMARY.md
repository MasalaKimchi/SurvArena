---
status: completed
---

# No-HPO Manuscript Benchmark

GSD command attempted:

```bash
node /Users/justin/.codex/get-shit-done/bin/gsd-tools.cjs init quick 'Run no-HPO manuscript benchmark sequentially by method and build comprehensive ELO'
```

The installed workflow reported `roadmap_exists: false`, so this task is tracked
as a quick artifact while benchmark execution proceeds inline.

## Outcome

- Ran the full `configs/benchmark/manuscript_v1.yaml` no-HPO matrix sequentially
  by dataset and method into `results/manuscript_dataset_model/`.
- Built final complete-coverage Elo outputs from 24 eligible methods and all 7
  manuscript datasets into `results/manuscript_elo/`.
- Updated the manuscript Elo figure asset at
  `docs/assets/elo_manuscript_no_hpo_uno_c.png`.
- Recorded `mitra_survival_frozen` as excluded from the final Elo in
  `results/manuscript_elo/ineligible_methods.csv`: it completed 5/7 dataset
  pairs, but `support` produced 0/15 successful folds and `flchain` produced
  1/15. Verbose AutoGluon diagnostics showed Mitra was skipped on `support`
  because the estimated memory requirement exceeded the configured
  `ag.max_memory_usage_ratio=1.3`; a higher memory-guard diagnostic began
  training but exceeded the intended 120-second per-fold no-HPO wall-clock
  behavior and was stopped.
- Used code-review-graph change analysis on the touched loader/script/test/docs
  set; it reported low risk, no affected execution flows, and one graph-level
  `load_dataset` test gap that is covered by the new dataset loader tests.
