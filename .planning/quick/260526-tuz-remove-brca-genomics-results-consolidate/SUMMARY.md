---
status: completed
---

# Remove BRCA And Consolidate Results

GSD command:

```bash
node /Users/justin/.codex/get-shit-done/bin/gsd-tools.cjs init quick 'Remove BRCA genomics results consolidate manuscript outputs and emit PNG ELO figures only'
```

## Outcome

- Removed all result paths containing `tcga_brca_xena`.
- Consolidated manuscript-grade outputs under `results/manuscript_grade/`.
- Rebuilt genomics complete-coverage input under
  `results/manuscript_grade/genomics_no_hpo/dataset_model_complete_eligible/`.
- Rebuilt genomics ELO under `results/manuscript_grade/genomics_no_hpo/elo/`.
- The four-cohort genomics ELO now includes `tabpfn_survival` with complete
  coverage. `mitra_survival_frozen` remains ineligible because it had 0/15
  successful folds on each remaining genomics cohort.
- Updated `scripts/build_manuscript_elo.py` so figures are saved as PNG only.
- Removed existing manuscript-grade PDF figures.

## Top Genomics Uno C ELO

| Rank | Method | Elo |
| ---: | --- | ---: |
| 1 | `catboost_cox` | 1732.38 |
| 2 | `xgboost_cox` | 1704.95 |
| 3 | `extra_survival_trees` | 1614.28 |
| 4 | `xgboost_aft` | 1614.28 |
| 5 | `catboost_survival_aft` | 1610.90 |
| 11 | `tabpfn_survival` | 1492.07 |

## Verification

```bash
python -m pytest tests/test_foundation.py tests/test_benchmark_runner.py
ruff check scripts/build_manuscript_elo.py survarena/methods/foundation/tabpfn_survival.py tests/test_foundation.py tests/test_benchmark_runner.py
python scripts/build_manuscript_elo.py --input-dir results/manuscript_grade/genomics_no_hpo/dataset_model_complete_eligible --output-dir /tmp/survarena_elo_png_check --metric uno_c --bootstrap 10 --strict-coverage --no-asset
```

