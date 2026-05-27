---
status: completed
---

# Manuscript Genomics Benchmark And ELO

GSD command:

```bash
node /Users/justin/.codex/get-shit-done/bin/gsd-tools.cjs init quick 'Run manuscript genomics benchmark and build per-metric ELO outputs'
```

## Outcome

- Ran all 125 configured dataset/method process jobs for
  `configs/benchmark/manuscript_genomics_v1.yaml`.
- Wrote raw matrix artifacts under `results/manuscript_genomics_dataset_model/`.
- Built a strict complete-coverage ELO input with 19 eligible methods across all
  five genomics datasets under
  `results/manuscript_genomics_dataset_model_complete_eligible/`.
- Built genomics ELO outputs under `results/genomics_elo/` for 15 metrics:
  `uno_c`, `harrell_c`, `ibs`, `td_auc_25`, `td_auc_50`, `td_auc_75`,
  `brier_25`, `brier_50`, `brier_75`, `net_benefit_25`, `net_benefit_50`,
  `net_benefit_75`, `decision_curve_aunb_25`, `decision_curve_aunb_50`, and
  `decision_curve_aunb_75`.
- Recorded `calibration_slope_50` as not ELO-eligible because only 91/95
  eligible dataset-method pairs had non-null values.
- Recorded ineligible dataset-method pairs in
  `results/genomics_elo/ineligible_dataset_method_pairs.csv`.
- Added TabPFN prediction batching support and a fake-backbone unit test after
  diagnosing BRCA MPS prediction OOM; the rerun remained outside the practical
  no-HPO runtime envelope, so TabPFN remained ineligible for BRCA.
- Used code-review-graph change analysis; it reported low risk and no affected
  flows.

## Top Uno C ELO

| Rank | Method | Elo |
| ---: | --- | ---: |
| 1 | `xgboost_cox` | 1675.73 |
| 2 | `catboost_cox` | 1629.76 |
| 3 | `xgboost_aft` | 1620.99 |
| 4 | `extra_survival_trees` | 1612.37 |
| 5 | `catboost_survival_aft` | 1589.43 |

## Verification

```bash
python -m pytest tests/test_foundation.py tests/test_dataset_loaders.py tests/test_benchmark_runner.py
ruff check survarena/methods/foundation/tabpfn_survival.py scripts/prepare_cancer_survival_datasets.py tests/test_foundation.py tests/test_benchmark_runner.py tests/test_dataset_loaders.py
```

