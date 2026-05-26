---
status: completed
---

# Top SurvBoard Dataset Integration

GSD command:

```bash
node /Users/justin/.codex/get-shit-done/bin/gsd-tools.cjs init quick 'Select and integrate top SurvBoard datasets for SurvArena'
```

## Outcome

- Added dataset configs for `tcga_brca_xena`, `tcga_skcm_xena`, and
  `tcga_ov_xena`.
- Retained and annotated existing `tcga_luad_xena` and `tcga_kirc_xena` as
  SurvBoard-aligned candidates.
- Expanded `scripts/prepare_cancer_survival_datasets.py` so `--dataset all`
  prepares all five top cohorts.
- Expanded `configs/benchmark/manuscript_genomics_v1.yaml` to target the five
  selected cohorts while preserving the `manuscript_v1` model list and no-HPO
  protocol settings.
- Prepared all five local Parquet artifacts with `--max-genes 500`.
- Updated dataset docs with selection criteria, download commands, and
  SurvBoard source links.
- Used code-review-graph change analysis; it reported low risk and no affected
  flows.

## Prepared Local Shapes

| Dataset | Rows | Features | Events | Event rate |
| --- | ---: | ---: | ---: | ---: |
| `tcga_brca_xena` | 1203 | 500 | 197 | 0.1638 |
| `tcga_luad_xena` | 576 | 500 | 211 | 0.3663 |
| `tcga_kirc_xena` | 605 | 500 | 199 | 0.3289 |
| `tcga_skcm_xena` | 459 | 500 | 222 | 0.4837 |
| `tcga_ov_xena` | 428 | 500 | 265 | 0.6192 |

## Verification

```bash
python scripts/prepare_cancer_survival_datasets.py --dataset all --max-genes 500
python -m survarena.run_benchmark --config configs/benchmark/manuscript_genomics_v1.yaml --dry-run
python -m pytest tests/test_dataset_loaders.py tests/test_benchmark_runner.py
ruff check scripts/prepare_cancer_survival_datasets.py tests/test_benchmark_runner.py tests/test_dataset_loaders.py
```

