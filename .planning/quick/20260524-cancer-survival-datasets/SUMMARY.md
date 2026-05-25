---
status: complete
---

# Cancer Survival Dataset Expansion

Completed local-file dataset integration, preparation script, configs, and docs
for `tcga_luad_xena` and `tcga_kirc_xena`. `metabric_cbioportal` was removed
from the runnable candidate set because it duplicates the existing manuscript
`metabric` cohort.

Verification:

- `python scripts/prepare_cancer_survival_datasets.py --dataset all --max-genes 500`
- `python -m pytest tests/test_dataset_loaders.py`
- `ruff check survarena/data/loaders.py scripts/prepare_cancer_survival_datasets.py tests/test_dataset_loaders.py`
