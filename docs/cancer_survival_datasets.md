# Cancer Survival Dataset Candidates

Last reviewed: 2026-05-26.

This note records the first biomedical survival datasets to add as optional
large-track candidates. They are intentionally local-only because the raw omics
matrices are larger than the built-in toy/standard datasets and should not be
committed to the repository.

## Why These Datasets

Recent multi-omics survival benchmarking work uses TCGA, ICGC, TARGET, and
METABRIC as the core cancer-program sources. SurvBoard, for example, reports 28
cancer datasets across TCGA, ICGC, TARGET, and METABRIC, with TCGA as the
largest and most commonly used multi-omics cancer survival resource.

UCSC Xena is the most practical first download path for TCGA because it exposes
pre-compiled clinical, survival, and RNA-seq matrices as tab-delimited files
that can be read directly from Python. The five runnable TCGA cohorts were
chosen from SurvBoard's TCGA roster using these importance criteria:

- common use in cancer survival benchmark papers and SurvBoard coverage
- sufficient survival rows and events for repeated benchmark splits
- direct public RNA-seq and OS-label availability through UCSC Xena
- cancer-type diversity across breast, lung, kidney, melanoma, and ovarian
  disease contexts

METABRIC cBioPortal was evaluated as a candidate but removed from the runnable
suite because the manuscript benchmark already includes METABRIC through the
standard `metabric` dataset.

## Top Candidates

| Dataset id | Source | Why it belongs | Default prepared features | Local artifact |
| --- | --- | --- | --- | --- |
| `tcga_brca_xena` | UCSC Xena GDC hub | Largest and most recognizable TCGA cancer cohort; central in SurvBoard and broad survival-model literature. | Top-variance STAR-count genes from `TCGA-BRCA.star_counts.tsv.gz`. | `data/processed/cancer_survival/tcga_brca_xena.parquet` |
| `tcga_luad_xena` | UCSC Xena GDC hub | Popular TCGA lung cancer survival cohort with direct RNA-seq and OS labels. | Top-variance STAR-count genes from `TCGA-LUAD.star_counts.tsv.gz`. | `data/processed/cancer_survival/tcga_luad_xena.parquet` |
| `tcga_kirc_xena` | UCSC Xena GDC hub | Popular TCGA kidney cancer survival cohort with strong event signal and direct RNA-seq labels. | Top-variance STAR-count genes from `TCGA-KIRC.star_counts.tsv.gz`. | `data/processed/cancer_survival/tcga_kirc_xena.parquet` |
| `tcga_skcm_xena` | UCSC Xena GDC hub | Melanoma cohort with comparatively high event rate and strong biological relevance for survival modeling. | Top-variance STAR-count genes from `TCGA-SKCM.star_counts.tsv.gz`. | `data/processed/cancer_survival/tcga_skcm_xena.parquet` |
| `tcga_ov_xena` | UCSC Xena GDC hub | Ovarian cancer cohort with many observed events, useful for discrimination and calibration stress-testing. | Top-variance STAR-count genes from `TCGA-OV.star_counts.tsv.gz`. | `data/processed/cancer_survival/tcga_ov_xena.parquet` |

## Download And Prepare

Prepare all five genomics datasets:

```bash
python scripts/prepare_cancer_survival_datasets.py --dataset all --max-genes 1000
```

Prepare only one dataset:

```bash
python scripts/prepare_cancer_survival_datasets.py --dataset tcga_brca_xena --max-genes 1000
python scripts/prepare_cancer_survival_datasets.py --dataset tcga_luad_xena --max-genes 1000
python scripts/prepare_cancer_survival_datasets.py --dataset tcga_kirc_xena --max-genes 1000
python scripts/prepare_cancer_survival_datasets.py --dataset tcga_skcm_xena --max-genes 1000
python scripts/prepare_cancer_survival_datasets.py --dataset tcga_ov_xena --max-genes 1000
```

The script writes raw downloads under `data/raw/cancer_survival/` and prepared
Parquet files under `data/processed/cancer_survival/`. Both directories are
ignored by git except for their placeholders.

## Integration

Each candidate has a config in `configs/datasets/`. Once the Parquet artifact
exists, it can be loaded like any other configured dataset:

```python
from pathlib import Path

from survarena.data.loaders import load_dataset

dataset = load_dataset("tcga_luad_xena", repo_root=Path("."))
print(dataset.X.shape, dataset.event.mean())
```

To smoke-test through the benchmark runner, start with a single fast method:

```bash
survarena benchmark run \
  --config configs/benchmark/manuscript_v1.yaml \
  --dataset tcga_luad_xena \
  --method coxph \
  --dry-run
```

Do not add these candidates to `configs/benchmark/manuscript_v1.yaml` until the
first complete run records wall-clock time and memory use. The TCGA expression
matrices are high-dimensional, so start with `--max-genes 500` or `1000` before
trying full matrices.

Use `configs/benchmark/manuscript_genomics_v1.yaml` for the isolated genomics
track. It mirrors the manuscript no-HPO protocol and model list while targeting
the five SurvBoard-aligned TCGA cohorts.

## Notes On Full-Scale Variants

- TCGA can be expanded beyond the top-five track by adding another
  `TCGA-<COHORT>` entry to `XENA_COHORTS` in
  `scripts/prepare_cancer_survival_datasets.py`.
- METABRIC full microarray expression is available through cBioPortal DataHub,
  but it duplicates the existing manuscript `metabric` cohort. Keep it out of
  the genomics track unless a future experiment explicitly compares cohort
  variants.
- CPTAC is attractive for proteomics, and cBioPortal exposes several CPTAC
  studies with mass-spectrometry protein profiles. However, older proteomics
  studies may not include clean OS fields in cBioPortal, while newer GDC CPTAC
  studies expose survival fields but not always protein matrices in the same
  cBioPortal study. Treat CPTAC as the next curation pass rather than one of the
  first three runnable candidates.

## Source Pages Checked

- UCSC Xena TCGA help: https://ucsc-xena.gitbook.io/project/public-data-we-host/tcga
- UCSC Xena download overview: https://xena.ucsc.edu/download-data/
- cBioPortal downloads/API/DataHub docs: https://docs.cbioportal.org/downloads/
- cBioPortal METABRIC study: https://www.cbioportal.org/study/summary?id=brca_metabric
- SurvBoard benchmark: https://survboard.science/
- SurvBoard source config: https://github.com/BoevaLab/survboard/blob/main/config/config.json
- SurvBoard Zenodo data record: https://zenodo.org/records/16616663
- CPTAC data portal: https://proteomics.cancer.gov/data-portal
