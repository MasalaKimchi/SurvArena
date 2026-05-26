# Top SurvBoard Dataset Integration

## Task

Select the five most important SurvBoard-aligned public omics survival datasets
for SurvArena and make them accessible through local preparation, dataset
configs, and the genomics manuscript benchmark config.

## Selection Criteria

- SurvBoard benchmark coverage and common use in cancer survival literature.
- Public, low-friction UCSC Xena access to OS labels and RNA-seq matrices.
- Enough rows and observed events for repeated benchmark splits.
- Cancer-type diversity rather than multiple near-duplicate cohorts.
- No duplicate METABRIC variant because the standard manuscript suite already
  contains METABRIC.

## Selected Cohorts

- `tcga_brca_xena`
- `tcga_luad_xena`
- `tcga_kirc_xena`
- `tcga_skcm_xena`
- `tcga_ov_xena`

