# Manuscript Genomics Benchmark Config

## Task

Remove the duplicate METABRIC cBioPortal candidate and add a genomics-only
manuscript benchmark config that mirrors `manuscript_v1.yaml` while targeting
the non-duplicate TCGA genomics datasets.

## Scope

- Add `configs/benchmark/manuscript_genomics_v1.yaml`.
- Keep model list, split protocol, metrics, AutoGluon settings, and HPO/no-HPO
  budget identical to `configs/benchmark/manuscript_v1.yaml`.
- Target only `tcga_luad_xena` and `tcga_kirc_xena`.
- Remove `metabric_cbioportal` from active dataset configs, prep script choices,
  docs, and local generated artifacts.

