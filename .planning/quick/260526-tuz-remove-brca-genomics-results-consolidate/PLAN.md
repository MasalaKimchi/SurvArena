# Remove BRCA And Consolidate Results

## Task

Remove BRCA-associated genomics result artifacts so the strict genomics ELO can
include complete-coverage foundation results where available, consolidate
manuscript-grade outputs under one result tree, and stop saving ELO figures as
PDF files.

## Scope

- Remove `tcga_brca_xena` result directories and retry artifacts.
- Rebuild the genomics complete-coverage input over the remaining four TCGA
  genomics datasets.
- Rebuild per-metric genomics ELO in the consolidated result tree.
- Keep manuscript-grade outputs under `results/manuscript_grade/`.
- Change `scripts/build_manuscript_elo.py` to write PNG figures only.

