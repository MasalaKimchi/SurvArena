# Retry Genomics Foundation Models

## Task

Retry failed/incomplete foundation-model runs from the genomics manuscript
matrix and update ELO eligibility if the retries produce complete fold coverage.

## Attempts

- Retry `tabpfn_survival` on `tcga_brca_xena` with CPU device to avoid Apple MPS
  prediction OOM.
- Run verbose `mitra_survival_frozen` diagnostics on `tcga_brca_xena` with a
  larger AutoGluon memory guard to determine whether the failure is recoverable.

