---
status: completed
---

# Retry Genomics Foundation Models

GSD command:

```bash
node /Users/justin/.codex/get-shit-done/bin/gsd-tools.cjs init quick 'Retry genomics foundation models and update ELO eligibility'
```

## Outcome

- `tabpfn_survival` on `tcga_brca_xena` was retried with `device: cpu` to avoid
  the original MPS prediction OOM. The run remained CPU-bound beyond the
  practical no-HPO repair window and was stopped, so TabPFN remains incomplete
  on BRCA for the strict genomics ELO.
- `mitra_survival_frozen` was retried on `tcga_brca_xena` with verbose logging
  and `ag.max_memory_usage_ratio: 2.0`. AutoGluon still skipped Mitra before
  training: estimated memory was about 28 GB versus about 6 GB available, and
  it suggested a memory ratio above 5.2. This is not appropriate for the local
  manuscript no-HPO CPU/memory contract.
- Existing strict genomics ELO outputs remain unchanged: foundation methods are
  represented in the raw matrix, but not in the complete-coverage ELO.

