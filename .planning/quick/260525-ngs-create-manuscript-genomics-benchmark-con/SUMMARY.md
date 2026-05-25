---
status: completed
---

# Manuscript Genomics Benchmark Config

GSD command:

```bash
node /Users/justin/.codex/get-shit-done/bin/gsd-tools.cjs init quick 'Create manuscript genomics benchmark config and remove duplicate METABRIC dataset'
```

## Outcome

- Added `configs/benchmark/manuscript_genomics_v1.yaml`.
- Verified it differs from `configs/benchmark/manuscript_v1.yaml` only by
  `benchmark_id`, `datasets`, and `notes`.
- Kept the same 25-model list and the same no-HPO budget fields, including
  `hpo.enabled: false`, `timeout_seconds: 120`, and the foundation-model
  default parameter overrides.
- Removed `metabric_cbioportal` from runnable dataset configs, cancer dataset
  preparation choices, docs, and local generated raw/processed artifacts.

