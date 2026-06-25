---
status: complete
created: 2026-06-15
completed: 2026-06-19
---

# Run Current-Default Manuscript Matrices

## Objective

Run the current canonical discrete-hazard-default manuscript no-HPO benchmark matrices across all configured clinical and
genomics datasets and methods, without overwriting historical retained artifacts mid-run.

## Runs

- Clinical: `configs/benchmark/manuscript_v1.yaml`
  - output root: `results/manuscript_grade/clinical_no_hpo_current_default/dataset_model`
  - 7 datasets x 27 methods x 15 splits
- Genomics: `configs/benchmark/manuscript_genomics_v1.yaml`
  - output root: `results/manuscript_grade/genomics_no_hpo_current_default/dataset_model`
  - 5 datasets x 27 methods x 15 splits

## Notes

Use `scripts/run_manuscript_by_dataset_model.sh` with `--resume` so interrupted cells can be continued. Promote to the
retained manuscript artifact paths only after coverage and publishability audits pass.
