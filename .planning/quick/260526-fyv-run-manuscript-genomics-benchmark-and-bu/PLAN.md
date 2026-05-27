# Manuscript Genomics Benchmark And ELO

## Task

Run the `manuscript_genomics_v1` benchmark matrix across all configured TCGA
genomics datasets and all configured methods, skip completed jobs when present,
then build genomics ELO outputs separately for each metric with strict coverage.

## Execution

- Benchmark config: `configs/benchmark/manuscript_genomics_v1.yaml`
- Matrix output: `results/manuscript_genomics_dataset_model/`
- Complete-coverage ELO input: `results/manuscript_genomics_dataset_model_complete_eligible/`
- ELO output: `results/genomics_elo/`

