# No-HPO Manuscript Benchmark

## Task

Run `configs/benchmark/manuscript_v1.yaml` no-HPO benchmark sequentially by
method, then rebuild comprehensive manuscript Elo outputs.

## Execution Notes

- Use built-in manuscript datasets from `manuscript_v1.yaml`.
- Treat duplicate cohort variants as out of scope for manuscript-suite
  additions.
- Store matrix outputs under `results/manuscript_dataset_model/<dataset>/<method>/`.
- Rebuild Elo with `scripts/build_manuscript_elo.py` after all method/dataset
  jobs complete.
