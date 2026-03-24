# Protocol

SurvArena compares full pipelines, not single lucky runs. The contract is:
shared splits, shared budgets, training-side preprocessing only, and disk-first
artifacts.

## Standard Benchmark

`configs/benchmark/standard_v1.yaml` defines the default research track:

- task: right-censored tabular survival prediction
- split strategy: repeated nested CV
- outer loop: 5 folds x 3 repeats
- inner loop: 3 folds
- seeds: `[11, 22, 33, 44, 55]`
- primary metric: `harrell_c`
- secondary metrics: `harrell_c`, `ibs`, `td_auc`
- default methods: `coxph`, `coxnet`, `rsf`, `deepsurv`

`configs/benchmark/large_v1.yaml` is the fixed-split large-track placeholder for `kkbox`.

## User Dataset Comparison

`compare_survival_models(...)` and `survarena compare` use the same reporting
style on user data.

Supported split strategies:

- `fixed_split` for a quick one-seed comparison
- `repeated_nested_cv` for stricter benchmark-style evaluation

## Evaluation Rules

- every method sees the same split definitions
- preprocessing is fit on training-side data only
- hyperparameter search uses the configured inner validation budget
- the selected config is refit before outer-test evaluation
- seeds are passed through to stochastic methods

## Metrics

- discrimination: Harrell C-index, Uno C-index
- overall survival quality: integrated Brier score, time-dependent AUC
- efficiency: fit time, inference time, peak memory

Metric computation is backed by `torchsurv`.

## Reproducibility

- split definitions are persisted under `data/splits/<task_id>/`
- split manifests guard against stale splits after dataset or config changes
- benchmark and method config hashes are recorded in run payloads
- experiment manifests capture run-level metadata

## Output Contract

Benchmark-style runs write to `results/summary/exp_<YYYYMMDD_HHMMSS>/`:

- `<benchmark_id>_fold_results.csv`
- `<benchmark_id>_seed_summary.csv`
- `<benchmark_id>_overall_summary.json`
- `<benchmark_id>_leaderboard.csv`
- `<benchmark_id>_leaderboard.json`
- `<benchmark_id>_run_records.jsonl.gz`
- `<benchmark_id>_run_records_index.json`
- `experiment_manifest.json`

Key per-run fields in fold results:

- `benchmark_id`
- `dataset_id`
- `method_id`
- `split_id`
- `seed`
- `validation_score`
- `harrell_c`
- `uno_c`
- `ibs`
- `fit_time_sec`
- `infer_time_sec`
- `peak_memory_mb`
- `status`
