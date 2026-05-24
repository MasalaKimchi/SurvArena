# Protocol

SurvArena compares full pipelines, not single lucky runs. The contract is:
shared splits, shared budgets, training-side preprocessing only, and disk-first
artifacts.

Last reviewed against the benchmark config: 2026-05-24.

## Manuscript Benchmark

`configs/benchmark/manuscript_v1.yaml` defines the retained manuscript track:

- task: right-censored tabular survival prediction
- split strategy: repeated nested CV
- outer loop: 5 folds x 3 repeats
- inner loop: 3 folds
- seeds: `[11, 22, 33, 44, 55]`
- primary metric: `uno_c`
- secondary metrics: `harrell_c`, `ibs`, `td_auc`, `brier`, `calibration`, and `net_benefit`
- methods: the native manuscript portfolio plus `tabpfn_survival` and `mitra_survival_frozen`

Benchmark profiles (see `validate_benchmark_profile_contract` in
`survarena/benchmark/runner.py`):

- `manuscript`: full native method portfolio, extended secondaries, and stricter reporting expectations

## Manuscript Scope

The main-paper benchmark scope is `configs/benchmark/manuscript_v1.yaml`: native
Python survival adapters, frozen/bounded foundation adapters, shared repeated
nested CV, and no-HPO/default-policy comparison. Separate smoke, standard,
cloud, HPO, KKBox, and XGBSE expansion configs have been retired from the
maintained benchmark surface.

Optional **robustness** blocks in benchmark YAML (`robustness.enabled`, `tracks`,
`severity_levels`) control optional perturbation tracks; when disabled, only the
baseline track runs.

## User Dataset Comparison

`compare_survival_models(...)` and `survarena compare` use the same reporting
style on user data.

Supported split strategies:

- `fixed_split` for a quick one-seed comparison
- `repeated_nested_cv` for stricter benchmark-style evaluation

## Evaluation Rules

- every method sees the same split definitions
- preprocessing is fit on training-side data only
- no-HPO benchmark mode fits configured defaults directly on each outer-training split
- `comparison_modes` is retained as `[no_hpo]` for manuscript-grade runs
- the selected config is refit before outer-test evaluation
- seeds are passed through to stochastic methods

See [`training_strategy.md`](training_strategy.md) for the fold geometry and
runtime planning ranges for manuscript-shaped runs.

## Metrics

- discrimination: Harrell C-index, Uno C-index
- overall survival quality: integrated Brier score, Brier score at 25/50/75% event-time horizons, time-dependent AUC
- calibration and utility: median-horizon calibration slope/intercept and median-horizon net benefit
- efficiency: fit time, inference time, peak memory

Metric computation is backed by `torchsurv`.

## Reproducibility

- split definitions are persisted under `data/splits/<task_id>/`
- split manifests guard against stale splits after dataset or config changes
- benchmark and method config hashes are recorded in run payloads
- experiment manifests capture run-level metadata

## Automated protocol check

`scripts/validate_benchmark_protocol.sh` runs a dry run plus a one-dataset
one-method execution against a benchmark config (default:
`configs/benchmark/manuscript_v1.yaml`) and asserts that expected
summary artifacts exist. Set `BENCHMARK_CONFIG`, `WORK_DIR`, or `PYTHON_BIN` to
override defaults.

The same checks are available through the CLI surface:

```bash
survarena benchmark plan --config configs/benchmark/manuscript_v1.yaml
survarena benchmark doctor --config configs/benchmark/manuscript_v1.yaml --load-datasets
survarena benchmark run --config configs/benchmark/manuscript_v1.yaml --dry-run
```

## Output Contract

Benchmark-style runs write to a generated results directory or to the directory
provided with `--output-dir`, which is useful for protocol validation and
resumable focused runs.
Outputs are always core CSV artifacts and each run writes:

- `<model_name>_fold_results.csv`: atomic split/method rows with metrics,
  runtime, HPO governance, robustness, retry, and status fields
- `<model_name>_leaderboard.csv`: per-dataset method aggregates ranked by the
  primary metric
- `<model_name>_run_diagnostics.csv`: dataset curation, run/failure summaries,
  and HPO trial rows in one auditable table

Fold-result schemas include identifiers, split and mode fields, core and
time-horizon metrics, timing and memory telemetry, HPO governance fields,
robustness/parity fields, retry metadata, and status.
