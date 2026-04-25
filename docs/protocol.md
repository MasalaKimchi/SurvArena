# Protocol

SurvArena compares full pipelines, not single lucky runs. The contract is:
shared splits, shared budgets, training-side preprocessing only, and disk-first
artifacts.

## Standard Benchmark

`configs/benchmark/standard_v1.yaml` defines the default standard track:

- task: right-censored tabular survival prediction
- split strategy: repeated nested CV
- outer loop: 5 folds x 3 repeats
- inner loop: 3 folds
- seeds: `[11, 22, 33, 44, 55]`
- primary metric: `uno_c`
- secondary metrics: `harrell_c`, `ibs`, `td_auc`
- default methods: `coxph`, `coxnet`, `rsf`, `deepsurv`

`configs/benchmark/manuscript_v1.yaml` and the smoke suite configs add
`brier`, `calibration`, and `net_benefit` to `secondary_metrics`, plus the
usual time-horizon and decision-curve settings for those outputs.

Benchmark profiles (see `validate_benchmark_profile_contract` in
`survarena/benchmark/runner.py`):

- `smoke`: small folds/repeats/seeds for CI and quick checks, not for statistical claims
- `standard`: balanced rigor for routine method iteration (outer folds/repeats and multi-seed lists suitable for stable comparisons)
- `manuscript`: full native method portfolio, extended secondaries, and stricter reporting expectations

Optional **robustness** blocks in benchmark YAML (`robustness.enabled`, `tracks`,
`severity_levels`) control optional perturbation tracks; when disabled, only the
baseline track runs.

The large `kkbox` dataset is config-only today and requires a custom local
loader before it can participate in benchmark runs.

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
- hyperparameter search uses the configured inner validation budget
- native methods can run Optuna-based HPO when `hpo.enabled: true`
- `comparison_modes` controls whether configs emit `no_hpo`, `hpo`, or both result tracks
- the selected config is refit before outer-test evaluation
- seeds are passed through to stochastic methods

See [`training_strategy.md`](training_strategy.md) for the fold geometry,
no-HPO/HPO training flows, and runtime planning ranges for smoke, standard, and
manuscript-shaped runs.

## Metrics

- discrimination: Harrell C-index, Uno C-index
- overall survival quality: integrated Brier score, Brier score at 25/50/75% event-time horizons, time-dependent AUC
- calibration and utility: median-horizon calibration slope/intercept and median-horizon net benefit
- efficiency: fit time, inference time, peak memory
- manuscript comparison: per-dataset ranks, mean/median rank, pairwise win rate, ELO-style ratings, bootstrap confidence intervals, failure rate, missing-metric rate
- strong inference artifacts: paired significance tests with multiple-comparison correction and critical-difference summaries

Metric computation is backed by `torchsurv`.

## Reproducibility

- split definitions are persisted under `data/splits/<task_id>/`
- split manifests guard against stale splits after dataset or config changes
- benchmark and method config hashes are recorded in run payloads
- experiment manifests capture run-level metadata

## Automated protocol check

`scripts/validate_benchmark_protocol.sh` runs a dry run plus a one-dataset
one-method execution against a benchmark config (default:
`configs/benchmark/smoke.yaml`) and asserts that expected
summary artifacts exist. Set `BENCHMARK_CONFIG`, `WORK_DIR`, or `PYTHON_BIN` to
override defaults.

## Output Contract

Benchmark-style runs write to `results/summary/exp_<YYYYMMDD_HHMMSS>/`.
Stable entry points include:

- `README.md`
- `experiment_navigator.json`
- `experiment_manifest.json`
- fold results, seed summaries, overall summaries, and leaderboards
- dataset curation, failure, and missing-metric summaries when available
- compact run ledgers and ledger indexes
- HPO trial ledgers and summaries when HPO runs
- manuscript summaries and comparison artifacts

`README.md` and `experiment_navigator.json` are the human and machine entry
points for each experiment folder. The compact run ledger is the canonical
comprehensive per-run artifact: shared manifest fields live once in
`<benchmark_id>_run_records_compact_index.json`, while per-run sections live in
`<benchmark_id>_run_records_compact.jsonl.gz`.
Set `exports.write_full_run_ledger: true` in a benchmark config to also emit the
legacy full `<benchmark_id>_run_records.jsonl.gz` and index. Run ledgers and
manuscript summaries include explicit `schema_version` fields.

Manuscript comparison artifacts depend on the configured artifact layout. The
compact layout emits a report, figures, and summary metadata; the full layout
also emits detailed rank, pairwise, significance, ELO, bootstrap, failure, and
missing-metric CSVs.

Fold-result schemas include identifiers, split and mode fields, core and
time-horizon metrics, timing and memory telemetry, HPO governance fields,
robustness/parity fields, retry metadata, and status.
