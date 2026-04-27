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
`configs/benchmark/smoke_aft.yaml` is the targeted AFT adapter stability smoke:
it runs Weibull, LogNormal, LogLogistic, XGBoost AFT, and CatBoost AFT across
all standard built-in datasets with paired no-HPO and one-trial HPO modes.

Benchmark profiles (see `validate_benchmark_profile_contract` in
`survarena/benchmark/runner.py`):

- `smoke`: small folds/repeats/seeds for CI and quick checks, not for statistical claims
- `standard`: balanced rigor for routine method iteration (outer folds/repeats and multi-seed lists suitable for stable comparisons)
- `manuscript`: full native method portfolio, extended secondaries, and stricter reporting expectations

## Manuscript Scope

The main-paper benchmark scope is `configs/benchmark/manuscript_v1.yaml`: native
Python survival adapters, shared repeated nested CV, and no-HPO/default-policy
comparison. Paired no-HPO/HPO analysis is a budget/sensitivity track represented
by `configs/benchmark/standard_v1.yaml` or by an explicitly named future config,
not by the main manuscript config.

`configs/benchmark/manuscript_autogluon_v1.yaml` is an appendix AutoGluon track
with AutoGluon-managed HPO, bagging, stacking, and refit. Foundation adapters are
exploratory readiness checks in `configs/benchmark/smoke.yaml` and can be
isolated with `configs/benchmark/smoke_foundation.yaml`. They run as frozen
pretrained tabular feature extractors plus lightweight survival heads by
default, and do not support main-paper claims until a separate manuscript-grade
foundation config and evidence bundle are created.

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
- failed or non-finite Optuna candidates are recorded as invalid trial results,
  and selection keeps the best valid incumbent, including defaults when no
  sampled candidate improves them
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

Benchmark-style runs write to
`results/summary/<dataset_id>/<benchmark_id>/<model_name>/`.
If that model folder already contains CSV artifacts, a timestamp suffix is
added (`<model_name>_<YYYYMMDD_HHMMSS>`) to prevent overwrites.
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
