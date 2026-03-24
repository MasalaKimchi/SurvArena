# SurvArena

SurvArena is a reproducible framework for tabular right-censored survival modeling.
It combines two workflows in one repo:

- a high-level `SurvivalPredictor` API for fitting survival models on user-owned tabular data
- a config-driven benchmark runner for fair, repeatable method comparisons on built-in datasets

The project aims to feel like AutoML for survival analysis without dropping the benchmark rigor
needed for research and reproducibility.

## What SurvArena Supports Today

### User-facing predictor workflow

The current predictor stack can:

- read training and test data from a pandas `DataFrame`, CSV, or Parquet file
- accept explicit tuning/validation data or create an automatic stratified holdout when tuning data is not provided
- switch to bagged out-of-fold model selection with `num_bag_folds` / `num_bag_sets` when you want a stronger AutoML-style fit flow
- validate `time` and `event` labels and infer feature metadata for numerical, categorical, datetime, and text columns
- surface dataset diagnostics such as low-event warnings, ID-like features, and high-cardinality columns
- fit a preset-driven or explicitly selected model portfolio and rank candidates with a unified leaderboard
- retain only the top-ranked fitted models by default, with `retain_top_k_models` or CLI flags available when you want a wider retained portfolio
- return the best model, optional model-specific predictions, survival curves, fit summaries, and persisted predictor artifacts
- stay quiet by default for notebook use, with `verbose=True` available when you want tuning logs

Available predictor presets:

- `fast`: `coxph`, `rsf`
- `medium`: `coxph`, `coxnet`, `rsf`, `deepsurv`
- `best`: `coxph`, `coxnet`, `rsf`, `deepsurv`, `deepsurv_moco`
- `all` (default): `best` portfolio plus eligible foundation adapters
- `foundation`: starts with `coxph` and adds eligible foundation-model adapters

Additional opt-in model adapters are available through explicit `included_models`
or `--models` selection:

- `weibull_aft`, `lognormal_aft`, `loglogistic_aft`: parametric AFT regressors from lifelines
- `aalen_additive`: additive hazards regression from lifelines with regularization
- `fast_survival_svm`: large-margin ranking model with a calibrated survival head
- `gradient_boosting_survival`: gradient-boosted Cox model with tree base learners
- `componentwise_gradient_boosting`: sparse component-wise Cox boosting
- `extra_survival_trees`: extremely randomized survival trees
- `xgboost_cox`, `catboost_cox`: conventional tabular gradient boosting backends with calibrated Cox survival curves
- `xgboost_aft`, `catboost_survival_aft`: boosted AFT regressors with distributional survival curves
- `logistic_hazard`, `pmf`, `mtlr`, `deephit_single`, `pchazard`, `cox_time`: neural survival methods from pycox

Experimental foundation-model support is available through:

- `tabpfn_survival`
- `mitra_survival`

Those adapters are optional and can be installed either together or one by one:

- `foundation`: installs both foundation adapters
- `foundation-tabpfn`: installs only `tabpfn_survival`
- `foundation-mitra`: installs only `mitra_survival`

They are added to predictor presets only when dataset-shape and runtime checks pass.

### TabPFN access setup (required)

TabPFN v2.5 checkpoints are gated on Hugging Face. Before using `tabpfn_survival`, complete:

1. Accept model terms at [https://huggingface.co/Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5).
2. Authenticate with one of:
   - `hf auth login`
   - `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` environment variable

If authentication is missing, SurvArena will fail the `tabpfn_survival` fit with a Hugging Face gated-model error.

### Benchmark workflow

The benchmark runner currently covers:

- datasets: `support`, `metabric`, `aids`, `gbsg2`, `flchain`, `whas500`
- a placeholder config track for `kkbox`
- default benchmark-track methods: `coxph`, `coxnet`, `rsf`, `deepsurv`, `deepsurv_moco`, `tabpfn_survival`, `mitra_survival`
- additional registered methods for explicit benchmark configs: `weibull_aft`, `lognormal_aft`, `loglogistic_aft`, `aalen_additive`, `fast_survival_svm`, `gradient_boosting_survival`, `componentwise_gradient_boosting`, `extra_survival_trees`, `xgboost_cox`, `xgboost_aft`, `catboost_cox`, `catboost_survival_aft`, `logistic_hazard`, `pmf`, `mtlr`, `deephit_single`, `pchazard`, `cox_time`
- metrics: Harrell's C-index, Uno's C-index, integrated Brier score, and time-dependent AUC
- protocol infrastructure for repeated nested CV, shared seeds, manifests, and aggregated summaries

## Installation

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

That setup installs the package in editable mode plus the `dev` extra.

To include the optional foundation-model stack:

```bash
INSTALL_EXTRAS=dev,foundation ./scripts/setup_env.sh
```

To install only one optional backbone:

```bash
INSTALL_EXTRAS=dev,foundation-tabpfn ./scripts/setup_env.sh
INSTALL_EXTRAS=dev,foundation-mitra ./scripts/setup_env.sh
```

To inspect whether foundation adapters are installed and ready before fitting:

```bash
survarena foundation-check
python scripts/check_environment.py --include-foundation
```

If you only want the core library without repo tooling:

```bash
python -m pip install -e .
```

## Quick Start

### Python predictor API

```python
from survarena import SurvivalPredictor

predictor = SurvivalPredictor(
    label_time="time",
    label_event="event",
    eval_metric="harrell_c",
    presets="medium",
    retain_top_k_models=2,
)

predictor.fit(
    train_data="my_train.csv",
    tuning_data="my_validation.csv",
    test_data="my_test.csv",
    dataset_name="my_dataset",
    time_limit=1800,
    hyperparameter_tune_kwargs={"num_trials": 12, "timeout": 120},
    refit_full=True,
    num_bag_folds=5,
)

leaderboard = predictor.leaderboard()
risk_scores = predictor.predict_risk("my_test.csv")
survival_curves = predictor.predict_survival("my_test.csv")
summary = predictor.fit_summary()
predictor.plot_kaplan_meier_comparison("my_test.csv")
predictor.save()
```

If `tuning_data` is omitted, `SurvivalPredictor.fit(...)` automatically creates a stratified validation holdout using the preset default or an explicit `holdout_frac=...` override.
When `time_limit` is provided, SurvArena treats it as an approximate wallclock budget for the overall fit and allocates the selection budget across the remaining candidate models.
Use `hyperparameter_tune_kwargs` to override fit-level HPO controls such as `num_trials` and per-model tuning timeout, and set `refit_full=False` if you want retained models to stay on the selection-train portion instead of refitting on all available non-test data.
Set `num_bag_folds >= 2` to replace the single holdout with bagged OOF selection and averaged fold-model inference; `num_bag_sets` repeats that fold schedule for a stronger, slower fit.
Use `included_models` / `excluded_models` to take explicit control of which methods run, and set `retain_top_k_models=None` when you want to keep every successful candidate instead of only the top-ranked models.

### CLI predictor API

Repo-local invocation:

```bash
python -m survarena.cli fit \
  --train my_train.csv \
  --tuning my_validation.csv \
  --test my_test.csv \
  --time-col time \
  --event-col event \
  --models coxph,rsf,deepsurv \
  --retain-top-k-models 2 \
  --time-limit 1800 \
  --tuning-timeout 120 \
  --num-trials 12 \
  --num-bag-folds 5 \
  --dataset-name my_dataset
```

After `python -m pip install -e .`, the equivalent console command is:

```bash
survarena fit \
  --train my_train.csv \
  --tuning my_validation.csv \
  --test my_test.csv \
  --time-col time \
  --event-col event \
  --models coxph,rsf,deepsurv \
  --retain-top-k-models 2 \
  --time-limit 1800 \
  --tuning-timeout 120 \
  --num-trials 12 \
  --num-bag-folds 5 \
  --dataset-name my_dataset
```

### Compare API

Python:

```python
from survarena import compare_survival_models

summary = compare_survival_models(
    "my_train.csv",
    time_col="time",
    event_col="event",
    dataset_name="my_dataset",
    models=["coxph", "rsf", "deepsurv"],
    split_strategy="fixed_split",
    seeds=[11],
    n_trials=0,
)
```

CLI:

```bash
survarena compare \
  --data my_train.csv \
  --time-col time \
  --event-col event \
  --models coxph,rsf,deepsurv \
  --split-strategy fixed_split \
  --seeds 11 \
  --n-trials 0 \
  --save-path results/summary/my_dataset_compare
```

Use `fixed_split` for the quickest benchmark-style comparison on a user dataset, or switch to `repeated_nested_cv` when you want a more rigorous, slower protocol with multiple seeds and outer folds.

### Benchmark runner

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml \
  --dataset support \
  --method coxph \
  --limit-seeds 1 \
  --n-trials 2
```

## Predictor Artifacts

By default, `SurvivalPredictor.fit(...)` writes artifacts to `results/predictor/<dataset_name>/`:

- `leaderboard.csv`
- `fit_summary.json`
- `predictor.pkl`
- `predictor_manifest.json`
- `kaplan_meier_comparison.png` when `plot_kaplan_meier_comparison(...)` is called

The fit summary includes:

- best method and best params
- validation strategy details, including whether explicit tuning data or an automatic holdout was used
- bagging details, including the requested number of bag folds and bag sets when OOF selection is enabled
- refit strategy details, including whether validation rows were folded back into final training
- time-budget metadata such as the requested fit budget and observed elapsed fit time
- resolved portfolio and portfolio notes
- dataset diagnostics
- per-model test metrics when test data is provided
- foundation-model catalog metadata

## Benchmark Artifacts

The benchmark engine writes:

- persisted splits to `data/splits/`
- split manifests to `data/splits/<task_id>/manifest.json`
- experiment outputs to `results/summary/exp_<YYYYMMDD_HHMMSS>/`
  - `<benchmark_id>_fold_results.csv`
  - `<benchmark_id>_seed_summary.csv`
  - `<benchmark_id>_overall_summary.json`
  - `<benchmark_id>_leaderboard.csv`
  - `<benchmark_id>_leaderboard.json`
  - `<benchmark_id>_run_records.jsonl.gz`
  - `<benchmark_id>_run_records_index.json`
  - `experiment_manifest.json`

The timestamped experiment directory keeps runs easy to compare and prevents accidental overwrite.

## Documentation

- **Protocol, fairness, artifacts, and reproducibility:** [`docs/protocol.md`](docs/protocol.md)
- **Datasets and metadata:** [`docs/datasets.md`](docs/datasets.md)
- **Environment setup and smoke checks:** [`docs/environment.md`](docs/environment.md)
- **Design blueprint:** [`blueprint.md`](blueprint.md)
- **AutoML positioning notes:** [`docs/autogluon_comparison.md`](docs/autogluon_comparison.md)
- **Tabular foundation-model roadmap:** [`docs/tabular_foundation_models_todo.md`](docs/tabular_foundation_models_todo.md)

## Examples

- **Quickstart notebook:** [`examples/survival_predictor_quickstart.ipynb`](examples/survival_predictor_quickstart.ipynb)
- **Examples overview and shipped sample artifacts:** [`examples/README.md`](examples/README.md)

## Current Gaps and Next Steps

SurvArena still has room to grow, especially around:

- ensembling, bagging, and richer AutoML orchestration
- stronger artifact management for all trained models
- more adaptive portfolio search and runtime budgeting
- broader foundation-model coverage beyond the currently wired adapters
- more large-cohort dataset integrations such as a real KKBox loader

## Add a New Method

1. Implement `BaseSurvivalMethod` in `survarena/methods/`.
2. Add a method config in `configs/methods/`.
3. Register the method in `survarena/methods/registry.py`.
4. Ensure `predict_risk` and `predict_survival` return standardized outputs.

## License

MIT (see `LICENSE`).
