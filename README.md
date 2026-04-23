# SurvArena

SurvArena is a Python toolkit for tabular survival analysis on right-censored
data. It supports two complementary workflows:

- `SurvivalPredictor`: an AutoML-style interface for fitting survival models on
  a user dataset
- the benchmark runner: a config-driven workflow for reproducible, shared-split
  method comparisons

The project is designed around practical survival modeling workflows: explicit
time/event labels, consistent preprocessing, comparable validation splits,
leaderboards, persisted artifacts, and manuscript-friendly benchmark summaries.

## Features

- Fit from a pandas `DataFrame`, CSV file, or Parquet file.
- Validate duration and event labels before training.
- Infer feature types and apply training-side preprocessing only.
- Select models with presets or explicit model ids.
- Use automatic validation holdouts, explicit tuning sets, or bagged
  out-of-fold selection.
- Rank fitted models with a unified leaderboard and optional test metrics.
- Save and reload predictors for later inference.
- Predict risk scores and survival curves.
- Plot Kaplan-Meier comparisons for quick model inspection.
- Run benchmark-style comparisons on built-in or user-provided datasets.
- Export fold results, seed summaries, run ledgers, rank summaries, bootstrap
  confidence intervals, and experiment manifests.
- Inspect optional tabular foundation-model readiness before fitting.

## Repository Layout

```text
survarena/                 Python package
configs/datasets/          Built-in dataset metadata
configs/methods/           Model adapter configurations
configs/benchmark/         Benchmark experiment configurations
docs/                      Environment, protocol, dataset, and backend docs
examples/                  Notebook and sample predictor artifacts
scripts/                   Environment setup and validation helpers
tests/                     Pytest suite
data/                      Local raw, processed, and split data directories
```

## Python Environment

SurvArena is tested for modern CPython environments:

- preferred: Python 3.11
- supported by the setup script: Python 3.10, 3.11, and 3.12
- package metadata: `requires-python = ">=3.10"`

Use a repo-local virtual environment. The dependencies include compiled and
modeling-heavy packages such as `scikit-survival`, `torch`, `torchsurv`,
`autogluon.tabular`, `xgboost`, and `catboost`, so isolated environments are
strongly recommended.

### Recommended Setup

```bash
PYTHON_BIN=python3.11 ./scripts/setup_env.sh
source .venv/bin/activate
```

The setup script creates `.venv`, upgrades `pip`, installs SurvArena in editable
mode with developer tooling by default, and runs the environment check.

Useful setup overrides:

```bash
# Use a different supported interpreter.
PYTHON_BIN=python3.10 ./scripts/setup_env.sh
PYTHON_BIN=python3.12 ./scripts/setup_env.sh

# Use a different virtual environment directory.
VENV_DIR=.venv311 PYTHON_BIN=python3.11 ./scripts/setup_env.sh

# Install optional foundation-model extras.
INSTALL_EXTRAS=dev,foundation PYTHON_BIN=python3.11 ./scripts/setup_env.sh
INSTALL_EXTRAS=dev,foundation-tabpfn PYTHON_BIN=python3.11 ./scripts/setup_env.sh
INSTALL_EXTRAS=dev,foundation-mitra PYTHON_BIN=python3.11 ./scripts/setup_env.sh
```

### Manual Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

Core package only:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[foundation]"
python -m pip install -e ".[foundation-tabpfn]"
python -m pip install -e ".[foundation-mitra]"
python -m pip install -e ".[tracking]"
```

### Validate the Environment

```bash
python scripts/check_environment.py
python scripts/check_environment.py --include-foundation
survarena foundation-check
```

The environment check reports the active Python executable, virtual environment
status, core imports, optional foundation imports, foundation runtime readiness,
and smoke checks for the `torchsurv` metrics used by SurvArena.

## Quick Start

### Python API

```python
from survarena import SurvivalPredictor

predictor = SurvivalPredictor(
    label_time="time",
    label_event="event",
    presets="medium",
    eval_metric="uno_c",
    retain_top_k_models=2,
)

predictor.fit(
    train_data="train.csv",
    tuning_data="valid.csv",
    test_data="test.csv",
    dataset_name="my_dataset",
    time_limit=1800,
    hyperparameter_tune_kwargs={"num_trials": 12, "timeout": 120},
    refit_full=True,
    num_bag_folds=5,
)

leaderboard = predictor.leaderboard()
summary = predictor.fit_summary()
risk = predictor.predict_risk("test.csv")
survival = predictor.predict_survival("test.csv")
predictor.plot_kaplan_meier_comparison("test.csv")
predictor.save()
```

If `tuning_data` is omitted, SurvArena creates a stratified validation holdout.
Set `num_bag_folds >= 2` to use bagged out-of-fold model selection.

### Command Line

```bash
survarena fit \
  --train train.csv \
  --tuning valid.csv \
  --test test.csv \
  --time-col time \
  --event-col event \
  --presets medium \
  --retain-top-k-models 2 \
  --time-limit 1800 \
  --autogluon-num-trials 12 \
  --tuning-timeout 120 \
  --num-bag-folds 5 \
  --dataset-name my_dataset
```

The CLI prints the fit summary JSON after training and writes predictor
artifacts to disk.

## Data Requirements

Input data can be provided as:

- a pandas `DataFrame`
- a CSV file
- a Parquet file

Each dataset must include:

- a duration column, passed as `label_time` or `--time-col`
- an event indicator column, passed as `label_event` or `--event-col`
- feature columns usable by the selected model adapters

Event labels should indicate whether the event was observed. Duration values
should be positive numeric survival or follow-up times. Optional id columns or
columns that should not be modeled can be removed before fitting or passed to
the compare API as drop columns.

Built-in dataset configs live in `configs/datasets/`. See
[`docs/datasets.md`](docs/datasets.md) for the current benchmark datasets and
metadata contract.

## Presets and Models

Built-in presets:

- `fast`: `coxph`, `rsf`
- `medium`: `coxph`, `coxnet`, `rsf`, `deepsurv`
- `best`: `medium` plus `deepsurv_moco`
- `all`: `best` plus eligible foundation adapters
- `foundation`: `coxph` plus eligible foundation adapters
- `autogluon`: AutoGluon-backed survival adapter

Registered adapters can also be selected explicitly with `included_models` in
Python or `--models` on the CLI. Method configs are stored in
`configs/methods/` and cover classical survival models, tree methods, boosting,
pycox deep models, XGBoost/CatBoost variants, AutoGluon-backed fitting, and
experimental foundation-model adapters.

Use `enable_foundation_models=True` or `--enable-foundation-models` to let
non-foundation presets consider eligible foundation adapters.

## Compare API

Use `compare_survival_models(...)` for benchmark-style comparisons on a user
dataset.

```python
from survarena import compare_survival_models

summary = compare_survival_models(
    "train.csv",
    time_col="time",
    event_col="event",
    dataset_name="my_dataset",
    models=["coxph", "rsf", "deepsurv"],
    split_strategy="fixed_split",
    seeds=[11],
)
```

CLI equivalent:

```bash
survarena compare \
  --data train.csv \
  --time-col time \
  --event-col event \
  --dataset-name my_dataset \
  --models coxph,rsf,deepsurv \
  --split-strategy fixed_split \
  --seeds 11
```

`fixed_split` is the quick path. Use `repeated_nested_cv` for stricter
benchmark-style evaluation with shared outer and inner splits.

## Benchmark Runner

Tracked benchmark configs:

- `configs/benchmark/standard_v1.yaml`: repeated nested CV on `support`,
  `metabric`, `aids`, `gbsg2`, `flchain`, and `whas500`
- `configs/benchmark/large_v1.yaml`: fixed-split placeholder for `kkbox`
- `configs/benchmark/cloud_comprehensive_all_models_hpo.yaml`: cloud-scale
  all-datasets/all-models run with explicit AutoGluon HPO controls
- `configs/benchmark/models/`: one-method benchmark configs, including
  AutoGluon-backed default and tuned variants

Example:

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml \
  --dataset support \
  --method coxph \
  --limit-seeds 1
```

Cloud-scale comprehensive run helper:

```bash
scripts/run_cloud_comprehensive.sh
# optionally scope a single cloud worker
DATASET=support METHOD=rsf scripts/run_cloud_comprehensive.sh
```

Dry-run a benchmark configuration without fitting models:

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml \
  --dry-run
```

The standard protocol uses shared split definitions, training-side
preprocessing, configured tuning budgets, refit-before-test evaluation, and
seeded stochastic methods. See [`docs/protocol.md`](docs/protocol.md) for the
full benchmark contract.

## Foundation Models

Currently wired foundation adapters:

- `tabpfn_survival`
- `mitra_survival`

Install and inspect foundation support:

```bash
INSTALL_EXTRAS=dev,foundation PYTHON_BIN=python3.11 ./scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_environment.py --include-foundation
survarena foundation-check
```

TabPFN requires access to the gated model on Hugging Face:

- accept the terms for
  [Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- authenticate with `hf auth login` or set `HF_TOKEN` /
  `HUGGINGFACE_HUB_TOKEN`

Foundation adapters are experimental. Check runtime readiness before including
them in long benchmark runs.

## Outputs and Artifacts

Predictor runs write to `results/predictor/<dataset_name>/` by default:

- `leaderboard.csv`
- `fit_summary.json`
- `predictor.pkl`
- `predictor_manifest.json`
- `kaplan_meier_comparison.png` when requested

`fit_summary.json` includes portfolio notes, dataset diagnostics, retained
models, per-model test metrics, and the foundation-model catalog.

Benchmark runs write timestamped experiment folders under `results/summary/`.
Each experiment can include:

- fold results
- seed summaries
- overall summaries
- leaderboards
- rank summaries
- pairwise win rates
- ELO-style ratings
- bootstrap confidence intervals
- failure and missing-metric summaries
- compressed run ledgers
- experiment manifests

Split definitions are persisted under `data/splits/<task_id>/` so repeated runs
can reuse consistent evaluation partitions.

## Development

Install developer dependencies with the recommended setup script or manually:

```bash
python -m pip install -e ".[dev]"
```

Common checks:

```bash
python scripts/check_environment.py
python -m compileall survarena
python -m survarena.run_benchmark --dry-run
pytest
ruff check survarena tests scripts
```

The default `requirements.txt` installs the editable package with `dev` extras:

```bash
python -m pip install -r requirements.txt
```

## Documentation

- [Environment](docs/environment.md)
- [Benchmark protocol](docs/protocol.md)
- [Datasets](docs/datasets.md)
- [AutoGluon backend](docs/autogluon_backend.md)
- [AutoGluon comparison notes](docs/autogluon_comparison.md)
- [Foundation-model TODO](docs/tabular_foundation_models_todo.md)
- [Examples](examples/README.md)
- [Project blueprint](blueprint.md)

## License

SurvArena is released under the MIT License. See [`LICENSE`](LICENSE).
