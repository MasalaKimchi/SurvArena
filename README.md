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
python scripts/check_environment.py
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

### First Smoke Run

After setup, start with commands that check the benchmark wiring before running
many model fits:

```bash
source .venv/bin/activate

# Confirm imports and metric backends.
python scripts/check_environment.py

# Inspect the smoke benchmark plan without fitting models.
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/smoke.yaml \
  --dry-run

# Run the smallest practical built-in benchmark: one dataset, one method,
# one seed/repeat, and the smoke fold geometry.
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/smoke.yaml \
  --dataset whas500 \
  --method coxph \
  --limit-seeds 1
```

The one-dataset smoke run writes a timestamped folder under `results/summary/`
with fold results, a leaderboard, compact run records, and an experiment
manifest. Once this passes, broaden gradually: add more methods, then more
datasets, then move from `smoke.yaml` to `standard_v1.yaml` or a derived
manuscript config.

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

Preset membership and model adapter availability are defined in code and
configuration, not duplicated in this README. Use `presets` for the maintained
default portfolios, or select registered adapters explicitly with
`included_models` in Python and `--models` on the CLI.

Method configs live in `configs/methods/`. Foundation-model details and runtime
readiness checks are documented in
[`docs/foundation_models.md`](docs/foundation_models.md).

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

For a first benchmark run, use `configs/benchmark/smoke.yaml` with `--dataset`,
`--method`, and `--limit-seeds 1` as shown in [First Smoke Run](#first-smoke-run).
The unscoped smoke config is still small relative to standard/manuscript runs,
but it covers all six built-in datasets and every manuscript default method.
For remote execution through Codex, see [Cloud Runs Through Codex](CLOUD_RUN.md).

Tracked benchmark configs:

- `configs/benchmark/standard_v1.yaml`: standard native portfolio (Cox, RSF,
  DeepSurv) on the six built-in standard datasets, repeated nested CV
- `configs/benchmark/manuscript_v1.yaml`: full native manuscript portfolio,
  repeated nested CV
- `configs/benchmark/manuscript_autogluon_v1.yaml`: AutoGluon-only manuscript
  track
- `configs/benchmark/cloud_comprehensive_all_models_hpo.yaml`: cloud-scale
  all-datasets (including `kkbox` when available) and full method matrix with
  AutoGluon and native HPO controls
- `configs/benchmark/smoke.yaml`: small single-seed no-HPO smoke across all
  standard built-in datasets (CI and `scripts/validate_benchmark_protocol.sh`)

To evaluate a **single method** (for example one cloud worker per method), use
`--method` and optionally `--dataset` with `standard_v1.yaml` or
`manuscript_v1.yaml` instead of maintaining one YAML per model.

Example:

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml \
  --dataset support \
  --method coxph \
  --limit-seeds 1
```

Simple smoke examples:

```bash
# Dry run only: parse config and print resolved datasets/methods/modes.
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/smoke.yaml \
  --dry-run

# Tiny end-to-end run.
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/smoke.yaml \
  --dataset whas500 \
  --method coxph \
  --limit-seeds 1

# Slightly broader smoke on one dataset and all smoke methods.
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/smoke.yaml \
  --dataset whas500 \
  --limit-seeds 1
```

For smoke no-HPO runs, SurvArena fits each method's configured defaults directly
on each outer-training split. Inner folds are used when HPO is enabled and a
method has a search space.

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

Resume a partial benchmark run (reusing an output directory):

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/cloud_comprehensive_all_models_hpo.yaml \
  --output-dir results/summary/exp_resume_target \
  --resume \
  --max-retries 2
```

The standard protocol uses shared split definitions, training-side
preprocessing, configured tuning budgets, refit-before-test evaluation, and
seeded stochastic methods. Benchmark configs use `comparison_modes` to choose
`no_hpo`, `hpo`, or both result tracks. See
[`docs/protocol.md`](docs/protocol.md) for the full benchmark contract and
[`docs/training_strategy.md`](docs/training_strategy.md) for fold geometry and
runtime planning.

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

SurvArena writes two main kinds of outputs:

- predictor artifacts for one fitted `SurvivalPredictor`
- benchmark artifacts for multi-method, multi-split comparisons

Predictor artifacts live under `results/predictor/<dataset_name>/` by default.
The most useful files are:

- `leaderboard.csv`: ranked fitted models and metrics
- `fit_summary.json`: model portfolio notes, dataset diagnostics, retained
  models, per-model test metrics, and foundation-model information
- `predictor.pkl`: reloadable predictor object
- `predictor_manifest.json`: reproducibility metadata
- `kaplan_meier_comparison.png`: optional plot when requested

Benchmark runs live under timestamped folders in `results/summary/`. Start with
the generated `README.md` and `experiment_navigator.json`; they point to the
important summaries for that experiment.

Benchmark folders group outputs by purpose:

- performance tables: fold results, seed summaries, overall summaries, and
  leaderboards
- ranking analysis: rank summaries, pairwise win rates, corrected significance
  tests, ELO-style ratings, critical-difference summaries, and bootstrap
  confidence intervals
- run auditing: failure summaries, missing-metric summaries, HPO trial ledgers,
  compact run ledgers, indexes, and experiment manifests

The compact run ledger is the default comprehensive per-run artifact. Benchmark
configs can set `exports.write_full_run_ledger: true` to additionally emit the
legacy full run-record JSONL and index for downstream workflows that still need
the redundant format.

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
./scripts/validate_benchmark_protocol.sh
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
- [Foundation models roadmap](docs/foundation_models.md)
- [Examples](examples/README.md)
- [Project blueprint](blueprint.md)

## License

SurvArena is released under the MIT License. See [`LICENSE`](LICENSE).
