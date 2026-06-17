# SurvArena

SurvArena is a Python toolkit for right-censored tabular survival analysis. It
supports two everyday workflows:

- **Fit one dataset** with `SurvivalPredictor`, an AutoML-style interface for
  training, ranking, saving, and reloading survival models.
- **Run reproducible benchmarks** from YAML configs with shared splits, fixed
  runtime budgets, compact artifacts, and manuscript-friendly summaries.

The project is aimed at practical model selection: explicit time/event labels,
training-side preprocessing only, comparable validation splits, clear
leaderboards, and disk-first result artifacts.

## Start Here

| Goal | First command | Details |
| --- | --- | --- |
| Install a local environment | `PYTHON_BIN=python3.11 ./scripts/setup_env.sh` | [`docs/environment.md`](docs/environment.md) |
| Try your own CSV or Parquet dataset | `survarena pilot --data train.csv --time-col time --event-col event --dataset-name my_dataset` | [Pilot your own dataset](#pilot-your-own-dataset) |
| Fit and save a predictor | `survarena fit --train train.csv --time-col time --event-col event --dataset-name my_dataset` | [Fit a predictor](#fit-a-predictor) |
| Inspect a benchmark before running it | `survarena benchmark plan --config configs/benchmark/manuscript_v1.yaml` | [`docs/benchmarking_workflow.md`](docs/benchmarking_workflow.md) |
| Run the smallest built-in benchmark slice | `survarena benchmark run --config configs/benchmark/manuscript_v1.yaml --dataset whas500 --method coxph --limit-seeds 1` | [Benchmark runner](#benchmark-runner) |
| Check optional foundation adapters | `survarena foundation-check` | [`docs/foundation_models.md`](docs/foundation_models.md) |

The full documentation map lives in [`docs/index.md`](docs/index.md).

## Repository Layout

```text
survarena/                 Python package
configs/datasets/          Built-in dataset metadata
configs/methods/           Model adapter configurations
configs/benchmark/         Benchmark experiment configurations
docs/                      User, benchmark, protocol, and contributor docs
scripts/                   Environment, validation, and reporting helpers
tests/                     Pytest suite
data/                      Local raw, processed, and split data directories
results/                   Local experiment outputs
```

## Install

Use a repo-local virtual environment. Dependencies include compiled and
modeling-heavy packages such as `scikit-survival`, `torch`, `torchsurv`,
`autogluon.tabular`, `xgboost`, and `catboost`.

```bash
PYTHON_BIN=python3.11 ./scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_environment.py
```

Supported Python versions are 3.10, 3.11, and 3.12; Python 3.11 is preferred.
For manual setup, optional extras, and foundation-model dependency notes, see
[`docs/environment.md`](docs/environment.md).

## Validate the Install

Start with commands that check wiring before fitting many models:

```bash
source .venv/bin/activate

# Confirm imports and metric backends.
python scripts/check_environment.py

# Inspect the maintained benchmark plan without fitting models.
survarena benchmark plan --config configs/benchmark/manuscript_v1.yaml

# Run one small built-in benchmark slice end to end.
survarena benchmark run \
  --config configs/benchmark/manuscript_v1.yaml \
  --dataset whas500 \
  --method coxph \
  --limit-seeds 1
```

For a deeper protocol spot-check, run:

```bash
./scripts/validate_benchmark_protocol.sh
```

Before treating local artifacts as publishable manuscript evidence, run:

```bash
python scripts/audit_manuscript_publishability.py --strict
```

The generated report is [`docs/manuscript_publishability.md`](docs/manuscript_publishability.md).

## Pilot Your Own Dataset

Use `survarena pilot` for a small benchmark-style read before committing to a
larger run:

```bash
survarena pilot \
  --data train.csv \
  --time-col time \
  --event-col event \
  --dataset-name my_dataset
```

The pilot command uses the fast preset by default, evaluates the same data path
as `compare_survival_models(...)`, and prints a compact JSON summary with
aggregate C-index metrics plus artifact paths. Add `--models coxph,rsf` for
explicit model control or `--repeated` for a small 3-fold x 2-repeat pilot.

Input data can be a pandas `DataFrame`, CSV file, or Parquet file. Each dataset
must include:

- a duration column passed as `label_time` or `--time-col`
- an event indicator column passed as `label_event` or `--event-col`
- feature columns usable by the selected adapters

Event labels should indicate whether the event was observed. Duration values
should be positive numeric survival or follow-up times. See
[`docs/datasets.md`](docs/datasets.md) for built-in datasets, user-data notes,
and the dataset metadata contract.

Model IDs for `--models` and `included_models` are listed in
[`docs/methods.md`](docs/methods.md).

## Fit a Predictor

Python API:

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
)

leaderboard = predictor.leaderboard()
risk = predictor.predict_risk("test.csv")
survival = predictor.predict_survival("test.csv")
predictor.save()
```

CLI equivalent:

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
  --dataset-name my_dataset
```

If `tuning_data` is omitted, SurvArena creates a stratified validation holdout.
Set `num_bag_folds >= 2` in Python or `--num-bag-folds` in the CLI for bagged
out-of-fold model selection.

## Compare Models

Use `compare_survival_models(...)` for benchmark-style comparisons on a user
dataset:

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

The maintained manuscript benchmark is
[`configs/benchmark/manuscript_v1.yaml`](configs/benchmark/manuscript_v1.yaml).
It covers the retained built-in dataset suite, native Python survival adapters,
and frozen/bounded foundation adapters in no-HPO/default-policy mode.

Use the staged CLI for expensive runs:

```bash
# Estimate run units, splits, and output layout.
survarena benchmark plan --config configs/benchmark/manuscript_v1.yaml

# Check config shape, dataset/method references, and foundation readiness.
survarena benchmark doctor --config configs/benchmark/manuscript_v1.yaml

# Run one dataset/method slice.
survarena benchmark run \
  --config configs/benchmark/manuscript_v1.yaml \
  --dataset whas500 \
  --method coxph \
  --limit-seeds 1

# Summarize benchmark artifacts.
survarena benchmark report results/manuscript_grade/clinical_no_hpo/elo
```

`python -m survarena.run_benchmark` remains available for batch workers and
scripts. Prefer `--config`; `--benchmark-config` is accepted as a compatibility
alias.

For split geometry, no-HPO versus HPO behavior, output schemas, resume patterns,
and Elo/reporting artifacts, see:

- [`docs/benchmarking_workflow.md`](docs/benchmarking_workflow.md)
- [`docs/protocol.md`](docs/protocol.md)
- [`docs/training_strategy.md`](docs/training_strategy.md)

## Foundation Models

Foundation adapters are optional. Check readiness before including them in long
benchmark runs:

```bash
INSTALL_EXTRAS=dev,foundation PYTHON_BIN=python3.11 ./scripts/setup_env.sh
source .venv/bin/activate
python scripts/check_environment.py --include-foundation
survarena foundation-check
```

For user data, the shortest evaluation path is:

```bash
survarena pilot --data my_survival_data.csv --time-col time --event-col event --foundation
```

See [`docs/foundation_models.md`](docs/foundation_models.md) for adapter status,
skip rules, Hugging Face authentication notes, and manuscript-scope policy.

## Outputs

Predictor artifacts live under `results/predictor/<dataset_name>/` by default:

- `leaderboard.csv`
- `fit_summary.json`
- `predictor.pkl`
- `predictor_manifest.json`
- optional `kaplan_meier_comparison.png`

Benchmark runs write model-prefixed core CSV artifacts and an
`experiment_manifest.json` to a generated results directory or `--output-dir`.
Split definitions are persisted under `data/splits/<task_id>/` so repeated runs
reuse consistent evaluation partitions.

## Development

Install developer dependencies with the setup script or manually:

```bash
python -m pip install -e ".[dev]"
```

Common checks:

```bash
python scripts/check_environment.py
python -m compileall survarena
survarena benchmark run --config configs/benchmark/manuscript_v1.yaml --dry-run
./scripts/validate_benchmark_protocol.sh
pytest
ruff check survarena tests scripts
```

Contribution guides:

- [`docs/contributing_method_adapters.md`](docs/contributing_method_adapters.md)
- [`docs/contributing_datasets.md`](docs/contributing_datasets.md)

## License

SurvArena is released under the MIT License. See [`LICENSE`](LICENSE).
