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
| Install the locked local environment | `uv sync --locked` | [`docs/environment.md`](docs/environment.md) |
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

Install the project-pinned uv release, then create the repo-local environment
from the committed developer/CI lock. Dependencies include compiled and
modeling-heavy packages such as `scikit-survival`, `torch`, `torchsurv`,
`autogluon.tabular`, `xgboost`, and `catboost`.

```bash
curl -LsSf https://astral.sh/uv/0.12.8/install.sh | sh
uv sync --locked
uv run --no-sync python scripts/check_environment.py
```

Supported Python versions are 3.10, 3.11, and 3.12; Python 3.11 is preferred.
The checked-in `.python-version` selects 3.11 when no interpreter is supplied.
For optional extras, the pip compatibility path, and foundation-model
dependency notes, see
[`docs/environment.md`](docs/environment.md).

## Validate the Install

Start with commands that check wiring before fitting many models:

```bash
# Confirm imports and metric backends.
uv run --no-sync python scripts/check_environment.py

# Inspect the maintained benchmark plan without fitting models.
uv run --no-sync survarena benchmark plan --config configs/benchmark/manuscript_v1.yaml

# Run one small built-in benchmark slice end to end.
uv run --no-sync survarena benchmark run \
  --config configs/benchmark/manuscript_v1.yaml \
  --dataset whas500 \
  --method coxph \
  --limit-seeds 1
```

For a deeper protocol spot-check, run:

```bash
uv run --no-sync ./scripts/validate_benchmark_protocol.sh
```

Before treating local artifacts as publishable manuscript evidence, run:

```bash
uv run --no-sync python scripts/audit_manuscript_publishability.py --strict
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
uv sync --locked --extra foundation
uv run --no-sync python scripts/check_environment.py --include-foundation
uv run --no-sync survarena foundation-check
```

For user data, the shortest evaluation path is:

```bash
uv run --no-sync survarena pilot --data my_survival_data.csv --time-col time --event-col event --foundation
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

The primary developer path uses the committed lock. The default `dev`
dependency group includes the repository's Ruff, mypy, and pytest versions.

```bash
uv sync --locked
uv lock --check
```

Common checks:

```bash
uv run --no-sync ruff check survarena tests scripts
uv run --no-sync python -m mypy survarena/core survarena/benchmark/resume.py survarena/data/splitters.py scripts/audit_manuscript_publishability.py
uv run --no-sync python -m pytest -q
uv run --no-sync python -m compileall -q survarena
uv run --no-sync python scripts/audit_manuscript_publishability.py --strict
uv sync --locked --only-group docs --python 3.11
uv run --no-sync sphinx-build -n -W --keep-going -b html docs docs/_build/html
```

After an explicit sync, `uv run --no-sync` guarantees that checks use the
installed lock state without silently changing it. `scripts/setup_env.sh` and
`python -m pip install -e ".[dev]"` remain supported compatibility paths, but
they are not lock-equivalent workflows.

The mypy gate is an explicit incremental kernel scope with imported legacy modules skipped, not a whole-package type claim. The strict audit currently exits 2 with `publishable=false` because historical matrices predate behavior-changing fixes; a traceback is a verification failure.

The same baseline is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml), with the semantic WHAS500/CoxPH check in [`.github/workflows/benchmark-smoke.yml`](.github/workflows/benchmark-smoke.yml). See [`docs/test_status.md`](docs/test_status.md) for observed local evidence, [`docs/ci_and_reproducibility.md`](docs/ci_and_reproducibility.md) for configured-versus-observed automation, and [the v2 roadmap](.planning/ROADMAP.md) for the `survbench-1.0-rc1` target.

Contribution guides:

- [`docs/contributing_method_adapters.md`](docs/contributing_method_adapters.md)
- [`docs/contributing_datasets.md`](docs/contributing_datasets.md)

## License

SurvArena is released under the MIT License. See [`LICENSE`](LICENSE).
