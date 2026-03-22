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
- validate `time` and `event` labels and infer feature metadata for numerical, categorical, datetime, and text columns
- surface dataset diagnostics such as low-event warnings, ID-like features, and high-cardinality columns
- fit a preset-driven model portfolio and rank candidates with a unified leaderboard
- return the best model, model-specific predictions, survival curves, fit summaries, and persisted predictor artifacts
- stay quiet by default for notebook use, with `verbose=True` available when you want tuning logs

Available predictor presets:

- `fast`: `coxph`, `rsf`
- `medium`: `coxph`, `coxnet`, `rsf`, `deepsurv`
- `best`: `coxph`, `coxnet`, `rsf`, `deepsurv`, `deepsurv_moco`
- `foundation`: starts with `coxph` and adds eligible foundation-model adapters

Experimental foundation-model support is available through:

- `tabpfn_survival`
- `mitra_survival`

Those adapters are optional, depend on extra packages already listed in `requirements.txt`, and are
added only when dataset-shape and dependency checks pass.

### Benchmark workflow

The benchmark runner currently covers:

- datasets: `support`, `metabric`, `aids`, `gbsg2`, `flchain`, `whas500`, `pbc`
- a placeholder config track for `kkbox`
- methods: `coxph`, `coxnet`, `rsf`, `deepsurv`, `deepsurv_moco`, `tabpfn_survival`, `mitra_survival`
- metrics: Harrell's C-index, Uno's C-index, integrated Brier score, and time-dependent AUC
- protocol infrastructure for repeated nested CV, shared seeds, manifests, and aggregated summaries

## Installation

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

If you want the `survarena` shell command in addition to `python -m ...` entrypoints:

```bash
python -m pip install -e .
```

## Quick Start

### Python predictor API

In this repo layout, import the predictor from `src`:

```python
from src import SurvivalPredictor

predictor = SurvivalPredictor(
    label_time="time",
    label_event="event",
    eval_metric="harrell_c",
    presets="medium",
)

predictor.fit(
    train_data="my_train.csv",
    test_data="my_test.csv",
    dataset_name="my_dataset",
)

leaderboard = predictor.leaderboard()
risk_scores = predictor.predict_risk("my_test.csv")
survival_curves = predictor.predict_survival("my_test.csv")
summary = predictor.fit_summary()
predictor.plot_kaplan_meier_comparison("my_test.csv")
predictor.save()
```

### CLI predictor API

Repo-local invocation:

```bash
python -m src.cli fit \
  --train my_train.csv \
  --test my_test.csv \
  --time-col time \
  --event-col event \
  --presets medium \
  --dataset-name my_dataset
```

After `python -m pip install -e .`, the equivalent console command is:

```bash
survarena fit \
  --train my_train.csv \
  --test my_test.csv \
  --time-col time \
  --event-col event \
  --presets medium \
  --dataset-name my_dataset
```

### Benchmark runner

```bash
python -m src.run_benchmark \
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
- resolved portfolio and portfolio notes
- dataset diagnostics
- per-model test metrics when test data is provided
- foundation-model catalog metadata

## Benchmark Artifacts

The benchmark engine writes:

- persisted splits to `data/splits/`
- split manifests to `data/splits/<task_id>/manifest.json`
- compact per-run ledgers to `results/runs/`
- aggregate summaries and tables to `results/summaries/` and `results/tables/`

Only `results/summaries/` is intended for git tracking; run ledgers and table exports are local artifacts.

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

1. Implement `BaseSurvivalMethod` in `src/methods/`.
2. Add a method config in `configs/methods/`.
3. Register the method in `src/run_benchmark.py`.
4. Ensure `predict_risk` and `predict_survival` return standardized outputs.

## License

MIT (see `LICENSE`).
