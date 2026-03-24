# SurvArena

SurvArena is a tabular survival toolkit for right-censored data.
It has two main workflows:

- `SurvivalPredictor` for AutoML-style fitting on your own dataset
- a config-driven benchmark runner for reproducible method comparisons

## What You Get

- fit from a pandas `DataFrame`, CSV, or Parquet file
- validate `time` and `event` labels and infer feature types
- use presets or explicit model lists
- choose an automatic holdout, explicit tuning split, or bagged OOF selection
- rank models with a unified leaderboard and optional test metrics
- save/load predictors, export summaries, and plot Kaplan-Meier comparisons
- compare models on a user dataset with `compare_survival_models`
- inspect optional foundation-model readiness before fitting

## Install

Recommended setup:

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

Optional extras:

```bash
INSTALL_EXTRAS=dev,foundation ./scripts/setup_env.sh
INSTALL_EXTRAS=dev,foundation-tabpfn ./scripts/setup_env.sh
INSTALL_EXTRAS=dev,foundation-mitra ./scripts/setup_env.sh
```

Core library only:

```bash
python -m pip install -e .
```

## Quick Start

### Python

```python
from survarena import SurvivalPredictor

predictor = SurvivalPredictor(
    label_time="time",
    label_event="event",
    presets="medium",
    eval_metric="harrell_c",
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
Set `num_bag_folds >= 2` for bagged OOF selection.

### CLI

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
  --num-trials 12 \
  --tuning-timeout 120 \
  --num-bag-folds 5 \
  --dataset-name my_dataset
```

## Presets and Models

Built-in presets:

- `fast`: `coxph`, `rsf`
- `medium`: `coxph`, `coxnet`, `rsf`, `deepsurv`
- `best`: `medium` plus `deepsurv_moco`
- `all`: `best` plus eligible foundation adapters
- `foundation`: `coxph` plus eligible foundation adapters

More registered adapters can be selected with `included_models` or `--models`.
The current registry includes classical, tree, boosting, pycox, XGBoost/CatBoost,
and experimental foundation-model adapters defined in [`configs/methods/`](configs/methods/).

Use `enable_foundation_models=True` to let `fast`, `medium`, or `best` consider
eligible foundation adapters.

## Compare API

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
    n_trials=0,
)
```

`fixed_split` is the quick path. Use `repeated_nested_cv` for a stricter benchmark-style run.

## Foundation Models

Currently wired adapters:

- `tabpfn_survival`
- `mitra_survival`

Check runtime readiness before fitting:

```bash
survarena foundation-check
python scripts/check_environment.py --include-foundation
```

TabPFN note:

- accept the gated model terms at [Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5)
- run `hf auth login`, or set `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`

## Benchmark Runner

Tracked benchmark configs:

- `configs/benchmark/standard_v1.yaml`: repeated nested CV on `support`, `metabric`, `aids`, `gbsg2`, `flchain`, `whas500`
- `configs/benchmark/large_v1.yaml`: fixed-split placeholder for `kkbox`

Example:

```bash
python -m survarena.run_benchmark \
  --benchmark-config configs/benchmark/standard_v1.yaml \
  --dataset support \
  --method coxph \
  --limit-seeds 1 \
  --n-trials 2
```

## Artifacts

Predictor runs write to `results/predictor/<dataset_name>/`:

- `leaderboard.csv`
- `fit_summary.json`
- `predictor.pkl`
- `predictor_manifest.json`
- `kaplan_meier_comparison.png` when requested

Benchmark runs write timestamped experiment folders under `results/summary/`.
Each run includes fold results, seed summaries, leaderboards, a run ledger,
and an experiment manifest.

`fit_summary.json` includes portfolio notes, dataset diagnostics, retained
models, per-model test metrics, and the foundation-model catalog.

## Docs

- [docs/protocol.md](docs/protocol.md)
- [docs/datasets.md](docs/datasets.md)
- [docs/environment.md](docs/environment.md)
- [blueprint.md](blueprint.md)
- [docs/autogluon_comparison.md](docs/autogluon_comparison.md)
- [docs/tabular_foundation_models_todo.md](docs/tabular_foundation_models_todo.md)
- [examples/README.md](examples/README.md)

## License

MIT. See [`LICENSE`](LICENSE).
