# SurvArena

SurvArena is a concise, reproducible framework for tabular right-censored survival modeling.
It pairs a rigorous benchmark engine with a simple AutoML-style user experience so teams can
both compare methods fairly and run strong baselines on their own data with minimal setup.

## Product Direction

SurvArena should feel like **AutoGluon for tabular survival analysis**:

- bring your own CSV, Parquet file, or DataFrame
- specify `time` and `event` columns once
- automatically preprocess features and validate labels
- train and compare a portfolio of survival models under one interface
- return a leaderboard, the best model, risk scores, and survival curves

The key design principle is a two-layer workflow:

- **Simple mode:** a high-level `SurvivalPredictor` API for user-owned datasets
- **Research mode:** the current config-driven benchmark runner for reproducible comparisons

This keeps the benchmark rigor already present in SurvArena while making the framework much
easier for practitioners to adopt on private datasets.

## Target User Experience

```python
from survarena import SurvivalPredictor

predictor = SurvivalPredictor(
    label_time="time",
    label_event="event",
    eval_metric="harrell_c",
    presets="medium",
)

predictor.fit(train_data="my_data.csv")
leaderboard = predictor.leaderboard()
pred_risk = predictor.predict_risk("my_test.csv")
pred_surv = predictor.predict_survival("my_test.csv")
```

```bash
survarena fit \
  --train my_data.csv \
  --time-col time \
  --event-col event \
  --presets medium
```

Under this interface, SurvArena should automatically:

- infer feature types and build a train-only preprocessing pipeline
- validate right-censored survival targets
- choose a model portfolio based on data shape and preset budget
- tune and rank models consistently
- save artifacts, validation summaries, and reusable predictors

The predictor surface now also aims to be notebook-friendly by default:

- quiet training output unless `verbose=True`
- leaderboard columns with explicit validation metric names
- Kaplan-Meier comparison plotting after fitting
- optional experimental tabular foundation-model inclusion via `enable_foundation_models=True`
- a dedicated `foundation` preset for users who want the foundation-model portfolio first

## What It Covers (Milestone 1)

- **Datasets:** SUPPORT, METABRIC, AIDS, GBSG2, FLCHAIN, WHAS500, PBC (+ KKBox target track)
- **Methods:** CoxPH, CoxNet, Random Survival Forest (RSF), DeepSurv
- **Protocol:** repeated nested CV with shared seeds and comparable tuning budget
- **Metrics:** Harrell's C-index (primary default), Uno's C-index, IBS, time-dependent AUC

## Documentation

- **Protocol, fairness, artifacts, and reproducibility:** [`docs/protocol.md`](docs/protocol.md)
- **Datasets and metadata:** [`docs/datasets.md`](docs/datasets.md)
- **Environment setup and smoke checks:** [`docs/environment.md`](docs/environment.md)
- **Design blueprint:** [`blueprint.md`](blueprint.md)
- **AutoGluon-style comparison and current gaps:** [`docs/autogluon_comparison.md`](docs/autogluon_comparison.md)
- **Tabular foundation-model roadmap:** [`docs/tabular_foundation_models_todo.md`](docs/tabular_foundation_models_todo.md)

## Examples

- **Quickstart notebook:** [`examples/survival_predictor_quickstart.ipynb`](examples/survival_predictor_quickstart.ipynb)
- **Examples overview:** [`examples/README.md`](examples/README.md)

## Quick Start

```bash
./scripts/setup_env.sh
python -m src.run_benchmark --benchmark-config configs/benchmark/standard_v1.yaml --dataset support --method coxph --limit-seeds 1 --n-trials 2
```

## Outputs

SurvArena writes:

- persisted splits to `data/splits/`
- split manifests to `data/splits/<task_id>/manifest.json`
- compact per-run ledger (`<benchmark_id>_run_records.jsonl.gz` + index JSON) to `results/runs/`
- aggregate summaries/tables to `results/summaries/` and `results/tables/`

Only `results/summaries/` is intended for git tracking; run ledgers and table exports are local artifacts.

## Planned Additions

- **AutoML-style user data entrypoint:** add a `SurvivalPredictor` API and CLI so users can fit survival models on their own data without writing loaders or configs first
- **Preset-driven portfolio search:** support `fast`, `medium`, and `best` modes that control model coverage, tuning depth, and optional ensembling
- **Additional loss functions:** broader objective support for classic and neural survival models (protocol and metrics context in [`docs/protocol.md`](docs/protocol.md))
- **TorchSurv deep survival model capacity:** expand beyond baseline classical methods with deep model training/evaluation support (evaluation workflow in [`docs/protocol.md`](docs/protocol.md))
- **Expanded dataset tracks:** continue adding medium/large cohorts with loader and metadata standards (dataset standards in [`docs/datasets.md`](docs/datasets.md))
- **Flexible model portfolio:** broaden support from classical baselines to tree, boosting, deep, and future tabular foundation-model integrations
- **Leaderboard hardening:** stronger reporting fields and submission-readiness checks (artifact contract in [`docs/protocol.md`](docs/protocol.md))
- **Stronger validation checks:** additional reproducibility and environment diagnostics for CI/local runs (setup and smoke checks in [`docs/environment.md`](docs/environment.md))

## Add a New Method

1. Implement `BaseSurvivalMethod` in `src/methods/`.
2. Add method config in `configs/methods/`.
3. Register the method in `src/run_benchmark.py`.
4. Ensure `predict_risk` and `predict_survival` return standardized outputs.

## License

MIT (see `LICENSE`).
