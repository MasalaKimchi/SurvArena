# SurvArena

SurvArena is a concise, reproducible benchmark for tabular right-censored survival models.  
It standardizes splits, tuning budgets, evaluation metrics, and artifacts so methods are compared as full pipelines.

## What It Covers (Milestone 1)

- **Datasets:** SUPPORT, METABRIC, GBSG2, FLCHAIN, WHAS500, PBC (+ KKBox target track)
- **Methods:** CoxPH, CoxNet, Random Survival Forest (RSF)
- **Protocol:** repeated nested CV with shared seeds and comparable tuning budget
- **Metrics:** Harrell's C-index (primary default), Uno's C-index, IBS, time-dependent AUC

## Documentation

- **Protocol, fairness, artifacts, and reproducibility:** [`docs/protocol.md`](docs/protocol.md)
- **Datasets and metadata:** [`docs/datasets.md`](docs/datasets.md)
- **Environment setup and smoke checks:** [`docs/environment.md`](docs/environment.md)
- **Design blueprint:** [`blueprint.md`](blueprint.md)

## Quick Start

```bash
./scripts/setup_env.sh
python -m src.run_benchmark --benchmark-config configs/benchmark/standard_v1.yaml --dataset support --method coxph --limit-seeds 1 --n-trials 2
```

## Outputs

SurvArena writes:

- persisted splits to `data/splits/`
- compact per-run ledger (`<benchmark_id>_run_records.jsonl.gz` + index JSON) to `results/runs/`
- aggregate summaries/tables to `results/summaries/` and `results/tables/`

Only `results/summaries/` is intended for git tracking; run ledgers and table exports are local artifacts.

## Planned Additions

- **Additional loss functions:** broader objective support for classic and neural survival models (protocol and metrics context in [`docs/protocol.md`](docs/protocol.md))
- **TorchSurv deep survival model capacity:** expand beyond baseline classical methods with deep model training/evaluation support (evaluation workflow in [`docs/protocol.md`](docs/protocol.md))
- **Expanded dataset tracks:** continue adding medium/large cohorts with loader and metadata standards (dataset standards in [`docs/datasets.md`](docs/datasets.md))
- **Leaderboard hardening:** stronger reporting fields and submission-readiness checks (artifact contract in [`docs/protocol.md`](docs/protocol.md))
- **Stronger validation checks:** additional reproducibility and environment diagnostics for CI/local runs (setup and smoke checks in [`docs/environment.md`](docs/environment.md))

## Add a New Method

1. Implement `BaseSurvivalMethod` in `src/methods/`.
2. Add method config in `configs/methods/`.
3. Register the method in `src/run_benchmark.py`.
4. Ensure `predict_risk` and `predict_survival` return standardized outputs.

## License

MIT (see `LICENSE`).
