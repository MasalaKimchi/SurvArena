# Environment

## Supported Python

- 3.11 preferred
- 3.10 and 3.12 supported

## Quick Setup

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

The setup script:

- creates `.venv`
- installs SurvArena in editable mode
- runs `scripts/check_environment.py`

Useful overrides:

- `PYTHON_BIN=python3.11 ./scripts/setup_env.sh`
- `INSTALL_EXTRAS=dev,foundation ./scripts/setup_env.sh`
- `INSTALL_EXTRAS=dev,foundation-tabpfn ./scripts/setup_env.sh`

## Manual Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

Optional foundation extras:

```bash
python -m pip install -e ".[foundation]"
python scripts/check_environment.py --include-foundation
survarena foundation-check
```

## What the Check Covers

- virtual environment detection
- core imports such as `numpy`, `pandas`, `torch`, `torchsurv`, `autogluon.tabular`, `lifelines`, `sksurv`, `xgboost`, and `catboost`
- optional foundation imports such as `tabpfn` and `autogluon.tabular`
- runtime readiness messages for wired foundation adapters
- smoke tests for Harrell C-index, Uno C-index, integrated Brier score, and time-dependent AUC

## Smoke Checks

```bash
python -m compileall survarena
python -m survarena.run_benchmark --dry-run
```

Use `python scripts/check_environment.py --include-foundation` and
`survarena foundation-check` for optional foundation dependency readiness.

End-to-end protocol spot-check (dry run plus a tiny fit and artifact checks;
see `docs/protocol.md`):

```bash
./scripts/validate_benchmark_protocol.sh
```

Optional environment overrides: `BENCHMARK_CONFIG`, `WORK_DIR`, `PYTHON_BIN`.

## Output Locations

- splits: `data/splits/<task_id>/`
- predictor artifacts: `results/predictor/<dataset_name>/`
- benchmark runs: `results/summary/<dataset_id>/<benchmark_id>/<model_name>/` (or `<model_name>_<timestamp>` on collision)

Benchmark experiment folders contain core CSV outputs plus the experiment
manifest.

Treat timestamped benchmark output folders as local run artifacts unless you are
intentionally publishing curated results.
