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
- benchmark runs: `results/summary/<benchmark_id>_<model_name>_<YYYYMMDD_HHMMSS>/`

Benchmark experiment folders contain generated `README.md` and
`experiment_navigator.json` entry points. Exact artifacts vary by benchmark
config, enabled comparison modes, HPO settings, and manuscript artifact layout.

The compact run ledger is the default comprehensive per-run artifact. Set
`exports.write_full_run_ledger: true` only when a downstream workflow needs the
legacy full run ledger.

Treat timestamped benchmark output folders as local run artifacts unless you are
intentionally publishing curated results.
