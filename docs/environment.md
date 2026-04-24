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
- `INSTALL_EXTRAS=dev,foundation-mitra ./scripts/setup_env.sh`

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

If an optional dependency is missing, `--dry-run` reports it and exits cleanly.

End-to-end protocol spot-check (dry run plus a tiny fit and artifact checks;
see `docs/protocol.md`):

```bash
./scripts/validate_benchmark_protocol.sh
```

Optional environment overrides: `BENCHMARK_CONFIG`, `WORK_DIR`, `PYTHON_BIN`.

## Output Locations

- splits: `data/splits/<task_id>/`
- predictor artifacts: `results/predictor/<dataset_name>/`
- benchmark runs: `results/summary/exp_<YYYYMMDD_HHMMSS>/`

Benchmark experiment folders contain:

- `<benchmark_id>_fold_results.csv`
- `<benchmark_id>_seed_summary.csv`
- `<benchmark_id>_overall_summary.json`
- `<benchmark_id>_leaderboard.csv`
- `<benchmark_id>_leaderboard.json`
- `<benchmark_id>_run_records.jsonl.gz`
- `<benchmark_id>_run_records_index.json`
- `experiment_manifest.json`

Treat timestamped benchmark output folders as local run artifacts unless you are
intentionally publishing curated results.
