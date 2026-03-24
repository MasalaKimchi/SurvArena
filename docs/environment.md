# Environment Setup and Verification

## Recommended Python

- Python 3.11 (preferred)
- Python 3.12 (supported)
- Python 3.10 (supported)

## One-command setup

```bash
./scripts/setup_env.sh
```

This script:

- creates `.venv`
- installs SurvArena in editable mode with the `dev` extra
- runs `scripts/check_environment.py`
- keeps SurvArena isolated from the global Python environment
- defaults to `python` rather than `python3` so it avoids accidentally picking an unsupported interpreter such as Python 3.13

To include the optional foundation-model adapters during setup:

```bash
INSTALL_EXTRAS=dev,foundation ./scripts/setup_env.sh
```

To install only one optional foundation adapter:

```bash
INSTALL_EXTRAS=dev,foundation-tabpfn ./scripts/setup_env.sh
INSTALL_EXTRAS=dev,foundation-mitra ./scripts/setup_env.sh
```

## Manual setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
```

To enable the optional foundation-model adapters:

```bash
python -m pip install -e ".[foundation]"
python scripts/check_environment.py --include-foundation
```

Or install just one:

```bash
python -m pip install -e ".[foundation-tabpfn]"
python -m pip install -e ".[foundation-mitra]"
survarena foundation-check
```

SurvArena should be run from the repo-local `.venv` whenever possible. The optional
foundation-model stack can force transitive version changes that are better kept
out of a shared global interpreter.

## What is validated

- whether Python is running inside a virtual environment
- importability of core dependencies (`numpy`, `pandas`, `yaml`, `torch`, `torchsurv`, `optuna`, `lifelines`, `sksurv`, `xgboost`, `catboost`)
- optional importability of foundation-model dependencies (`tabpfn`, `autogluon.tabular`)
- per-backbone runtime readiness, including install hints and TabPFN auth warnings
- `torchsurv` metric classes and IPCW API
- synthetic metric run for:
  - Uno C-index
  - Harrell C-index
  - Integrated Brier Score
  - time-dependent AUC

## Smoke Checks

```bash
python -m compileall survarena
python -m survarena.run_benchmark --dry-run
```

If dependencies are missing, `--dry-run` reports the issue and exits cleanly.

## Expected Artifacts After Runs

- Split files: `data/splits/<task_id>/*.json`
- Split manifest: `data/splits/<task_id>/manifest.json`
- Benchmark runs write timestamped experiment directories under `results/summary/exp_<YYYYMMDD_HHMMSS>/`
- Each experiment directory contains:
  - `<benchmark_id>_fold_results.csv`
  - `<benchmark_id>_seed_summary.csv`
  - `<benchmark_id>_overall_summary.json`
  - `<benchmark_id>_leaderboard.csv`
  - `<benchmark_id>_leaderboard.json`
  - `<benchmark_id>_run_records.jsonl.gz`
  - `<benchmark_id>_run_records_index.json`
  - `experiment_manifest.json`
- Standalone export helpers can also write canonical files under `results/tables/`, `results/summaries/`, and `results/runs/`.

Git tracking policy: timestamped experiment directories under `results/summary/` are local-only. Commit only curated summary artifacts when intentionally publishing benchmark results.
