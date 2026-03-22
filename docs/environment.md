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
- installs `requirements.txt`
- runs `scripts/check_environment.py`
- keeps SurvArena isolated from the global Python environment
- defaults to `python` rather than `python3` so it avoids accidentally picking an unsupported interpreter such as Python 3.13

## Manual setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

SurvArena should be run from the repo-local `.venv` whenever possible. The
foundation-model stack now includes `tabpfn` and `autogluon.tabular`, and those
packages can force transitive version changes that are better kept out of a
shared global interpreter.

## What is validated

- whether Python is running inside a virtual environment
- importability of core dependencies (`numpy`, `pandas`, `yaml`, `torch`, `torchsurv`, `optuna`, `lifelines`, `sksurv`)
- importability of foundation-model dependencies (`tabpfn`, `autogluon.tabular`)
- `torchsurv` metric classes and IPCW API
- synthetic metric run for:
  - Uno C-index
  - Harrell C-index
  - Integrated Brier Score
  - time-dependent AUC

## Smoke Checks

```bash
python -m compileall src
python -m src.run_benchmark --dry-run
```

If dependencies are missing, `--dry-run` reports the issue and exits cleanly.

## Expected Artifacts After Runs

- Split files: `data/splits/<task_id>/*.json`
- Split manifest: `data/splits/<task_id>/manifest.json`
- Per-run ledger: `results/runs/<benchmark_id>_run_records.jsonl.gz`
  - each line contains `manifest`, `metrics`, and optional failure traceback
- Ledger index: `results/runs/<benchmark_id>_run_records_index.json`
- Aggregates:
  - `results/tables/fold_results.csv` (canonical per-run table)
  - `results/summaries/seed_summary.csv`
  - `results/summaries/overall_summary.json`
  - `results/tables/leaderboard.csv` (aggregate summary)
  - `results/summaries/leaderboard.json` (aggregate summary)

Git tracking policy: only `results/summaries/` should be committed.
