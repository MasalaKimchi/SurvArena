# Environment Setup and Verification

## Recommended Python

- Python 3.11 (preferred)
- Python 3.10 (supported)

## One-command setup

```bash
./scripts/setup_env.sh
```

This script:

- creates `.venv`
- installs `requirements.txt`
- runs `scripts/check_environment.py`

## Manual setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

## What is validated

- importability of core dependencies (`numpy`, `pandas`, `yaml`, `torch`, `torchsurv`, `optuna`, `lifelines`, `sksurv`)
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
- Per-run ledger: `results/runs/<benchmark_id>_run_records.jsonl.gz`
  - each line contains `manifest`, `metrics`, and optional failure traceback
- Ledger index: `results/runs/<benchmark_id>_run_records_index.json`
- Aggregates:
  - `results/tables/fold_results.csv`
  - `results/summaries/seed_summary.csv`
  - `results/summaries/overall_summary.json`
  - `results/tables/leaderboard.csv`
  - `results/summaries/leaderboard.json`

Git tracking policy: only `results/summaries/` should be committed.
