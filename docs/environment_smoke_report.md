# Environment Smoke Report

Generated on 2026-04-22 from `/Users/justin/Documents/SurvArena`.

## Interpreter

- Python: 3.12.2
- Platform observed in run manifest: macOS arm64
- Active executable used for smoke runs: `python`
- Note: `./.venv312/bin/python` was not present in this workspace, so the available conda Python was used.

## Key Package Versions

- `autogluon.tabular==1.5.0`
- `catboost==1.2.10`
- `xgboost==2.1.4`
- `scikit-survival==0.24.1`
- `lifelines==0.30.0`
- `torch==2.2.2`
- `pycox==0.3.0`
- `torchsurv==0.1.5`
- `tabpfn==6.4.1`
- `numpy==1.26.4`
- `pandas==2.2.2`
- `scikit-learn==1.6.1`

`optuna==4.8.0` is still installed in the active environment, but it is not used by the current SurvArena benchmark path.

## Smoke Benchmark

- Config: `configs/benchmark/smoke_all_models_no_hpo.yaml`
- Dataset: `whas500`
- Protocol: 1 seed, 2 outer folds, no AutoGluon HPO
- Result directory: `results/summary/exp_20260422_232902`
- Run count: 46 fold runs
- Failure rate: 0.0 for all 23 methods
- Missing metric rate: 0.0 for all reported metrics

## Warnings Observed

- Matplotlib/fontconfig could not write to the user cache directories and used temporary cache directories.
- `sksurv` CoxPH emitted overflow warnings on the tiny smoke split.
- Deep Cox models emitted tied event-time warnings and used Efron's tie handling.
- PyCox discretized models warned that some requested cuts were not unique on the tiny smoke split.

These warnings did not prevent successful model fitting or metric export.
