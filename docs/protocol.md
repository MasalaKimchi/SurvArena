# SurvArena Protocol (Milestone 1)

This document defines the evaluation protocol, reproducibility rules, and
result artifact contract for SurvArena.

## Standard v1

- Task: right-censored tabular survival prediction.
- Split design: repeated nested cross-validation.
  - Outer loop: 5 folds x 3 repeats (configurable).
  - Each outer repeat uses one benchmark seed; repeats must not silently recycle the same seed.
  - Inner loop: 3 folds for hyperparameter tuning.
- Seed policy: shared benchmark seed list across all methods.
- Hyperparameter selection: highest mean inner-validation primary metric from benchmark config (default Harrell's C-index).
- Search budget: bounded by `n_trials` and optional wall-clock `timeout_seconds` from benchmark config.

## Metrics

- Primary: Harrell's C-index (default, configurable via benchmark config).
- Secondary: Uno's C-index, integrated Brier score, time-dependent AUC.
- Efficiency: fit time, inference time, peak process memory.
- Metric backend: `torchsurv`.

## Evaluation Flow

1. Load benchmark, dataset, and method configs.
2. Load persisted split JSON files plus split manifest if present; regenerate splits when the manifest no longer matches the active dataset fingerprint or split config.
3. For each dataset x method x outer split:
   - Fit preprocessing on outer-train only.
   - Run inner CV hyperparameter search.
   - Refit on full outer-train with best params.
   - Evaluate on outer-test and log metrics.
4. Write per-run records into a compact benchmark ledger
   (`<benchmark_id>_run_records.jsonl.gz`) with embedded manifest, metrics, and
   failure tracebacks.
5. Export fold tables and aggregate summaries.

## Fairness and Leakage Controls

- Identical split definitions for every method.
- Comparable tuning budgets across methods.
- Identical repeat/fold policy for reporting.
- No test leakage: preprocessing and tuning use training-side data only.
- Runtime seeds must be injected consistently into every stochastic method implementation.

## Artifact Contract

### Required Output Locations

- Splits: `data/splits/<task_id>/`.
- Split manifest: `data/splits/<task_id>/manifest.json`.
- Benchmark runner outputs: `results/summary/exp_<YYYYMMDD_HHMMSS>/`.
- Experiment directory contents: fold results CSV, seed summary CSV, overall summary JSON, leaderboard CSV/JSON, run-record ledger JSONL.GZ, run-record index JSON, and `experiment_manifest.json`.
- Standalone export helpers may also emit canonical files under `results/runs/`, `results/summaries/`, and `results/tables/`.

### Required Per-run Table Fields

These fields must appear in the canonical per-run export `results/tables/fold_results.csv`.

- `benchmark_id`
- `dataset_id`
- `method_id`
- `split_id`
- `seed`
- `validation_score`
- `uno_c`
- `harrell_c`
- `ibs`
- `tuning_time_sec`
- `runtime_sec`
- `fit_time_sec`
- `infer_time_sec`
- `peak_memory_mb`
- `status`

### Leaderboard Summary

- `results/tables/leaderboard.csv` and `results/summaries/leaderboard.json` are aggregate summaries over seed-level results.
- Summary leaderboards should not be treated as the canonical per-run artifact contract.

## Reproducibility Guarantees

- Shared global seed list from benchmark config.
- Persisted split definitions reused on reruns only when their split manifest matches the active split config and dataset fingerprint.
- Config-driven dataset and method selection.
- Per-run environment and git metadata recorded in manifest payloads.
- Deterministic config and split fingerprints (`benchmark_config_hash`,
  `method_config_hash`, `split_indices_hash`) recorded per run.
