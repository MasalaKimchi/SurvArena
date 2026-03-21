# SurvArena Protocol (Milestone 1)

This document defines the evaluation protocol, reproducibility rules, and
result artifact contract for SurvArena.

## Standard v1

- Task: right-censored tabular survival prediction.
- Split design: repeated nested cross-validation.
  - Outer loop: 5 folds x 3 repeats (configurable).
  - Inner loop: 3 folds for hyperparameter tuning.
- Seed policy: shared benchmark seed list across all methods.
- Hyperparameter selection: highest mean inner-validation primary metric from benchmark config (default Harrell's C-index).

## Metrics

- Primary: Harrell's C-index (default, configurable via benchmark config).
- Secondary: Uno's C-index, integrated Brier score, time-dependent AUC.
- Efficiency: fit time, inference time, peak process memory.
- Metric backend: `torchsurv`.

## Evaluation Flow

1. Load benchmark, dataset, and method configs.
2. Load persisted split JSON files if present; otherwise generate and persist.
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

## Artifact Contract

### Required Output Locations

- Splits: `data/splits/<task_id>/`.
- Runs: `results/runs/<benchmark_id>_run_records.jsonl.gz` (+ index JSON).
- Aggregates: `results/summaries/` and `results/tables/`.

### Required Leaderboard Fields

- `benchmark_id`
- `dataset_id`
- `method_id`
- `split_id`
- `seed`
- `uno_c`
- `harrell_c`
- `ibs`
- `fit_time_sec`
- `infer_time_sec`
- `peak_memory_mb`
- `status`

## Reproducibility Guarantees

- Shared global seed list from benchmark config.
- Persisted split definitions reused on reruns.
- Config-driven dataset and method selection.
- Per-run environment and git metadata recorded in manifest payloads.
- Deterministic config and split fingerprints (`benchmark_config_hash`,
  `method_config_hash`, `split_indices_hash`) recorded per run.
