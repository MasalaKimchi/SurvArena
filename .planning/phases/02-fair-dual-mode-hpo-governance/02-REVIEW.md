---
phase: 02-fair-dual-mode-hpo-governance
reviewed: 2026-04-23T12:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - survarena/api/compare.py
  - survarena/benchmark/runner.py
  - survarena/benchmark/tuning.py
  - survarena/logging/export.py
  - tests/test_compare_api.py
  - tests/test_dual_mode_hpo_governance.py
  - tests/test_hpo_config.py
findings:
  critical: 0
  warning: 2
  info: 4
  total: 6
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-04-23T12:00:00Z  
**Depth:** standard  
**Files Reviewed:** 7  
**Status:** issues_found

## Summary

Dual-mode execution (`no_hpo` then `hpo`), parity eligibility after successful pairs, and HPO budget fields on `run_payload` / fold rows are mostly consistent between `compare_survival_models` and `run_benchmark`. The main correctness risk is **downstream summarization**: `export_seed_summary`, `export_leaderboard`, and `export_manuscript_comparison` aggregate fold rows **without stratifying on `hpo_mode`**, so when both modes are parity-eligible the numeric metrics are averaged across modes. A second issue is **Optuna study direction** in `select_hyperparameters`, which always aligns with maximization while `metric_direction()` in statistics defines several **minimize** primary metrics (e.g. `ibs`, Brier). Tests exercise parity flags on fold results and budget columns but do not assert that leaderboards or manuscript tables preserve separate `hpo_mode` strata.

## Warnings

### WR-01: HPO study direction ignores minimize metrics

**File:** `survarena/benchmark/tuning.py:57-60`  
**Issue:** `_metric_direction_for_optimization` always returns `"maximize"`, so `select_hyperparameters` creates an Optuna study with `direction="maximize"` even when `primary_metric` is a loss (e.g. `ibs`, `brier_*`), for which `metric_direction()` returns `"minimize"` (`survarena/evaluation/statistics.py:19-28`, `:55-60`). Hyperparameter selection then optimizes the wrong direction for those metrics.  
**Fix:** Derive direction from the same source of truth as ranking (e.g. reuse `metric_direction(primary_metric)` and map to Optuna `"maximize"` / `"minimize"`), and adjust the NaN objective fallback in `_objective` to return the correct sentinel for each direction.

```python
from survarena.evaluation.statistics import metric_direction

def _metric_direction_for_optimization(primary_metric: str) -> str:
    return metric_direction(primary_metric)
```

Then in `_objective`, use `maximize = metric_direction(primary_metric) == "maximize"` (already computed) and keep `return float("-inf") if maximize else float("inf")` for NaN scores consistently.

### WR-02: Summaries and manuscript comparisons blend `hpo_mode` (dual-mode parity undermined)

**File:** `survarena/logging/export.py:125-143, 175-187, 224-236` (and call sites in `survarena/api/compare.py:341-361`, `survarena/benchmark/runner.py:786-803`)  
**Issue:** `export_seed_summary` groups only by `["benchmark_id", "dataset_id", "method_id", "seed"]` (`export.py:132-135`). `export_leaderboard` groups only by `["benchmark_id", "dataset_id", "method_id"]` (`export.py:185-187`). After parity gating, both `no_hpo` and `hpo` rows can be present for the same keys; `.mean(numeric_only=True)` **merges the two modes** into a single pseudo-metric. The manuscript path builds `comparative_leaderboard` with the same groupby keys (`export.py:232-234`), so ranks, pairwise win rates, CD diagrams, and bootstrap CIs can reflect **averages of HPO and non-HPO** rather than a chosen mode or side-by-side reporting.  

Additionally, `pairwise_significance` in `survarena/evaluation/statistics.py:274-276` builds `group_cols` from `["benchmark_id", "dataset_id", "split_id", "seed"]` but **does not include `hpo_mode`**. When the fold-level frame contains two eligible modes for the same split/seed/method, the merge path can pair rows incorrectly (many-to-many on duplicate keys).  

**Fix:** When `"hpo_mode"` exists in the frame, include it in `by_cols` for seed summaries, in leaderboard groupby keys, in `comparative_leaderboard` groupby, and in `pairwise_significance`’s `group_cols` (or pre-filter fold results to a single mode with explicit API). Document the intended consumer: separate leaderboards per mode vs. “primary” mode only.

## Info

### IN-01: Duplicated HPO telemetry normalization

**File:** `survarena/api/compare.py:262-278` vs `survarena/benchmark/runner.py:131-145`  
**Issue:** The same budget-field normalization logic exists in both modules, which raises drift risk if one path is updated without the other.  
**Fix:** Call a single shared helper (e.g. move `_normalize_hpo_budget_telemetry` to `tuning` or a small `survarena.benchmark.hpo_telemetry` module) from both `compare` and `runner`.

### IN-02: Library API uses `print` for progress

**File:** `survarena/api/compare.py:293-296`  
**Issue:** `compare_survival_models` prints to stdout for every evaluated split/mode, which is noisy for programmatic callers and bypasses structured logging.  
**Fix:** Prefer logging at DEBUG/INFO behind a verbosity flag, or accept an optional progress callback.

### IN-03: Tests do not guard aggregation / manuscript behavior for dual mode

**File:** `tests/test_compare_api.py`, `tests/test_dual_mode_hpo_governance.py`  
**Issue:** Coverage is strong for fold-level `parity_eligible`, budget columns, run-record metadata, and execution order. There are no assertions that `seed_summary` / `leaderboard` / manuscript outputs **separate or label** `hpo_mode`, so **WR-02** would not be caught by CI.  
**Fix:** Extend tests to load exported CSVs (or call export helpers with synthetic frames containing two modes) and assert distinct rows or explicit `hpo_mode` column in aggregated outputs.

### IN-04: `evaluate_split` failure path assumes NumPy import succeeded

**File:** `survarena/benchmark/runner.py:167-418`  
**Issue:** `np` is imported inside the `try` block. If `import numpy as np` fails, the `except` branch still references `np.nan`, which can raise `UnboundLocalError` and mask the original failure. Low likelihood in this project if NumPy is a hard dependency.  
**Fix:** Import NumPy at module level, or in `except` use `float("nan")` without `np`, or catch and re-raise without referencing `np`.

---

_Reviewed: 2026-04-23T12:00:00Z_  
_Reviewer: Claude (gsd-code-reviewer)_  
_Depth: standard_
