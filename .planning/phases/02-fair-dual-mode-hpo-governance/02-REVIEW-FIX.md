---
phase: 02-fair-dual-mode-hpo-governance
fix_scope: critical_warning
findings_in_scope: 2
fixed: 2
skipped: 0
iteration: 1
status: all_fixed
source_review: 02-REVIEW.md
---

# Phase 02: Code Review Fix Report

**Scope:** `critical_warning` (WR-01, WR-02; Info findings not in scope)  
**Iteration:** 1 (single pass; `--auto` not used)

## Summary

| ID | Status | Notes |
| --- | --- | --- |
| WR-01 | Fixed | `survarena/benchmark/tuning.py`: import `metric_direction` from `survarena.evaluation.statistics` and use it in `_metric_direction_for_optimization` so Optuna `create_study(direction=...)` matches ranking semantics for loss metrics (e.g. IBS, Brier). |
| WR-02 | Fixed | `survarena/logging/export.py`: `_group_keys_with_hpo_mode` inserts `hpo_mode` after `method_id` in seed summaries, leaderboards, manuscript `comparative_leaderboard` groupbys, and `multiple_comparison_summary` when `hpo_mode` is present. `survarena/evaluation/statistics.py`: stratum `hpo_mode` in `add_dataset_ranks`, `aggregate_rank_summary`, `pairwise_win_rate`, `elo_ratings`, `bootstrap_metric_ci`, `critical_difference_summary`, and `group_cols` in `pairwise_significance`. |

## Commit

- `fix(02): HPO optuna direction and hpo_mode-stratified summaries` (code under `survarena/`)

## Verification

- `pytest` full suite: **127 passed**, 6 skipped  
- `ruff check` on touched files: **pass**

## Follow-up (optional, out of scope for this run)

- Info-level items (IN-01–IN-04) in `02-REVIEW.md` remain; run `/gsd-code-review-fix 2 --all` if you want them tracked in a second fix pass.
