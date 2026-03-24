# SurvArena Blueprint

## Goal

Make survival modeling easy to start and easy to benchmark.
SurvArena should feel simple at the entrypoint while keeping the evaluation
rules strict enough for reproducible comparisons.

## Product Layers

- predictor mode: fit a portfolio on a user dataset with `SurvivalPredictor`
- compare mode: run benchmark-style comparisons on a user dataset with `compare_survival_models`
- benchmark mode: run tracked configs on built-in datasets with persisted splits and exported summaries

## Current Scope

- right-censored tabular survival only
- built-in benchmark datasets plus direct user dataset support
- classical, tree, boosting, deep, and experimental foundation adapters
- preset-driven model selection
- disk-first artifacts for predictors and benchmark runs

## Principles

- simple defaults for users bringing a CSV, Parquet file, or `DataFrame`
- fair comparisons based on shared splits and shared budgets
- no test leakage in preprocessing or tuning
- artifacts that are easy to inspect, diff, and reuse

## Near-Term Priorities

- ensembles and better search orchestration
- richer reporting and explainability
- stronger per-model artifact management
- broader foundation-model coverage
- real large-track datasets such as KKBox
