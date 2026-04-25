# SurvArena and AutoGluon-Style UX

## What Already Feels Similar

- one predictor object, one `fit(...)` call, one leaderboard
- preset portfolios plus explicit model selection
- automatic holdout or bagged OOF selection
- feature typing and dataset diagnostics
- prediction APIs, save/load, and Kaplan-Meier comparison plots
- quiet-by-default training flow for notebooks
- a benchmark-style compare API for user datasets

## What Is Still Missing

- native weighted ensembles
- adaptive resource scheduling
- richer explainability and calibration visualization
- stronger per-model artifact management
- broader foundation-model search and tuning controls

## Bottom Line

SurvArena now has an AutoML-style surface for survival data. It is still
benchmark-first under the hood and not yet as full-featured as AutoGluon.
