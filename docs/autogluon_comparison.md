# SurvArena vs. AutoGluon-Style UX

This note clarifies what SurvArena can already do in an AutoGluon-like way,
what is still missing, and what needs to improve for a truly polished user
experience.

## What Already Works

The current `SurvivalPredictor` interface already supports the core first-step
experience:

- accept a user `DataFrame`, CSV, or Parquet file
- require only `time` and `event` labels
- automatically preprocess tabular features
- accept explicit tuning data or create an automatic validation holdout
- run a portfolio of models via `fast`, `medium`, and `best` presets
- compare models through a leaderboard
- expose `predict_risk(...)` and `predict_survival(...)`
- plot Kaplan-Meier comparisons after fitting
- persist basic artifacts such as a leaderboard CSV and fit summary JSON
- stay quiet by default instead of streaming raw tuning logs

For many users, this is already the right high-level shape: one object, one fit
call, one leaderboard.

## What Is Still Missing Compared to AutoGluon

AutoGluon is much more mature as an end-to-end AutoML system. SurvArena still
lacks several important pieces:

### 1. Richer training orchestration

- no bagging / stacking / weighted ensembling
- no multi-level model reuse or meta-learning
- no bagged out-of-fold training flow comparable to AutoGluon's `num_bag_folds`
- no adaptive per-model time scheduler with AutoGluon-style resource controls

### 2. Better model management

- no per-model artifact folders with reusable preprocessors and fitted weights
- no model-pruning or cleanup policy for retained fitted candidates
- no richer model inspection metadata such as fold histories or OOF artifacts

### 3. More complete user-facing reporting

- leaderboard is basic and centered on validation score
- no rich summary of train/test metrics across all candidate models
- no feature importance, permutation importance, or explainability views
- no calibration plots or survival-curve diagnostics exported automatically

### 4. Better data intelligence

- no advanced feature metadata abstraction
- no explicit handling for text, datetime, grouped features, or ID semantics
- no automatic target sanity analysis beyond binary event and numeric time checks
- no automated warnings for heavy censoring, low-event regimes, or leakage risks

### 5. Better scaling ergonomics

- only a lightweight sequential wallclock scheduler today, without resource-aware per-model shaping
- no resource-aware GPU/CPU policy exposed at the predictor level
- no caching of transformed folds or intermediate model artifacts
- no resumable training session management

### 6. Broader AutoML search behavior

- presets are static rather than dynamically portfolio-aware
- no search-space adaptation based on dataset shape, sparsity, or censoring rate
- no portfolio warm-starting from prior results
- no automatic ensemble construction over top candidates

### 7. Foundation-model integration is still early

- an experimental `tabpfn`-based survival adapter can now be included
- it currently uses foundation-model embeddings plus a Cox survival head
- it is not yet a full foundation-model search stack with multiple backbones,
  richer fine-tuning controls, or production-grade artifact management

## Bottom Line

SurvArena now supports **simple comparison in an AutoGluon-style shape**, but it
is not yet AutoGluon-level in maturity, orchestration, or model-management depth.

The current state is best described as:

> AutoGluon-like entrypoint and workflow, backed by a benchmark-oriented survival
> engine, with major AutoML features still to be built.
