# Examples

This folder contains user-facing examples for the AutoML-style
`SurvivalPredictor` interface.

## Files

- `survival_predictor_quickstart.ipynb`: a detailed beginner-friendly
  walkthrough using the built-in `gbsg2` dataset to explain survival labels,
  train/test splitting, model fitting, leaderboard interpretation, prediction,
  plotting, and save/load behavior

## Intended Workflow

The notebook demonstrates the expected AutoML-style usage pattern:

1. load a survival dataset into a pandas `DataFrame`
2. specify the survival label columns
3. call `fit(...)`
4. inspect the leaderboard
5. generate risk and survival predictions
6. plot Kaplan-Meier comparisons

This is the SurvArena entrypoint that should feel closest to AutoGluon, while
still explaining the survival-analysis concepts that new users usually need on
their first pass.
