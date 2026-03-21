# Examples

This folder contains simple, user-facing examples for the new AutoML-style
`SurvivalPredictor` interface.

## Files

- `survival_predictor_quickstart.ipynb`: notebook showing how to fit,
  compare, and predict with a user-owned tabular survival dataset

## Intended Workflow

The notebook demonstrates the simplest expected usage pattern:

1. load or create a pandas `DataFrame`
2. specify the survival label columns
3. call `fit(...)`
4. inspect the leaderboard
5. generate risk and survival predictions
6. plot Kaplan-Meier comparisons

This is the SurvArena entrypoint that should feel closest to AutoGluon.
