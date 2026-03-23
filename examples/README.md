# Examples

This folder contains user-facing examples for the `SurvivalPredictor` workflow
and sample artifacts generated from that workflow.

## Notebook

- `survival_predictor_quickstart.ipynb`: a beginner-friendly walkthrough using
  the built-in `gbsg2` dataset to explain survival labels, train/test splitting,
  model fitting, leaderboard interpretation, prediction, plotting, and
  save/load behavior

## Shipped Sample Outputs

The `examples/results/` tree includes example predictor artifacts you can browse
without rerunning the notebook:

- `results/predictor/gbsg2_quickstart/`: quickstart run outputs including
  `leaderboard.csv`, `fit_summary.json`, `predictor.pkl`,
  `predictor_manifest.json`, and `kaplan_meier_comparison.png`
- `results/predictor/notebook_demo/`: notebook-generated summary artifacts
- `results/predictor/gbsg2_standard_v1/`: additional leaderboard and summary
  outputs for the built-in `gbsg2` track

## Intended Workflow

The notebook demonstrates the repo-local predictor flow:

1. load a survival dataset into a pandas `DataFrame`
2. create `SurvivalPredictor(label_time=..., label_event=...)`
3. call `fit(...)`
4. inspect `leaderboard()` and `fit_summary()`
5. generate `predict_risk(...)` and `predict_survival(...)`
6. save the fitted predictor and plot Kaplan-Meier comparisons

In this repository checkout, the Python import path is:

```python
from survarena import SurvivalPredictor
```
