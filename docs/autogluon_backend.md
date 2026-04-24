# AutoGluon-Backed Training

SurvArena includes `autogluon_survival`, a literal AutoGluon Tabular-backed
adapter for right-censored survival benchmarks.

AutoGluon does not natively fit censored survival objectives. The v1 adapter
therefore trains a binary event-risk model with `TabularPredictor` and keeps
SurvArena's external survival evaluation unchanged. Risk predictions come from
AutoGluon's event probability. Survival curves are calibrated with a Breslow
baseline estimated from the training data and the fitted risk scores.

To run a **single** native model or `autogluon_survival` on the standard
six-dataset matrix, use `configs/benchmark/standard_v1.yaml` (or
`manuscript_v1.yaml`) with `--method` and `--dataset` as needed, or use
`configs/benchmark/manuscript_autogluon_v1.yaml` for an AutoGluon-managed
preset, bagging, stacking, and refit run on all standard datasets.

The benchmark ledger records backend-neutral fields including
`training_backend`, `hpo_backend`, `autogluon_presets`, `autogluon_best_model`,
`autogluon_model_count`, `autogluon_path`, `bagging_folds`, and `stack_levels`.
