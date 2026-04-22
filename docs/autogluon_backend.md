# AutoGluon-Backed Training

SurvArena includes `autogluon_survival`, a literal AutoGluon Tabular-backed
adapter for right-censored survival benchmarks.

AutoGluon does not natively fit censored survival objectives. The v1 adapter
therefore trains a binary event-risk model with `TabularPredictor` and keeps
SurvArena's external survival evaluation unchanged. Risk predictions come from
AutoGluon's event probability. Survival curves are calibrated with a Breslow
baseline estimated from the training data and the fitted risk scores.

Use `configs/benchmark/models/*_default.yaml` to run one native model at a
time. Use `standard_v1_autogluon_survival_default.yaml` for a no-HPO AutoGluon
baseline and `standard_v1_autogluon_survival_autogluon.yaml` for an
AutoGluon-managed preset/HPO/bagging/stacking run.

The benchmark ledger records backend-neutral fields including
`training_backend`, `hpo_backend`, `autogluon_presets`, `autogluon_best_model`,
`autogluon_model_count`, `autogluon_path`, `bagging_folds`, and `stack_levels`.
