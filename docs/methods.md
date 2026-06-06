# Methods

Built-in method configs live in `configs/methods/`. Use the `method_id` values
below with `included_models`, `--models`, or benchmark YAML `methods`.

Last reviewed against `configs/methods/` and the method registry: 2026-06-05.

## How to Select Methods

Python:

```python
from survarena import SurvivalPredictor

predictor = SurvivalPredictor(
    label_time="time",
    label_event="event",
    included_models=["coxph", "rsf"],
)
```

CLI:

```bash
survarena pilot \
  --data train.csv \
  --time-col time \
  --event-col event \
  --models coxph,rsf
```

Benchmark YAML:

```yaml
methods:
  - coxph
  - rsf
```

Prefer presets for normal predictor use and explicit method IDs for controlled
experiments. Foundation adapters are optional; run `survarena foundation-check`
before including them in long runs.

## Available Model Adapters

| Method ID | Model / adapter | Family | Package source | Benchmark use |
| --- | --- | --- | --- | --- |
| `coxph` | Cox proportional hazards | Classical | `scikit-survival` | Manuscript |
| `coxnet` | Regularized Cox model | Classical | `scikit-survival` | Manuscript |
| `weibull_aft` | Weibull accelerated failure time | Classical | `lifelines` | Manuscript |
| `lognormal_aft` | Log-normal accelerated failure time | Classical | `lifelines` | Manuscript |
| `loglogistic_aft` | Log-logistic accelerated failure time | Classical | `lifelines` | Manuscript |
| `aalen_additive` | Aalen additive hazards | Classical | `lifelines` | Manuscript |
| `fast_survival_svm` | Fast survival SVM | Classical | `scikit-survival` | Manuscript |
| `rsf` | Random survival forest | Tree ensemble | `scikit-survival` | Manuscript |
| `extra_survival_trees` | Extra survival trees | Tree ensemble | `scikit-survival` | Manuscript |
| `gradient_boosting_survival` | Gradient boosting survival analysis | Boosting | `scikit-survival` | Manuscript |
| `componentwise_gradient_boosting` | Componentwise gradient boosting survival analysis | Boosting | `scikit-survival` | Manuscript |
| `xgboost_cox` | XGBoost Cox objective adapter | Boosting | `xgboost` | Manuscript |
| `xgboost_aft` | XGBoost AFT objective adapter | Boosting | `xgboost` | Manuscript |
| `catboost_cox` | CatBoost Cox-style calibrated adapter | Boosting | `catboost` | Manuscript |
| `catboost_survival_aft` | CatBoost survival AFT adapter | Boosting | `catboost` | Manuscript |
| `deepsurv` | DeepSurv neural Cox model | Deep learning | `torchsurv` | Manuscript |
| `deepsurv_moco` | DeepSurv momentum-loss variant | Deep learning | `torchsurv` | Manuscript |
| `logistic_hazard` | Logistic hazard neural survival model | Deep learning | `pycox` | Manuscript |
| `pmf` | PMF neural discrete-time survival model | Deep learning | `pycox` | Manuscript |
| `mtlr` | MTLR neural survival model | Deep learning | `pycox` | Manuscript |
| `deephit_single` | DeepHit single-risk model | Deep learning | `pycox` | Manuscript |
| `pchazard` | Piecewise constant hazard neural model | Deep learning | `pycox` | Manuscript |
| `cox_time` | Cox-Time neural survival model | Deep learning | `pycox` | Manuscript |
| `tabpfn_survival` | TabPFN horizon survival adapter | Foundation | `tabpfn` | Manuscript |
| `tabpfn_discrete_hazard_survival` | TabPFN pooled discrete-time hazard adapter | Foundation | `tabpfn` | Manuscript comparison |
| `tabicl_survival` | TabICL horizon survival adapter | Foundation | `tabicl` | Manuscript |
| `tabicl_discrete_hazard_survival` | TabICL pooled discrete-time hazard adapter | Foundation | `tabicl` | Manuscript comparison |
| `tabm_survival` | TabM horizon survival adapter | Foundation | `autogluon.tabular` TABM | Manuscript |
| `tabm_discrete_hazard_survival` | TabM pooled discrete-time hazard adapter | Foundation | `autogluon.tabular` TABM | Manuscript comparison |
| `realtabpfn_survival` | RealTabPFN-V2 horizon survival adapter | Foundation | `autogluon.tabular` REALTABPFN-V2 | Manuscript |
| `realtabpfn_discrete_hazard_survival` | RealTabPFN-V2 pooled discrete-time hazard adapter | Foundation | `autogluon.tabular` REALTABPFN-V2 | Manuscript comparison |
| `mitra_survival_frozen` | Frozen Mitra event-risk adapter | Foundation | `autogluon.tabular` MITRA | Available, excluded from manuscript no-HPO |

## Adding Methods

Method adapter contribution requirements are documented in
[`contributing_method_adapters.md`](contributing_method_adapters.md).
