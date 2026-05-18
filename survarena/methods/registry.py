from __future__ import annotations

from importlib import import_module
from functools import lru_cache
from typing import Any


_REGISTRY_TARGETS = {
    "coxph": ("survarena.methods.classical.coxph", "CoxPHMethod"),
    "coxnet": ("survarena.methods.classical.coxnet", "CoxNetMethod"),
    "weibull_aft": ("survarena.methods.classical.lifelines_models", "WeibullAFTMethod"),
    "lognormal_aft": ("survarena.methods.classical.lifelines_models", "LogNormalAFTMethod"),
    "loglogistic_aft": ("survarena.methods.classical.lifelines_models", "LogLogisticAFTMethod"),
    "aalen_additive": ("survarena.methods.classical.lifelines_models", "AalenAdditiveMethod"),
    "fast_survival_svm": ("survarena.methods.classical.fast_svm", "FastSurvivalSVMMethod"),
    "rsf": ("survarena.methods.tree.rsf", "RSFMethod"),
    "gradient_boosting_survival": (
        "survarena.methods.boosting.gradient_boosting",
        "GradientBoostingSurvivalMethod",
    ),
    "componentwise_gradient_boosting": (
        "survarena.methods.boosting.gradient_boosting",
        "ComponentwiseGradientBoostingMethod",
    ),
    "extra_survival_trees": ("survarena.methods.tree.extra_survival_trees", "ExtraSurvivalTreesMethod"),
    "xgboost_cox": ("survarena.methods.boosting.tabular_boosting", "XGBoostCoxMethod"),
    "xgboost_aft": ("survarena.methods.boosting.tabular_boosting", "XGBoostAFTMethod"),
    "xgbse_kaplan_neighbors": ("survarena.methods.boosting.xgbse_models", "XGBSEKaplanNeighborsMethod"),
    "catboost_cox": ("survarena.methods.boosting.tabular_boosting", "CatBoostCoxMethod"),
    "catboost_survival_aft": ("survarena.methods.boosting.tabular_boosting", "CatBoostSurvivalAFTMethod"),
    "deepsurv": ("survarena.methods.deep.deepsurv", "DeepSurvMethod"),
    "deepsurv_moco": ("survarena.methods.deep.deepsurv_moco", "DeepSurvMomentumMethod"),
    "logistic_hazard": ("survarena.methods.deep.pycox_models", "LogisticHazardMethod"),
    "pmf": ("survarena.methods.deep.pycox_models", "PMFMethod"),
    "mtlr": ("survarena.methods.deep.pycox_models", "MTLRMethod"),
    "deephit_single": ("survarena.methods.deep.pycox_models", "DeepHitSingleMethod"),
    "pchazard": ("survarena.methods.deep.pycox_models", "PCHazardMethod"),
    "cox_time": ("survarena.methods.deep.pycox_models", "CoxTimeMethod"),
    "tabpfn_survival": ("survarena.methods.foundation.tabpfn_survival", "TabPFNSurvivalMethod"),
    "tabpfn_survival_classifier": ("survarena.methods.foundation.tabpfn_survival", "TabPFNSurvivalClassifierMethod"),
    "tabpfn_survival_regressor": ("survarena.methods.foundation.tabpfn_survival", "TabPFNSurvivalRegressorMethod"),
    "tabpfn_survival_horizon": ("survarena.methods.foundation.tabpfn_survival", "TabPFNHorizonSurvivalMethod"),
    "mitra_survival": ("survarena.methods.automl.mitra_survival", "MitraSurvivalMethod"),
    "mitra_survival_frozen": ("survarena.methods.automl.mitra_survival", "MitraSurvivalFrozenMethod"),
}

_AUTOGLUON_METHOD_IDS = frozenset({"mitra_survival", "mitra_survival_frozen"})


def registered_method_ids() -> tuple[str, ...]:
    return tuple(_REGISTRY_TARGETS.keys())


def is_autogluon_method(method_id: str) -> bool:
    return method_id in _AUTOGLUON_METHOD_IDS


@lru_cache(maxsize=None)
def get_method_class(method_id: str) -> Any:
    if method_id not in _REGISTRY_TARGETS:
        raise ValueError(f"Unknown method_id '{method_id}'. Registered: {sorted(_REGISTRY_TARGETS.keys())}")
    module_name, symbol_name = _REGISTRY_TARGETS[method_id]
    module = import_module(module_name)
    return getattr(module, symbol_name)
