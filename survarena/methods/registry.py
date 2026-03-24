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
    "mitra_survival": ("survarena.methods.foundation.mitra_survival", "MitraSurvivalMethod"),
    "tabpfn_survival": ("survarena.methods.foundation.tabpfn_survival", "TabPFNSurvivalMethod"),
}


def registered_method_ids() -> tuple[str, ...]:
    return tuple(_REGISTRY_TARGETS.keys())


@lru_cache(maxsize=None)
def get_method_class(method_id: str) -> Any:
    if method_id not in _REGISTRY_TARGETS:
        raise ValueError(f"Unknown method_id '{method_id}'. Registered: {sorted(_REGISTRY_TARGETS.keys())}")
    module_name, symbol_name = _REGISTRY_TARGETS[method_id]
    module = import_module(module_name)
    return getattr(module, symbol_name)


@lru_cache(maxsize=1)
def method_registry() -> dict[str, Any]:
    return {method_id: get_method_class(method_id) for method_id in _REGISTRY_TARGETS}
