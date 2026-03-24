"""Deep survival method implementations."""

from importlib import import_module
from typing import Any


_DEEP_EXPORTS = {
    "DeepSurvMethod": ("survarena.methods.deep.deepsurv", "DeepSurvMethod"),
    "DeepSurvMomentumMethod": ("survarena.methods.deep.deepsurv_moco", "DeepSurvMomentumMethod"),
    "LogisticHazardMethod": ("survarena.methods.deep.pycox_models", "LogisticHazardMethod"),
    "PMFMethod": ("survarena.methods.deep.pycox_models", "PMFMethod"),
    "MTLRMethod": ("survarena.methods.deep.pycox_models", "MTLRMethod"),
    "DeepHitSingleMethod": ("survarena.methods.deep.pycox_models", "DeepHitSingleMethod"),
    "PCHazardMethod": ("survarena.methods.deep.pycox_models", "PCHazardMethod"),
    "CoxTimeMethod": ("survarena.methods.deep.pycox_models", "CoxTimeMethod"),
}

__all__ = list(_DEEP_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _DEEP_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _DEEP_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, symbol_name)
