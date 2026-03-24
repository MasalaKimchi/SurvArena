from __future__ import annotations

from importlib import import_module
from functools import lru_cache
from typing import Any


_REGISTRY_TARGETS = {
    "coxph": ("survarena.methods.classical.coxph", "CoxPHMethod"),
    "coxnet": ("survarena.methods.classical.coxnet", "CoxNetMethod"),
    "rsf": ("survarena.methods.tree.rsf", "RSFMethod"),
    "deepsurv": ("survarena.methods.deep.deepsurv", "DeepSurvMethod"),
    "deepsurv_moco": ("survarena.methods.deep.deepsurv_moco", "DeepSurvMomentumMethod"),
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
