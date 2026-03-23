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


@lru_cache(maxsize=1)
def method_registry() -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for method_id, (module_name, symbol_name) in _REGISTRY_TARGETS.items():
        module = import_module(module_name)
        registry[method_id] = getattr(module, symbol_name)
    return registry
