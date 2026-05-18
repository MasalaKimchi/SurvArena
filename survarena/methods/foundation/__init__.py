"""Foundation-model based survival methods."""

from importlib import import_module

from survarena.methods.foundation.catalog import FoundationModelSpec, available_foundation_model_specs, foundation_model_catalog
from survarena.methods.foundation.readiness import (
    FoundationRuntimeStatus,
    ensure_foundation_runtime_ready,
    foundation_runtime_catalog,
    foundation_runtime_status,
    foundation_runtime_status_for_method,
    rewrite_foundation_runtime_error,
)

__all__ = [
    "FoundationModelSpec",
    "FoundationRuntimeStatus",
    "TabPFNHorizonSurvivalMethod",
    "TabPFNSurvivalClassifierMethod",
    "TabPFNSurvivalMethod",
    "TabPFNSurvivalRegressorMethod",
    "available_foundation_model_specs",
    "ensure_foundation_runtime_ready",
    "foundation_runtime_catalog",
    "foundation_model_catalog",
    "foundation_runtime_status",
    "foundation_runtime_status_for_method",
    "rewrite_foundation_runtime_error",
]


def __getattr__(name: str):
    if name in {
        "TabPFNHorizonSurvivalMethod",
        "TabPFNSurvivalClassifierMethod",
        "TabPFNSurvivalMethod",
        "TabPFNSurvivalRegressorMethod",
    }:
        module = import_module("survarena.methods.foundation.tabpfn_survival")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
