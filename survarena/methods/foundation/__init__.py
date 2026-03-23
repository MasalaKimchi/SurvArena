"""Foundation-model based survival methods."""

from survarena.methods.foundation.catalog import FoundationModelSpec, available_foundation_model_specs, foundation_model_catalog

__all__ = [
    "FoundationModelSpec",
    "MitraSurvivalMethod",
    "TabPFNSurvivalMethod",
    "available_foundation_model_specs",
    "foundation_model_catalog",
]


def __getattr__(name: str):
    if name == "MitraSurvivalMethod":
        from survarena.methods.foundation.mitra_survival import MitraSurvivalMethod

        return MitraSurvivalMethod
    if name == "TabPFNSurvivalMethod":
        from survarena.methods.foundation.tabpfn_survival import TabPFNSurvivalMethod

        return TabPFNSurvivalMethod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
