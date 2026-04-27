from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FoundationModelSpec:
    method_id: str
    backbone: str
    provider: str
    task_support: tuple[str, ...]
    supports_finetune: bool
    supports_pretrained_weights: bool
    status: str
    notes: str
    dependency_module: str | None = None
    install_extra: str | None = None
    max_rows_hint: int | None = None
    max_features_hint: int | None = None
    requires_hf_auth: bool = False


_FOUNDATION_MODEL_SPECS: tuple[FoundationModelSpec, ...] = (
    FoundationModelSpec(
        method_id="tabpfn_survival",
        backbone="TabPFN",
        provider="Prior Labs",
        task_support=("regression", "classification"),
        supports_finetune=False,
        supports_pretrained_weights=True,
        status="implemented",
        notes="Supports explicit TabPFN v2/v2.5 selection plus custom checkpoint paths.",
        dependency_module="tabpfn",
        install_extra="foundation-tabpfn",
        max_rows_hint=10_000,
        max_features_hint=500,
        requires_hf_auth=True,
    ),
    FoundationModelSpec(
        method_id="tabicl_survival",
        backbone="TabICL",
        provider="AutoGluon / SODA-Inria",
        task_support=("classification", "regression"),
        supports_finetune=True,
        supports_pretrained_weights=True,
        status="catalog_only",
        notes="Candidate future backbone; not yet wired into a survival adapter.",
        max_rows_hint=30_000,
        max_features_hint=2_000,
    ),
    FoundationModelSpec(
        method_id="tabdpt_survival",
        backbone="TabDPT",
        provider="AutoGluon / Layer 6",
        task_support=("classification", "regression"),
        supports_finetune=False,
        supports_pretrained_weights=True,
        status="catalog_only",
        notes="Candidate future backbone; not yet wired into a survival adapter.",
        max_rows_hint=30_000,
        max_features_hint=2_000,
    ),
    FoundationModelSpec(
        method_id="realtabpfn_survival",
        backbone="RealTabPFN-2.5",
        provider="AutoGluon / Prior Labs",
        task_support=("classification", "regression"),
        supports_finetune=False,
        supports_pretrained_weights=True,
        status="catalog_only",
        notes="Candidate future backbone; not yet wired into a survival adapter.",
        max_rows_hint=50_000,
        max_features_hint=2_000,
    ),
)


def foundation_model_catalog() -> tuple[FoundationModelSpec, ...]:
    return _FOUNDATION_MODEL_SPECS


_WIRED_FOUNDATION_METHOD_IDS = frozenset({"tabpfn_survival"})


def available_foundation_model_specs() -> tuple[FoundationModelSpec, ...]:
    return tuple(spec for spec in _FOUNDATION_MODEL_SPECS if spec.method_id in _WIRED_FOUNDATION_METHOD_IDS)
