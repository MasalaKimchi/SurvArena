from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PresetConfig:
    name: str
    method_ids: tuple[str, ...]
    n_trials: int
    inner_folds: int
    scale_limit_rows: int | None = None
    portfolio_notes: tuple[str, ...] = ()


_PRESETS: dict[str, PresetConfig] = {
    "fast": PresetConfig(
        name="fast",
        method_ids=("coxph", "rsf"),
        n_trials=2,
        inner_folds=3,
        scale_limit_rows=100_000,
    ),
    "medium": PresetConfig(
        name="medium",
        method_ids=("coxph", "coxnet", "rsf", "deepsurv"),
        n_trials=8,
        inner_folds=3,
        scale_limit_rows=50_000,
    ),
    "best": PresetConfig(
        name="best",
        method_ids=("coxph", "coxnet", "rsf", "deepsurv", "deepsurv_moco"),
        n_trials=16,
        inner_folds=5,
        scale_limit_rows=25_000,
    ),
}


def resolve_preset(
    preset_name: str,
    *,
    n_rows: int,
    n_features: int,
    event_count: int | None = None,
    event_fraction: float | None = None,
    high_cardinality_feature_count: int = 0,
    has_datetime_features: bool = False,
    has_text_features: bool = False,
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
    enable_foundation_models: bool = False,
) -> PresetConfig:
    if preset_name not in _PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {sorted(_PRESETS)}")

    preset = _PRESETS[preset_name]
    method_ids = list(preset.method_ids)
    portfolio_notes: list[str] = []

    if preset.scale_limit_rows is not None and n_rows > preset.scale_limit_rows:
        method_ids = [method_id for method_id in method_ids if not method_id.startswith("deepsurv")]
        portfolio_notes.append(
            f"Skipped DeepSurv-family models because dataset rows ({n_rows}) exceed preset scaling limit."
        )

    if n_features > 5_000:
        method_ids = [method_id for method_id in method_ids if method_id != "deepsurv_moco"]
        portfolio_notes.append(
            f"Skipped deepsurv_moco because feature count ({n_features}) exceeds dense-network heuristic."
        )

    if event_count is not None and event_count < 25:
        method_ids = [method_id for method_id in method_ids if method_id not in {"deepsurv", "deepsurv_moco"}]
        portfolio_notes.append(
            f"Skipped deep survival models because only {event_count} observed events are available."
        )

    if event_fraction is not None and event_fraction < 0.1:
        method_ids = [method_id for method_id in method_ids if method_id != "deepsurv_moco"]
        portfolio_notes.append("Skipped deepsurv_moco because the observed event rate is below 10%.")

    foundation_supported = True
    if high_cardinality_feature_count > 0:
        foundation_supported = False
        portfolio_notes.append(
            "Skipped foundation models because high-cardinality categorical features currently require specialized encoding."
        )
    if has_datetime_features:
        foundation_supported = False
        portfolio_notes.append("Skipped foundation models because datetime-aware feature handling is not implemented yet.")
    if has_text_features:
        foundation_supported = False
        portfolio_notes.append("Skipped foundation models because text-aware feature handling is not implemented yet.")
    if event_count is not None and event_count < 25:
        foundation_supported = False
        portfolio_notes.append("Skipped foundation models because the event count is too low for stable survival-head fitting.")

    if enable_foundation_models and foundation_supported and n_rows <= 10_000 and n_features <= 500:
        method_ids.append("tabpfn_survival")
    elif enable_foundation_models and foundation_supported:
        portfolio_notes.append(
            "Skipped foundation models because dataset size exceeds the current TabPFN survival adapter heuristic."
        )

    if included_models is not None:
        include_set = set(included_models)
        method_ids = [method_id for method_id in method_ids if method_id in include_set]
        for method_id in included_models:
            if method_id not in method_ids:
                method_ids.append(method_id)

    if excluded_models is not None:
        exclude_set = set(excluded_models)
        method_ids = [method_id for method_id in method_ids if method_id not in exclude_set]

    if not method_ids:
        raise ValueError("Model portfolio is empty after applying preset filters.")

    return PresetConfig(
        name=preset.name,
        method_ids=tuple(dict.fromkeys(method_ids)),
        n_trials=preset.n_trials,
        inner_folds=preset.inner_folds,
        scale_limit_rows=preset.scale_limit_rows,
        portfolio_notes=tuple(dict.fromkeys(portfolio_notes)),
    )
