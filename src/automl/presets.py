from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PresetConfig:
    name: str
    method_ids: tuple[str, ...]
    n_trials: int
    inner_folds: int
    scale_limit_rows: int | None = None


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
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
) -> PresetConfig:
    if preset_name not in _PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {sorted(_PRESETS)}")

    preset = _PRESETS[preset_name]
    method_ids = list(preset.method_ids)

    if preset.scale_limit_rows is not None and n_rows > preset.scale_limit_rows:
        method_ids = [method_id for method_id in method_ids if not method_id.startswith("deepsurv")]

    if n_features > 5_000:
        method_ids = [method_id for method_id in method_ids if method_id != "deepsurv_moco"]

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
    )
