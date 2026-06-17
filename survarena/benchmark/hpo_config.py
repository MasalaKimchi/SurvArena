from __future__ import annotations

from typing import Any, Callable


def method_hpo_overrides(hpo_cfg: dict[str, Any], method_id: str) -> dict[str, Any]:
    overrides = hpo_cfg.get("method_overrides", {})
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError("hpo.method_overrides must be a mapping when provided.")
    method_override = overrides.get(method_id, {})
    if method_override is None:
        return {}
    if not isinstance(method_override, dict):
        raise ValueError(f"hpo.method_overrides.{method_id} must be a mapping when provided.")
    return dict(method_override)


def method_cfg_with_hpo_overrides(
    method_cfg: dict[str, Any],
    *,
    method_id: str,
    method_override: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(method_cfg)
    if "default_params" in method_override:
        default_params = method_override.get("default_params")
        if default_params is None:
            resolved["default_params"] = {}
        elif isinstance(default_params, dict):
            resolved["default_params"] = {
                **dict(resolved.get("default_params", {})),
                **dict(default_params),
            }
        else:
            raise ValueError(f"hpo.method_overrides.{method_id}.default_params must be a mapping or null.")
    if "search_space" in method_override:
        search_space = method_override.get("search_space")
        if search_space is None:
            resolved["search_space"] = {}
        elif isinstance(search_space, dict):
            resolved["search_space"] = dict(search_space)
        else:
            raise ValueError(f"hpo.method_overrides.{method_id}.search_space must be a mapping or null.")
    return resolved


def mode_hpo_cfg_with_method_overrides(
    hpo_cfg: dict[str, Any],
    *,
    hpo_enabled: bool,
    method_override: dict[str, Any],
) -> dict[str, Any]:
    mode_hpo_cfg = {key: value for key, value in hpo_cfg.items() if key != "method_overrides"}
    for key, value in method_override.items():
        if key != "search_space":
            mode_hpo_cfg[key] = value
    mode_hpo_cfg["enabled"] = bool(method_override.get("enabled", True)) if hpo_enabled else False
    return mode_hpo_cfg


def method_cfg_with_autogluon_defaults(
    method_cfg: dict[str, Any],
    autogluon_cfg: dict[str, Any] | None,
    *,
    is_autogluon_method: Callable[[str], bool],
) -> dict[str, Any]:
    if not is_autogluon_method(str(method_cfg.get("method_id"))):
        return method_cfg
    merged = dict(method_cfg)
    defaults = dict(method_cfg.get("default_params", {}))
    if autogluon_cfg:
        defaults.update(
            {
                "presets": autogluon_cfg.get("presets", defaults.get("presets", "medium")),
                "time_limit": autogluon_cfg.get("time_limit_seconds", defaults.get("time_limit")),
                "hyperparameter_tune_kwargs": autogluon_cfg.get(
                    "hyperparameter_tune_kwargs",
                    defaults.get("hyperparameter_tune_kwargs"),
                ),
                "num_bag_folds": autogluon_cfg.get("num_bag_folds", defaults.get("num_bag_folds", 0)),
                "num_stack_levels": autogluon_cfg.get("num_stack_levels", defaults.get("num_stack_levels", 0)),
                "refit_full": autogluon_cfg.get("refit_full", defaults.get("refit_full", False)),
            }
        )
    merged["default_params"] = defaults
    return merged
