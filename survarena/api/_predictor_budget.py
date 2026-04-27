from __future__ import annotations

from time import perf_counter
from typing import Any, Callable


def validate_time_limit(time_limit: float | None) -> float | None:
    if time_limit is None:
        return None
    resolved = float(time_limit)
    if resolved <= 0.0:
        raise ValueError("time_limit must be positive when provided.")
    return resolved


def validate_num_bag_folds(num_bag_folds: int) -> int:
    resolved = int(num_bag_folds)
    if resolved < 0:
        raise ValueError("num_bag_folds must be >= 0.")
    if resolved == 1:
        raise ValueError("num_bag_folds must be 0 or >= 2.")
    return resolved


def validate_num_bag_sets(num_bag_sets: int, *, num_bag_folds: int) -> int:
    resolved = int(num_bag_sets)
    if resolved < 1:
        raise ValueError("num_bag_sets must be >= 1.")
    if resolved > 1 and num_bag_folds <= 0:
        raise ValueError("num_bag_sets > 1 requires num_bag_folds >= 2.")
    return resolved


def validate_retain_top_k_models(retain_top_k_models: int | None) -> int | None:
    if retain_top_k_models is None:
        return None
    resolved = int(retain_top_k_models)
    if resolved < 1:
        raise ValueError("retain_top_k_models must be >= 1 or None.")
    return resolved


def resolve_hyperparameter_tune_kwargs(
    hyperparameter_tune_kwargs: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if hyperparameter_tune_kwargs is None:
        return None
    if not isinstance(hyperparameter_tune_kwargs, dict):
        raise TypeError("hyperparameter_tune_kwargs must be a dictionary when provided.")

    supported_keys = {"num_trials", "timeout", "timeout_seconds"}
    unexpected_keys = sorted(set(hyperparameter_tune_kwargs) - supported_keys)
    if unexpected_keys:
        raise ValueError(
            "Unsupported hyperparameter_tune_kwargs keys: "
            f"{unexpected_keys}. Supported keys: {sorted(supported_keys)}"
        )
    if "timeout" in hyperparameter_tune_kwargs and "timeout_seconds" in hyperparameter_tune_kwargs:
        raise ValueError("Specify only one of 'timeout' or 'timeout_seconds' in hyperparameter_tune_kwargs.")

    normalized: dict[str, Any] = {}
    num_trials = hyperparameter_tune_kwargs.get("num_trials")
    if num_trials is not None:
        resolved_num_trials = int(num_trials)
        if resolved_num_trials < 0:
            raise ValueError("hyperparameter_tune_kwargs num_trials must be >= 0.")
        normalized["num_trials"] = resolved_num_trials

    timeout_seconds = hyperparameter_tune_kwargs.get("timeout_seconds", hyperparameter_tune_kwargs.get("timeout"))
    if timeout_seconds is not None:
        resolved_timeout = float(timeout_seconds)
        if resolved_timeout <= 0.0:
            raise ValueError("hyperparameter_tune_kwargs timeout must be positive.")
        normalized["timeout_seconds"] = resolved_timeout

    return normalized


def resolve_tuning_timeout_seconds(fit_tune_kwargs: dict[str, Any] | None) -> float | None:
    if fit_tune_kwargs is None:
        return None
    timeout_seconds = fit_tune_kwargs.get("timeout_seconds")
    return None if timeout_seconds is None else float(timeout_seconds)


def remaining_fit_time(fit_started_at: float, time_limit: float | None) -> float:
    if time_limit is None:
        return float("inf")
    return max(0.0, float(time_limit) - float(perf_counter() - fit_started_at))


def next_method_time_limit(
    *,
    fit_started_at: float,
    selection_time_budget: float | None,
    remaining_methods: int,
    remaining_fit_time_fn: Callable[[float, float | None], float] = remaining_fit_time,
) -> float | None:
    if selection_time_budget is None:
        return None
    if remaining_methods <= 0:
        return 0.0
    remaining_budget = remaining_fit_time_fn(fit_started_at, selection_time_budget)
    if remaining_budget <= 0.0:
        return 0.0
    return remaining_budget / float(remaining_methods)


def merge_time_limits(first: float | None, second: float | None) -> float | None:
    limits = [float(limit) for limit in (first, second) if limit is not None]
    if not limits:
        return None
    return min(limits)
