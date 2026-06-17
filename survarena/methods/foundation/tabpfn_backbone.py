from __future__ import annotations

from typing import Any

import numpy as np


def kaplan_meier_survival_at(time_train: np.ndarray, event_train: np.ndarray, times: np.ndarray) -> np.ndarray:
    event_mask = np.asarray(event_train).astype(bool)
    train_time = np.asarray(time_train, dtype=np.float64)
    event_times = np.unique(train_time[event_mask])
    if event_times.size == 0:
        return np.ones_like(np.asarray(times, dtype=np.float64), dtype=np.float64)

    survival_values: list[float] = []
    survival = 1.0
    for event_time in event_times:
        at_risk = float(np.sum(train_time >= event_time))
        if at_risk <= 0.0:
            continue
        observed_events = float(np.sum((train_time == event_time) & event_mask))
        survival *= max(0.0, 1.0 - observed_events / at_risk)
        survival_values.append(survival)

    if not survival_values:
        return np.ones_like(np.asarray(times, dtype=np.float64), dtype=np.float64)
    return np.interp(
        np.asarray(times, dtype=np.float64),
        event_times[: len(survival_values)],
        np.asarray(survival_values, dtype=np.float64),
        left=1.0,
        right=float(survival_values[-1]),
    )


def build_tabpfn_classifier(
    *,
    n_estimators: int,
    fit_mode: str,
    model_version: str,
    checkpoint_path: str | None,
    device: str,
    seed: int | None,
) -> Any:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    base_kwargs = {
        "n_estimators": int(n_estimators),
        "fit_mode": str(fit_mode),
        "device": str(device),
        "random_state": seed,
        "ignore_pretraining_limits": True,
    }
    if checkpoint_path:
        return TabPFNClassifier(model_path=str(checkpoint_path), **base_kwargs)

    resolved_version = str(model_version).lower()
    if resolved_version in {"auto", "default"}:
        return TabPFNClassifier(**base_kwargs)
    version_map = {
        "v2": ModelVersion.V2,
        "v2.5": ModelVersion.V2_5,
        "v2_5": ModelVersion.V2_5,
    }
    if resolved_version not in version_map:
        raise ValueError("model_version must be one of {'auto', 'v2', 'v2.5'}.")
    return TabPFNClassifier.create_default_for_version(version=version_map[resolved_version], **base_kwargs)


_kaplan_meier_survival_at = kaplan_meier_survival_at
