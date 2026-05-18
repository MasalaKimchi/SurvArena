from __future__ import annotations

from typing import Any

import numpy as np

from survarena.methods.base import BaseSurvivalMethod
from survarena.methods.foundation.readiness import ensure_foundation_runtime_ready, rewrite_foundation_runtime_error


def _kaplan_meier_survival_at(time_train: np.ndarray, event_train: np.ndarray, times: np.ndarray) -> np.ndarray:
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


def _clean_horizon_event_probabilities(values: np.ndarray) -> np.ndarray:
    event_prob = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    event_prob = np.clip(event_prob, 0.0, 1.0)
    return np.maximum.accumulate(event_prob, axis=1)


class TabPFNSurvivalMethod(BaseSurvivalMethod):
    """Censored-aware TabPFN horizon classifier survival adapter."""

    def __init__(
        self,
        n_estimators: int = 4,
        fit_mode: str = "fit_preprocessors",
        model_version: str = "v2.5",
        checkpoint_path: str | None = None,
        horizon_quantiles: str | list[float] = "0.25-0.5-0.75",
        min_known_per_horizon: int = 20,
        aggregate_risk: str = "mean_event_probability",
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            fit_mode=fit_mode,
            model_version=model_version,
            checkpoint_path=checkpoint_path,
            horizon_quantiles=horizon_quantiles,
            min_known_per_horizon=min_known_per_horizon,
            aggregate_risk=aggregate_risk,
            device=device,
            seed=seed,
        )
        self.models_: list[Any | None] = []
        self.horizon_times_: np.ndarray | None = None
        self.constant_event_probabilities_: np.ndarray | None = None
        self.used_fallback_: list[bool] = []

    def foundation_metadata(self) -> dict[str, Any]:
        return {
            "foundation_backbone": "TabPFN",
            "foundation_backbone_task": "censored_aware_horizon_classification",
            "foundation_backbone_training": "frozen",
            "foundation_n_estimators": int(self.params["n_estimators"]),
            "foundation_horizon_count": 0 if self.horizon_times_ is None else int(len(self.horizon_times_)),
            "foundation_horizon_fallback_count": int(sum(self.used_fallback_)),
        }

    def _build_backbone(self) -> Any:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        base_kwargs = {
            "n_estimators": int(self.params["n_estimators"]),
            "fit_mode": str(self.params["fit_mode"]),
            "device": str(self.params["device"]),
            "random_state": self.params.get("seed"),
            "ignore_pretraining_limits": True,
        }
        checkpoint_path = self.params.get("checkpoint_path")
        if checkpoint_path:
            return TabPFNClassifier(model_path=str(checkpoint_path), **base_kwargs)

        model_version = str(self.params["model_version"]).lower()
        if model_version in {"auto", "default"}:
            return TabPFNClassifier(**base_kwargs)
        version_map = {
            "v2": ModelVersion.V2,
            "v2.5": ModelVersion.V2_5,
            "v2_5": ModelVersion.V2_5,
        }
        if model_version not in version_map:
            raise ValueError("model_version must be one of {'auto', 'v2', 'v2.5'}.")
        return TabPFNClassifier.create_default_for_version(version=version_map[model_version], **base_kwargs)

    @staticmethod
    def _parse_horizon_quantiles(value: str | list[float]) -> np.ndarray:
        if isinstance(value, str):
            quantiles = [float(part.strip()) for part in value.split("-") if part.strip()]
        else:
            quantiles = [float(part) for part in value]
        if not quantiles:
            raise ValueError("horizon_quantiles must contain at least one quantile.")
        array = np.asarray(quantiles, dtype=np.float64)
        if np.any((array <= 0.0) | (array >= 1.0)):
            raise ValueError("horizon_quantiles must be strictly between 0 and 1.")
        return np.unique(array)

    @staticmethod
    def _horizon_known_labels(
        *,
        time_train: np.ndarray,
        event_train: np.ndarray,
        horizon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        time = np.asarray(time_train, dtype=np.float64)
        event = np.asarray(event_train).astype(bool)
        positive = (time <= float(horizon)) & event
        negative = time > float(horizon)
        known = positive | negative
        return known, positive[known].astype(np.int32)

    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "TabPFNSurvivalMethod":
        del X_val, time_val, event_val
        ensure_foundation_runtime_ready("tabpfn_survival", checkpoint_path=self.params.get("checkpoint_path"))
        try:
            X_train_np = np.asarray(X_train, dtype=np.float32)
            time_train_np = np.asarray(time_train, dtype=np.float64)
            event_train_np = np.asarray(event_train, dtype=np.int32)
            if int(event_train_np.sum()) <= 0:
                raise ValueError("TabPFN survival training requires at least one observed event.")

            quantiles = self._parse_horizon_quantiles(self.params["horizon_quantiles"])
            event_times = time_train_np[event_train_np.astype(bool)]
            self.horizon_times_ = np.unique(np.quantile(event_times, quantiles).astype(np.float64))
            baseline_survival = _kaplan_meier_survival_at(time_train_np, event_train_np, self.horizon_times_)
            baseline_event_prob = 1.0 - baseline_survival

            self.models_ = []
            self.used_fallback_ = []
            constants: list[float] = []
            min_known = int(self.params["min_known_per_horizon"])
            for idx, horizon in enumerate(self.horizon_times_):
                known_mask, labels = self._horizon_known_labels(
                    time_train=time_train_np,
                    event_train=event_train_np,
                    horizon=float(horizon),
                )
                has_both_classes = np.unique(labels).size == 2
                if int(known_mask.sum()) < min_known or not has_both_classes:
                    self.models_.append(None)
                    self.used_fallback_.append(True)
                    constants.append(float(baseline_event_prob[idx]))
                    continue

                model = self._build_backbone()
                model.fit(X_train_np[known_mask], labels)
                self.models_.append(model)
                self.used_fallback_.append(False)
                constants.append(float(baseline_event_prob[idx]))

            self.constant_event_probabilities_ = _clean_horizon_event_probabilities(
                np.asarray(constants, dtype=np.float64)[None, :]
            )[0]
            return self
        except Exception as exc:
            raise rewrite_foundation_runtime_error(
                "tabpfn_survival",
                exc,
                checkpoint_path=self.params.get("checkpoint_path"),
            ) from exc

    @staticmethod
    def _positive_class_probability(model: Any, X: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
        if probabilities.ndim == 1:
            return probabilities.reshape(-1)
        classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))
        positive_positions = np.flatnonzero(classes.astype(str) == "1")
        if positive_positions.size:
            return probabilities[:, int(positive_positions[-1])]
        return probabilities[:, -1]

    def _horizon_event_probabilities(self, X: np.ndarray) -> np.ndarray:
        if self.horizon_times_ is None or self.constant_event_probabilities_ is None:
            raise RuntimeError("TabPFNSurvivalMethod must be fit before prediction.")
        X_np = np.asarray(X, dtype=np.float32)
        columns: list[np.ndarray] = []
        for model, fallback_prob in zip(self.models_, self.constant_event_probabilities_):
            if model is None:
                columns.append(np.full(X_np.shape[0], float(fallback_prob), dtype=np.float64))
            else:
                columns.append(self._positive_class_probability(model, X_np))
        if not columns:
            return np.zeros((X_np.shape[0], 0), dtype=np.float64)
        return _clean_horizon_event_probabilities(np.column_stack(columns))

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        horizon_event_probs = self._horizon_event_probabilities(X)
        if str(self.params["aggregate_risk"]) == "last_event_probability":
            return horizon_event_probs[:, -1].astype(np.float64)
        return horizon_event_probs.mean(axis=1).astype(np.float64)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        if self.horizon_times_ is None:
            raise RuntimeError("TabPFNSurvivalMethod must be fit before prediction.")
        eval_times = np.asarray(times, dtype=np.float64).reshape(-1)
        horizon_event_probs = self._horizon_event_probabilities(X)
        rows: list[np.ndarray] = []
        for row in horizon_event_probs:
            event_prob_at_times = np.interp(
                eval_times,
                self.horizon_times_,
                row,
                left=0.0,
                right=float(row[-1]),
            )
            rows.append(1.0 - event_prob_at_times)
        survival = np.vstack(rows) if rows else np.empty((0, eval_times.size), dtype=np.float64)
        survival = np.nan_to_num(survival, nan=1.0, posinf=1.0, neginf=1e-8)
        survival = np.clip(survival, 1e-8, 1.0)
        return np.minimum.accumulate(survival, axis=1).astype(np.float64)
