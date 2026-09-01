from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

# Allowed dependency direction (methods -> core): the capability record lives in
# the dependency-light core kernel and pulls in no heavy ML deps.
from survarena.core.models import ModelCapabilities


@dataclass(frozen=True, slots=True)
class SurvivalPredictions:
    risk: np.ndarray
    survival: np.ndarray


class BaseSurvivalMethod(ABC):
    # True iff fit() actually uses the (X_val, time_val, event_val) arguments
    # (e.g. for early stopping / internal tuning). The runner only carves a
    # validation holdout for such methods; others are fit on the full training
    # set. Conservative default: a method is assumed NOT to use validation unless
    # it declares otherwise.
    consumes_validation: bool = False

    # Additional capability dimensions (Phase-1a). These mirror the remaining
    # fields of ``survarena.core.models.ModelCapabilities`` and are surfaced via
    # the forward-looking ``capabilities()`` classmethod below. Conservative
    # defaults keep every existing adapter's behaviour unchanged.
    supports_early_stopping: bool = False
    refit_on_full_train: bool = False
    native_categoricals: bool = False
    supports_gpu: bool = False
    deterministic_given_seed: bool = True
    supports_survival_distribution: bool = True

    @classmethod
    def capabilities(cls) -> ModelCapabilities:
        # Forward interface: build the immutable capability record from the class
        # attributes above. ``uses_validation`` is intentionally sourced from
        # ``consumes_validation`` -- that Phase-0 flag stays the single source of
        # truth, so the runner's existing validation-holdout gate keeps working
        # unchanged while adapters migrate onto the typed contract in Phase-1b.
        return ModelCapabilities(
            uses_validation=cls.consumes_validation,
            supports_early_stopping=cls.supports_early_stopping,
            refit_on_full_train=cls.refit_on_full_train,
            native_categoricals=cls.native_categoricals,
            supports_gpu=cls.supports_gpu,
            deterministic_given_seed=cls.deterministic_given_seed,
        )

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        time_train: np.ndarray,
        event_train: np.ndarray,
        X_val: np.ndarray | None = None,
        time_val: np.ndarray | None = None,
        event_val: np.ndarray | None = None,
    ) -> "BaseSurvivalMethod":
        raise NotImplementedError

    @abstractmethod
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_bundle(self, X: np.ndarray, times: np.ndarray) -> SurvivalPredictions:
        return SurvivalPredictions(
            risk=np.asarray(self.predict_risk(X)),
            survival=np.asarray(self.predict_survival(X, times)),
        )

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def set_params(self, **kwargs: Any) -> "BaseSurvivalMethod":
        self.params.update(kwargs)
        return self


def to_structured_y(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    y = np.zeros(len(time), dtype=[("event", "?"), ("time", "f8")])
    y["event"] = event.astype(bool)
    y["time"] = time.astype(float)
    return y
