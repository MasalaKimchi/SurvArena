from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseSurvivalMethod(ABC):
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
