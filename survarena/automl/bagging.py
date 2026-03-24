from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from survarena.data.preprocess import TabularPreprocessor


@dataclass(slots=True)
class BaggedModelMember:
    model: Any
    preprocessor: TabularPreprocessor


class BaggedSurvivalEnsemble:
    def __init__(self, members: list[BaggedModelMember]) -> None:
        if not members:
            raise ValueError("BaggedSurvivalEnsemble requires at least one fitted member.")
        self.members = list(members)

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        member_predictions = [
            np.asarray(member.model.predict_risk(member.preprocessor.transform(X).to_numpy()), dtype=float)
            for member in self.members
        ]
        return np.mean(np.stack(member_predictions, axis=0), axis=0)

    def predict_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        evaluation_times = np.asarray(times, dtype=float)
        member_predictions = [
            np.asarray(
                member.model.predict_survival(member.preprocessor.transform(X).to_numpy(), evaluation_times),
                dtype=float,
            )
            for member in self.members
        ]
        return np.mean(np.stack(member_predictions, axis=0), axis=0)

    def __len__(self) -> int:
        return len(self.members)
