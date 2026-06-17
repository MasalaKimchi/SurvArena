from __future__ import annotations

from typing import Any

from survarena.methods.foundation.discrete_hazard import TabPFNDiscreteHazardSurvivalMethod
from survarena.methods.foundation.inference import positive_class_probability_with_backoff
from survarena.methods.foundation.tabpfn_backbone import _kaplan_meier_survival_at


class TabPFNSurvivalMethod(TabPFNDiscreteHazardSurvivalMethod):
    """Compatibility ID for the default TabPFN discrete-hazard survival adapter."""

    method_id = "tabpfn_survival"

    @staticmethod
    def _positive_class_probability(model: Any, X: Any, *, batch_size: int):
        return positive_class_probability_with_backoff(model, X, batch_size=batch_size)


__all__ = ["TabPFNSurvivalMethod", "_kaplan_meier_survival_at"]
