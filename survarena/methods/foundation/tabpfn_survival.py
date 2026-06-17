from __future__ import annotations

from survarena.methods.foundation.discrete_hazard import TabPFNDiscreteHazardSurvivalMethod
from survarena.methods.foundation.tabpfn_backbone import _kaplan_meier_survival_at


class TabPFNSurvivalMethod(TabPFNDiscreteHazardSurvivalMethod):
    """Compatibility ID for the default TabPFN discrete-hazard survival adapter."""

    method_id = "tabpfn_survival"


__all__ = ["TabPFNSurvivalMethod", "_kaplan_meier_survival_at"]
