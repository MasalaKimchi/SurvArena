"""Public SurvArena package."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["SurvivalPredictions", "SurvivalPredictor", "compare_survival_models"]

if TYPE_CHECKING:
    from survarena.api import (
        SurvivalPredictions,
        SurvivalPredictor,
        compare_survival_models,
    )


def __getattr__(name: str) -> object:
    # PEP 562 lazy re-export: keep `import survarena` free of the heavy dependency
    # chain (torch, autogluon, ...) that survarena.api pulls in. The public names are
    # only imported on first access and then cached in the module namespace.
    if name in __all__:
        from survarena.api import (
            SurvivalPredictions,
            SurvivalPredictor,
            compare_survival_models,
        )

        globals().update(
            SurvivalPredictions=SurvivalPredictions,
            SurvivalPredictor=SurvivalPredictor,
            compare_survival_models=compare_survival_models,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
