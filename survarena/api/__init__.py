"""High-level public APIs for SurvArena."""

from survarena.api.compare import compare_survival_models
from survarena.api.predictor import SurvivalPredictor

__all__ = ["SurvivalPredictor", "compare_survival_models"]
