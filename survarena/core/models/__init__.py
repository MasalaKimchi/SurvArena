"""Model capability contract (Phase-1a of the strangler-fig refactor).

Re-exports the dependency-light capability vocabulary shared by the runner and
the future typed adapters. Importing this package pulls in only the standard
library -- no numpy, torch, or model adapters -- so it stays importable on any
machine and honours the ``... -> models -> core`` dependency direction.
"""

from survarena.core.models.contract import ModelCapabilities, SurvivalModel

__all__ = ["ModelCapabilities", "SurvivalModel"]
