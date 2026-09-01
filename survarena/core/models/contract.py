"""Model capability contract for the SurvArena core kernel (Phase-1a).

This module is intentionally dependency-light: it imports only the standard
library (``dataclasses`` / ``typing``). It pulls in NO numpy, torch, or any model
adapter, so the leaderboard / results / query tooling and static type-checkers
can depend on it on any machine. Nothing here imports "upward" into
``survarena.bench``, ``survarena.api``, or the model adapters -- the allowed
dependency direction is ``platform -> bench -> models -> core``.

It formalises the Phase-0 ``consumes_validation`` flag on
``survarena.methods.base.BaseSurvivalMethod`` into a small, explicit contract:

* :class:`ModelCapabilities` -- a frozen record of the six capability dimensions
  the runner needs in order to spend the training data fairly.
* :class:`SurvivalModel` -- an aspirational, ``@runtime_checkable`` Protocol that
  documents the *future* typed adapter interface. Phase-1b migrates the existing
  adapters onto it; Phase-1a only introduces the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = ["ModelCapabilities", "SurvivalModel"]


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Declared capabilities of a survival model, consumed by the runner.

    The runner reads these *declared* flags instead of guessing from a method's
    name or from which positional ``fit`` arguments an adapter happens to ignore.
    That makes the train/validation policy an explicit, audited property of each
    model -- the structural form of the Phase-0 fairness fix.

    The record is immutable (``frozen=True``) and memory-light (``slots=True``);
    constructing one is the only per-model allocation on the capability path.
    """

    uses_validation: bool
    """Whether ``fit`` actually consumes a validation fold.

    The runner carves a stratified validation holdout out of the training data
    **only when this is True**; otherwise the model is fit on the full training
    set. This is the source-of-truth switch behind the Phase-0 fairness fix: a
    model that ignores validation is never silently handicapped by a holdout, and
    a model that needs one always receives it. No default -- every model must
    declare this explicitly.
    """

    supports_early_stopping: bool = False
    """Whether the model uses the validation fold to pick an iteration/epoch
    budget (e.g. gradient-boosting rounds, neural-net epochs). Informational for
    the runner/leaderboard; in practice implies ``uses_validation is True``."""

    refit_on_full_train: bool = False
    """Whether, after using validation for model selection, the model can be
    refit on ``train u val`` so it pays no data penalty at inference time. When
    True the runner may hand the held-out fold back once selection is done, so an
    early-stopping model is not permanently charged the holdout fraction."""

    native_categoricals: bool = False
    """Whether the model consumes categorical features natively (no one-hot /
    ordinal pre-encoding required). Lets the runner skip lossy encoding steps and
    pass raw categorical columns straight through."""

    supports_gpu: bool = False
    """Whether the model can use a GPU when one is available. Used by the runner
    for scheduling / device budgeting only; correctness must never depend on it."""

    deterministic_given_seed: bool = True
    """Whether ``fit`` / ``predict`` are reproducible given a fixed seed. When
    False the runner may record extra provenance or average across seeds.
    Conservative default is True (assume reproducibility)."""


@runtime_checkable
class SurvivalModel(Protocol):
    """Aspirational typed interface for a survival-model adapter.

    This Protocol documents the *target* shape of an adapter after the Phase-1b
    migration; it exists for static type-checking and documentation. It is NOT
    imposed on the existing ``BaseSurvivalMethod`` subclasses yet -- Phase-1a only
    introduces the contract vocabulary, and Phase-1b ports the adapters onto it.

    ``capabilities`` is declared here as a ``@classmethod`` (rather than the bare
    ``ClassVar`` sketch in the revamp plan) so it matches the concrete
    ``BaseSurvivalMethod.capabilities()`` shipped in Phase-1a: a single class-level
    entry point that computes the immutable :class:`ModelCapabilities` record.

    The parameter/return types are deliberately ``Any`` here because the concrete
    ``SurvivalDataset`` and array types live in sibling ``core`` subpackages that
    land later; Phase-1b will tighten these annotations (e.g. ``np.ndarray``).
    """

    @classmethod
    def capabilities(cls) -> ModelCapabilities:
        """Return the model's declared :class:`ModelCapabilities`."""
        ...

    def fit(self, train: Any, val: Optional[Any] = None) -> "SurvivalModel":
        """Fit on ``train``; consult ``val`` only when
        ``capabilities().uses_validation`` is True. Returns ``self``."""
        ...

    def predict_risk(self, X: Any) -> Any:
        """Return one scalar risk score per row (higher = riskier)."""
        ...

    def predict_survival(self, X: Any, times: Any) -> Any:
        """Return ``S(t | x)``: survival probabilities on the ``times`` grid."""
        ...
