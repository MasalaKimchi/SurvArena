"""Typed, versioned benchmark protocol schema.

``survarena.core.protocol`` replaces the ``configs/benchmark/*.yaml`` sprawl with
a single validated :class:`ProtocolSpec` (plus its sub-specs). See
``spec.py``'s module docstring and ``SURVARENA_REVAMP_PLAN.md`` §3.4 for the
design and versioning contract. This package is additive and imports no ML
dependencies.
"""

from survarena.core.protocol.spec import (
    ALLOWED_COMPARISON_MODES,
    EXPECTED_TASK_TYPE,
    PROTOCOL_VERSION,
    AutoGluonSpec,
    ExportsSpec,
    HpoSpec,
    MethodOverride,
    ProtocolSpec,
)

__all__ = [
    "ProtocolSpec",
    "AutoGluonSpec",
    "HpoSpec",
    "MethodOverride",
    "ExportsSpec",
    "PROTOCOL_VERSION",
    "ALLOWED_COMPARISON_MODES",
    "EXPECTED_TASK_TYPE",
]
