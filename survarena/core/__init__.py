"""SurvArena core kernel.

This package is the *preserved, dependency-light* core of the strangler-fig
refactor (see ``SURVARENA_REVAMP_PLAN.md``). Modules under ``survarena.core`` must
NOT import heavy ML dependencies (torch, autogluon, xgboost, ...) and must NOT
import "upward" into ``survarena.bench``, ``survarena.api``, or model adapters.
The allowed dependency direction is ``platform -> bench -> models -> core``.

Subpackages:
- ``core.models``   -- the model capability contract (``ModelCapabilities``,
  ``SurvivalModel``) that formalises the Phase-0 ``consumes_validation`` flag.
- ``core.results``  -- the immutable ``RunResult`` schema and the queryable
  results store (stdlib ``sqlite3`` reference backend; optional Parquet export).
- ``core.protocol`` -- the typed, versioned ``ProtocolSpec`` that replaces the
  benchmark-YAML sprawl.

Everything here is importable without the ML stack installed, which keeps the
leaderboard/results/query tooling runnable on any machine.
"""
