"""Immutable run-result schema and the queryable results store.

Phase-1a of the strangler-fig refactor. This subpackage is dependency-light
(stdlib ``sqlite3`` + ``pandas`` only) and importable without the ML stack:

* :class:`~survarena.core.results.schema.RunResult` -- the frozen, hashable-keyed
  record for one evaluated ``(protocol, dataset, method, hpo_mode, split, seed)``
  unit.
* :class:`~survarena.core.results.store.ResultStore` -- an idempotent,
  SQL-queryable store (SQLite reference backend, optional Parquet export).
"""

from __future__ import annotations

from survarena.core.results.schema import RunResult
from survarena.core.results.store import ResultStore

__all__ = ["RunResult", "ResultStore"]
