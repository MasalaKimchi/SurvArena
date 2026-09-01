"""Immutable per-run result record for the strangler-fig ``core`` kernel.

This module defines :class:`RunResult`, the *single canonical shape* for one
evaluated ``(protocol, dataset, method, hpo_mode, split, seed)`` unit. It is the
Phase-1a replacement for the ad-hoc flat ``record`` dictionary that
``survarena.benchmark.runner`` assembles per fold and hands to
``survarena.logging.export.export_fold_results``.

Design goals
------------
* **Immutable & value-typed** -- ``@dataclass(frozen=True, slots=True)`` so a
  result cannot be mutated in place after a run finishes, and ``slots`` keeps the
  per-row memory footprint small (one row per fold * seed * method * dataset can
  be many thousands of objects).
* **Dependency-light** -- stdlib only (``dataclasses``, ``hashlib``,
  ``datetime``). No numpy/pandas/torch import at module load, so the schema is
  importable on any machine (see ``survarena.core`` package docstring).
* **Faithful to the current runner** -- :meth:`RunResult.from_fold_row` maps the
  runner's flat dict (the value returned by ``evaluate_split`` / ``run_unit`` and
  materialised by ``export_fold_results``) onto this schema, tolerating absent
  keys. See the "Runner mapping" table below.

Runner mapping (survarena/benchmark/runner.py -> RunResult)
-----------------------------------------------------------
The runner's per-fold ``record`` dict (assembled around runner.py lines 452-499
for success and 551-588 for failure, then decorated at lines 908-923) carries::

    benchmark_id, dataset_id, method_id, split_id, seed, hpo_mode, status,
    comparison_ineligible, ineligible_reason,        # identity / eligibility
    validation_score, uno_c, harrell_c, ibs,         # CORE_METRIC_COLUMNS ...
    td_auc_25/50/75, validation_diagnostic_*,        # ... (metrics)
    brier_*, calibration_{slope,intercept}_abs_error_*, net_benefit_*,  # MANUSCRIPT
    tuning_time_sec, runtime_sec, fit_time_sec, infer_time_sec, peak_memory_mb  # timing

Scalar mapping (runner key -> RunResult field)::

    benchmark_id            -> benchmark_id
    dataset_id              -> dataset_id
    method_id               -> method_id
    hpo_mode                -> hpo_mode        (default "no_hpo" if absent, matching
                                                export_coverage_matrix's fallback)
    split_id                -> split_id
    seed                    -> seed
    status                  -> status          (default "unknown" if absent -> ineligible)
    comparison_ineligible   -> comparison_ineligible   (default False)
    ineligible_reason       -> ineligible_reason       (default "")
    fit_time_sec            -> fit_seconds
    infer_time_sec          -> predict_seconds
    peak_memory_mb          -> peak_rss_mb

The metric columns (``CORE_METRIC_COLUMNS`` + ``MANUSCRIPT_METRIC_COLUMNS`` from
``survarena.logging.export_shared``, plus any dynamic ``td_auc_/brier_/net_benefit_``
horizon columns) are gathered into the immutable ``metrics`` mapping. NaN/absent
metric values are simply *not stored* (sparse), which keeps "metric present"
semantics identical to ``eligible_frame``'s ``frame[metric].notna()`` filter.

New fields not produced by the current runner (``protocol_id``,
``dataset_version``, ``dataset_sha256``, ``method_version``, ``env_fingerprint``,
``created_at``) default sensibly; ``protocol_id`` is a required keyword on
:meth:`from_fold_row` because it is the new provenance anchor introduced by the
refactor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
import hashlib
import math
from typing import Any, cast

__all__ = ["RunResult", "KNOWN_METRIC_COLUMNS"]

# --- Canonical metric column names -----------------------------------------
# Mirrors survarena/logging/export_shared.py (CORE_METRIC_COLUMNS +
# MANUSCRIPT_METRIC_COLUMNS) as of schema_version 1. Duplicated here on purpose
# so that ``core.results`` stays decoupled from the legacy ``logging`` package
# (strangler-fig: new code must not depend on the module it will eventually
# replace). Keep in sync when the metric set changes and bump ``schema_version``.
_CORE_METRIC_COLUMNS: tuple[str, ...] = (
    "validation_score",
    "validation_diagnostic_score",
    "validation_diagnostic_test_score",
    "validation_diagnostic_gap",
    "validation_diagnostic_horizon_auc",
    "validation_diagnostic_test_horizon_auc",
    "validation_diagnostic_horizon_auc_gap",
    "uno_c",
    "harrell_c",
    "ibs",
    "td_auc_25",
    "td_auc_50",
    "td_auc_75",
)
_MANUSCRIPT_METRIC_COLUMNS: tuple[str, ...] = (
    "brier_25",
    "brier_50",
    "brier_75",
    "calibration_slope_abs_error_25",
    "calibration_slope_abs_error_50",
    "calibration_slope_abs_error_75",
    "calibration_intercept_abs_error_25",
    "calibration_intercept_abs_error_50",
    "calibration_intercept_abs_error_75",
    "net_benefit_25",
    "net_benefit_50",
    "net_benefit_75",
)
#: Full explicit allow-list of metric column names understood by ``from_fold_row``.
KNOWN_METRIC_COLUMNS: frozenset[str] = frozenset(_CORE_METRIC_COLUMNS + _MANUSCRIPT_METRIC_COLUMNS)

# Dynamic metrics: horizon-suffixed families whose fixed-horizon members are
# listed above (``td_auc_``/``brier_``/``net_benefit_``). A column is treated as
# a metric if it starts with one of these prefixes and the remainder is all
# digits (e.g. ``td_auc_10``).
_DYNAMIC_METRIC_PREFIXES: tuple[str, ...] = ("td_auc_", "brier_", "net_benefit_")

_METRIC_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "metric_support_policy",
    "metric_support_lower",
    "metric_support_upper",
    "evaluation_window_lower",
    "evaluation_window_upper",
    "full_distribution_eligible",
    "full_distribution_ineligible_reason",
    "horizon_requested_25",
    "horizon_requested_50",
    "horizon_requested_75",
    "horizon_used_25",
    "horizon_used_50",
    "horizon_used_75",
    "horizon_eligible_25",
    "horizon_eligible_50",
    "horizon_eligible_75",
    "horizon_reason_25",
    "horizon_reason_50",
    "horizon_reason_75",
)


def _is_dynamic_metric(name: str) -> bool:
    """Return ``True`` for a horizon-suffixed dynamic metric (e.g. ``td_auc_10``).

    A name matches when it starts with one of :data:`_DYNAMIC_METRIC_PREFIXES`
    and the remaining suffix is all digits. This does *not* exactly mirror
    ``export_shared.expand_dynamic_metric_columns`` (which only emits the
    horizons currently configured): it intentionally accepts *any* integer
    horizon suffix so future configurable horizons (Phase-2) are recognised
    without editing this module. The wider acceptance set is a deliberate
    superset -- safe because non-metric columns do not carry these prefixes
    followed by a pure-digit tail.
    """
    for prefix in _DYNAMIC_METRIC_PREFIXES:
        if name.startswith(prefix):
            suffix = name[len(prefix):]
            if suffix.isdigit():
                return True
    return False


# --- Tolerant scalar coercers (accept python/numpy/None/NaN) ----------------

def _as_float(value: object, default: float = float("nan")) -> float:
    """Coerce ``value`` to ``float`` (NaN passes through); ``default`` on failure."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(cast(Any, value))  # tolerates numpy scalars, numeric strings
    except (TypeError, ValueError):
        return default


def _as_metric_value(value: object) -> float | None:
    """Return a finite ``float`` metric value, or ``None`` if absent/NaN/non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _as_int(value: object, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        try:  # tolerate "3.0" / numpy float seeds
            return int(float(cast(Any, value)))
        except (TypeError, ValueError):
            return default


def _as_bool(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _first(row: Mapping, *keys: str) -> object:
    """Return the first present, non-None value among ``keys`` (else ``None``)."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunResult:
    """One immutable evaluated run.

    Constructed by keyword only (``kw_only=True``) so field declaration order is
    free and every call site is self-documenting. ``metrics`` is defensively
    copied into a read-only :class:`~types.MappingProxyType` in ``__post_init__``
    so the record is immutable all the way down.
    """

    schema_version: int = 1
    # --- identity ----------------------------------------------------------
    #: Intended to equal :attr:`ProtocolSpec.protocol_version` (the frozen release
    #: identifier), whereas ``benchmark_id`` is the human-readable config name --
    #: keeping the store's partition/query key unambiguous.
    protocol_id: str
    benchmark_id: str
    dataset_id: str
    dataset_version: str = ""
    dataset_sha256: str = ""
    method_id: str
    method_version: str = ""
    hpo_mode: str
    split_id: str
    seed: int
    # --- status / eligibility ---------------------------------------------
    status: str
    comparison_ineligible: bool = False
    ineligible_reason: str = ""
    # --- cost / provenance -------------------------------------------------
    fit_seconds: float = float("nan")
    predict_seconds: float = float("nan")
    peak_rss_mb: float = float("nan")
    env_fingerprint: str = ""
    created_at: str = ""  # ISO-8601 UTC; filled with "now" in __post_init__ if empty
    # --- metrics -----------------------------------------------------------
    metrics: Mapping[str, float] = field(default_factory=dict)
    metric_provenance: Mapping[str, object] = field(default_factory=dict)

    #: Natural key. ``run_id`` is a deterministic hash of exactly these fields.
    #: Note ``benchmark_id`` is intentionally *not* part of the key: it is a label
    #: on top of the (protocol, dataset, method, hpo_mode, split, seed) unit.
    PRIMARY_KEY = ("protocol_id", "dataset_id", "method_id", "hpo_mode", "split_id", "seed")

    def __post_init__(self) -> None:
        # Single source of truth for "metric present": drop NaN/None-valued
        # entries here, using the SAME predicate the store applies on write, so a
        # directly-constructed RunResult's ``metrics`` (and ``to_row()``) match
        # the sparse set that is actually persisted/reloaded. ``from_fold_row``
        # already filters via ``_as_metric_value``; this covers direct callers.
        cleaned = {
            name: value
            for name, value in dict(self.metrics).items()
            if value is not None
            and not (isinstance(value, float) and math.isnan(value))
        }
        # Freeze the metrics mapping (defensive copy -> read-only view).
        object.__setattr__(self, "metrics", MappingProxyType(cleaned))
        object.__setattr__(self, "metric_provenance", MappingProxyType(dict(self.metric_provenance)))
        # Default created_at to "now" (UTC, ISO-8601) when not supplied.
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(timezone.utc).isoformat())

    # -- identity ----------------------------------------------------------
    @property
    def run_id(self) -> str:
        """Stable, deterministic row id = SHA-1 over the PRIMARY_KEY fields.

        Two ``RunResult`` objects with equal primary-key fields always yield the
        same ``run_id`` regardless of metrics/timing, so re-running a unit
        overwrites (never duplicates) its stored row.
        """
        key = "\x1f".join(str(getattr(self, name)) for name in self.PRIMARY_KEY)
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_fold_row(
        cls,
        row: Mapping,
        *,
        protocol_id: str,
        schema_version: int = 1,
    ) -> "RunResult":
        """Build a :class:`RunResult` from a runner-style flat ``record`` dict.

        Pulls the known scalar keys (see module docstring for the full mapping)
        and gathers the remaining numeric *metric* columns into ``metrics``.
        Tolerant of absent keys: anything missing falls back to a sensible
        default. ``protocol_id`` is required because the legacy runner does not
        emit it.
        """
        metrics: dict[str, float] = {}
        for key, value in row.items():
            name = str(key)
            if name in KNOWN_METRIC_COLUMNS or _is_dynamic_metric(name):
                coerced = _as_metric_value(value)
                if coerced is not None:
                    metrics[name] = coerced
        metric_provenance = {name: row[name] for name in _METRIC_PROVENANCE_COLUMNS if name in row}

        return cls(
            schema_version=schema_version,
            protocol_id=str(protocol_id),
            benchmark_id=str(_first(row, "benchmark_id") or ""),
            dataset_id=str(_first(row, "dataset_id") or ""),
            dataset_version=str(_first(row, "dataset_version") or ""),
            dataset_sha256=str(_first(row, "dataset_sha256") or ""),
            method_id=str(_first(row, "method_id") or ""),
            method_version=str(_first(row, "method_version") or ""),
            # export_coverage_matrix defaults a missing hpo_mode to "no_hpo".
            hpo_mode=str(_first(row, "hpo_mode") or "no_hpo"),
            split_id=str(_first(row, "split_id") or ""),
            seed=_as_int(_first(row, "seed"), default=0),
            # A missing status is treated as "unknown" -> ineligible (safe).
            status=str(_first(row, "status") or "unknown"),
            comparison_ineligible=_as_bool(row.get("comparison_ineligible")),
            ineligible_reason=str(_first(row, "ineligible_reason") or ""),
            fit_seconds=_as_float(_first(row, "fit_seconds", "fit_time_sec")),
            predict_seconds=_as_float(_first(row, "predict_seconds", "infer_time_sec")),
            peak_rss_mb=_as_float(_first(row, "peak_rss_mb", "peak_memory_mb")),
            env_fingerprint=str(_first(row, "env_fingerprint") or ""),
            created_at=str(_first(row, "created_at") or ""),
            metrics=metrics,
            metric_provenance=metric_provenance,
        )

    # -- serialisation -----------------------------------------------------
    def to_row(self) -> dict:
        """Flatten to a wide row dict (scalars + metrics expanded as columns).

        Mirror image of :meth:`from_fold_row`; a list of these dicts builds a
        wide DataFrame with the same one-row-per-run shape as the runner's
        ``fold_results``.
        """
        row: dict = {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "benchmark_id": self.benchmark_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "dataset_sha256": self.dataset_sha256,
            "method_id": self.method_id,
            "method_version": self.method_version,
            "hpo_mode": self.hpo_mode,
            "split_id": self.split_id,
            "seed": self.seed,
            "status": self.status,
            "comparison_ineligible": self.comparison_ineligible,
            "ineligible_reason": self.ineligible_reason,
            "fit_seconds": self.fit_seconds,
            "predict_seconds": self.predict_seconds,
            "peak_rss_mb": self.peak_rss_mb,
            "env_fingerprint": self.env_fingerprint,
            "created_at": self.created_at,
        }
        # Metric columns last; never collide with the scalar keys above.
        row.update(self.metrics)
        row.update(self.metric_provenance)
        return row
