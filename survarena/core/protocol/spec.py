"""Typed, versioned :class:`ProtocolSpec` for the survival benchmark.

This module is part of the strangler-fig refactor described in
``SURVARENA_REVAMP_PLAN.md`` (§3.4). It replaces the *benchmark-YAML sprawl*
(``configs/benchmark/*.yaml`` parsed as loose ``dict``s) with a single,
validated, immutable schema.

Why a typed spec at all
-----------------------
Today every ``manuscript_*_v1.yaml`` re-declares the same protocol geometry
(outer/inner folds, repeats, seeds), the same metric set, and the same
foundation-adapter overrides, with nothing enforcing that they agree or are
even internally coherent. A ``ProtocolSpec`` gives one place to parse, validate
and reason about a benchmark definition, so downstream code depends on typed
fields instead of digging through nested dictionaries.

Why versioned
-------------
``protocol_version`` starts the versioning contract from the plan: a protocol
version *freezes the geometry, seeds, metric set and eligibility rules* for a
benchmark release (``survbench-v1.0`` in the plan's language). A model is
evaluated *against* a frozen protocol version; changing geometry or metrics is a
new version. The field defaults to :data:`PROTOCOL_VERSION` and is tolerant of
older YAML that predates the field (it simply inherits the default).

Design notes
------------
* Everything is a ``@dataclass(frozen=True)`` and deeply immutable: sequences
  are stored as ``tuple``s and free-form nested mappings as read-only
  ``MappingProxyType`` views (see :func:`_freeze`). Nothing here imports the ML
  stack, keeping ``survarena.core`` importable anywhere.
* Parsing is *lenient by default*: unknown keys are preserved in ``raw_extra``
  rather than raising, and semantic problems are surfaced by :meth:`validate`.
  The strict path (``from_mapping(..., strict=True)`` / :meth:`require_valid`)
  raises :class:`ValueError` when any issue is present.
* Stable, cross-config keys are modelled as fields; genuinely open-ended,
  per-method or upstream-library payloads (e.g. AutoGluon
  ``hyperparameter_tune_kwargs`` or per-adapter ``default_params``) are kept as
  frozen mappings, because typing them now would be premature.

This module is *additive*: it does not change how the runner loads configs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

# The current protocol schema version. Bumping this is a benchmark-release
# event: it declares that the frozen geometry / metric set below is what a
# citation points at (see SURVARENA_REVAMP_PLAN.md §3.4 / §7.3).
PROTOCOL_VERSION = "survbench-0.1"

# Comparison arms recognised by v1. The plan's tuned+ensembled headline arrives
# in Phase 2; for now the schema knows the maintained ``no_hpo`` arm and the
# existing ``hpo`` arm.
ALLOWED_COMPARISON_MODES = frozenset({"no_hpo", "hpo"})

# The single-event right-censored task this benchmark is scoped to (plan §0).
EXPECTED_TASK_TYPE = "right_censored_survival"


def _empty_map() -> Mapping[str, Any]:
    return MappingProxyType({})


# Known top-level keys per (sub-)spec. Anything outside these sets is preserved
# in the matching ``raw_extra`` field instead of crashing the parser.
_PROTOCOL_KEYS = frozenset({
    "benchmark_id", "task_type", "split_strategy", "outer_folds", "outer_repeats",
    "inner_folds", "seeds", "primary_metric", "profile", "comparison_modes",
    "secondary_metrics", "autogluon", "hpo", "time_horizons_quantiles", "exports",
    "datasets", "methods", "notes", "protocol_version",
})
_AUTOGLUON_KEYS = frozenset({
    "enabled", "presets", "time_limit_seconds", "hyperparameter_tune_kwargs",
    "num_bag_folds", "num_stack_levels", "refit_full",
})
_HPO_KEYS = frozenset({
    "enabled", "max_trials", "timeout_seconds", "sampler", "pruner",
    "n_startup_trials", "method_overrides",
})
_METHOD_OVERRIDE_KEYS = frozenset({"enabled", "search_space", "default_params"})
_EXPORTS_KEYS = frozenset({"profile", "manuscript_artifact_layout"})


def _freeze(value: Any) -> Any:
    """Recursively convert a plain YAML value into an immutable equivalent.

    Mappings become read-only ``MappingProxyType`` views and lists/tuples become
    ``tuple``s; scalars (including ``None`` and ``str``) are returned unchanged.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Inverse of :func:`_freeze`: rebuild plain ``dict``/``list``/scalars.

    Used by ``to_mapping``/``to_yaml`` so serialisation never sees a
    ``MappingProxyType`` or ``tuple`` (which PyYAML's safe dumper rejects).
    """
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _as_tuple(value: Any) -> tuple[Any, ...]:
    """Coerce a YAML value into a tuple without exploding bare strings."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_subblock(value: Any, label: str, issues: list[str]) -> Mapping[str, Any] | None:
    """Return ``value`` unchanged if it is a mapping (or ``None``); otherwise
    record a lenient issue and coerce it to an empty mapping.

    Honors the "lenient by default" contract: a malformed sub-block (e.g.
    ``autogluon: "yes"``) must not crash :meth:`ProtocolSpec.from_mapping`. It is
    treated as empty and the problem is appended to ``issues`` so it surfaces via
    :meth:`ProtocolSpec.validate` (and, on the strict path, ``require_valid``).
    """
    if value is None or isinstance(value, Mapping):
        return value
    issues.append(
        f"{label}: expected a mapping sub-block, got {type(value).__name__}; "
        f"treated as empty (lenient parse)"
    )
    return {}


@dataclass(frozen=True)
class MethodOverride:
    """Per-method HPO override (``hpo.method_overrides[<method>]``).

    ``enabled`` and ``search_space`` are stable keys; ``default_params`` is a
    deliberately loose frozen mapping because its keys are adapter-specific
    (e.g. ``n_intervals``, ``max_stacked_rows``, ``time_limit``).
    """

    enabled: bool = False
    search_space: Any = None
    default_params: Mapping[str, Any] = field(default_factory=_empty_map)
    raw_extra: Mapping[str, Any] = field(default_factory=_empty_map)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> MethodOverride:
        data = dict(data or {})
        extra = {key: value for key, value in data.items() if key not in _METHOD_OVERRIDE_KEYS}
        return cls(
            enabled=bool(data.get("enabled", False)),
            search_space=_freeze(data.get("search_space")),
            default_params=_freeze(data.get("default_params") or {}),
            raw_extra=_freeze(extra),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "search_space": _thaw(self.search_space),
            "default_params": _thaw(self.default_params),
        }
        payload.update(_thaw(self.raw_extra))
        return payload


@dataclass(frozen=True)
class AutoGluonSpec:
    """AutoGluon-backed adapter settings (the ``autogluon`` block)."""

    enabled: bool = False
    presets: Any = None
    time_limit_seconds: int | float | None = None
    hyperparameter_tune_kwargs: Any = None
    num_bag_folds: int = 0
    num_stack_levels: int = 0
    refit_full: bool = False
    raw_extra: Mapping[str, Any] = field(default_factory=_empty_map)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> AutoGluonSpec:
        data = dict(data or {})
        extra = {key: value for key, value in data.items() if key not in _AUTOGLUON_KEYS}
        return cls(
            enabled=bool(data.get("enabled", False)),
            presets=_freeze(data.get("presets")),
            time_limit_seconds=data.get("time_limit_seconds"),
            hyperparameter_tune_kwargs=_freeze(data.get("hyperparameter_tune_kwargs")),
            num_bag_folds=data.get("num_bag_folds", 0),
            num_stack_levels=data.get("num_stack_levels", 0),
            refit_full=bool(data.get("refit_full", False)),
            raw_extra=_freeze(extra),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "presets": _thaw(self.presets),
            "time_limit_seconds": self.time_limit_seconds,
            "hyperparameter_tune_kwargs": _thaw(self.hyperparameter_tune_kwargs),
            "num_bag_folds": self.num_bag_folds,
            "num_stack_levels": self.num_stack_levels,
            "refit_full": self.refit_full,
        }
        payload.update(_thaw(self.raw_extra))
        return payload


@dataclass(frozen=True)
class HpoSpec:
    """Hyper-parameter optimisation settings (the ``hpo`` block)."""

    enabled: bool = False
    max_trials: int = 0
    timeout_seconds: int | float | None = None
    sampler: str = ""
    pruner: str = ""
    n_startup_trials: int = 0
    method_overrides: Mapping[str, MethodOverride] = field(default_factory=_empty_map)
    raw_extra: Mapping[str, Any] = field(default_factory=_empty_map)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> HpoSpec:
        data = dict(data or {})
        extra = {key: value for key, value in data.items() if key not in _HPO_KEYS}
        raw_overrides = data.get("method_overrides")
        overrides: dict[str, MethodOverride] = {}
        if isinstance(raw_overrides, Mapping):
            overrides = {
                str(name): MethodOverride.from_mapping(payload)
                for name, payload in raw_overrides.items()
            }
        return cls(
            enabled=bool(data.get("enabled", False)),
            max_trials=data.get("max_trials", 0),
            timeout_seconds=data.get("timeout_seconds"),
            sampler=data.get("sampler", ""),
            pruner=data.get("pruner", ""),
            n_startup_trials=data.get("n_startup_trials", 0),
            method_overrides=MappingProxyType(overrides),
            raw_extra=_freeze(extra),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": self.enabled,
            "max_trials": self.max_trials,
            "timeout_seconds": self.timeout_seconds,
            "sampler": self.sampler,
            "pruner": self.pruner,
            "n_startup_trials": self.n_startup_trials,
            "method_overrides": {
                name: override.to_mapping() for name, override in self.method_overrides.items()
            },
        }
        payload.update(_thaw(self.raw_extra))
        return payload


@dataclass(frozen=True)
class ExportsSpec:
    """Artifact export settings (the ``exports`` block)."""

    profile: str = ""
    manuscript_artifact_layout: str = ""
    raw_extra: Mapping[str, Any] = field(default_factory=_empty_map)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> ExportsSpec:
        data = dict(data or {})
        extra = {key: value for key, value in data.items() if key not in _EXPORTS_KEYS}
        return cls(
            profile=data.get("profile", ""),
            manuscript_artifact_layout=data.get("manuscript_artifact_layout", ""),
            raw_extra=_freeze(extra),
        )

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "profile": self.profile,
            "manuscript_artifact_layout": self.manuscript_artifact_layout,
        }
        payload.update(_thaw(self.raw_extra))
        return payload


@dataclass(frozen=True)
class ProtocolSpec:
    """A single, validated, versioned benchmark protocol.

    One ``ProtocolSpec`` captures everything a ``configs/benchmark/*.yaml`` file
    declares today: the protocol geometry (folds/repeats/seeds), the metric set,
    the comparison arm(s), the AutoGluon/HPO/export sub-blocks, and the dataset
    and method rosters. It replaces the ad-hoc YAML dictionaries with a typed,
    immutable object (see the module docstring for the versioning rationale).

    Construction is intentionally forgiving so that partial or slightly-off
    configs still load; call :meth:`validate` (or the strict parse path) to learn
    whether a spec is actually coherent.
    """

    benchmark_id: str = ""
    task_type: str = EXPECTED_TASK_TYPE
    split_strategy: str = "repeated_nested_cv"
    outer_folds: int = 5
    outer_repeats: int = 3
    inner_folds: int = 3
    seeds: tuple[int, ...] = ()
    primary_metric: str = ""
    profile: str = ""
    comparison_modes: tuple[str, ...] = ()
    secondary_metrics: tuple[str, ...] = ()
    autogluon: AutoGluonSpec = field(default_factory=AutoGluonSpec)
    hpo: HpoSpec = field(default_factory=HpoSpec)
    time_horizons_quantiles: tuple[float, ...] = ()
    exports: ExportsSpec = field(default_factory=ExportsSpec)
    datasets: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    notes: str = ""
    protocol_version: str = PROTOCOL_VERSION
    raw_extra: Mapping[str, Any] = field(default_factory=_empty_map)
    #: Human-readable problems detected while *parsing* the mapping -- e.g. a
    #: malformed sub-block that was leniently coerced to empty (fix F). Surfaced
    #: by :meth:`validate` so ``require_valid``/strict parse reject them; it is
    #: intentionally NOT serialised by :meth:`to_mapping`, and stays empty for a
    #: well-formed mapping (so valid specs still round-trip and compare equal).
    parse_issues: tuple[str, ...] = ()

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, strict: bool = False) -> ProtocolSpec:
        """Build a spec from a plain mapping (e.g. ``yaml.safe_load`` output).

        Unknown top-level keys are preserved in ``raw_extra`` and reported as a
        warning by :meth:`validate`. Malformed sub-blocks (a non-mapping
        ``autogluon``/``hpo``/``exports``/``method_overrides`` value) do not raise
        on this lenient path: they are coerced to empty and recorded in
        ``parse_issues`` (which :meth:`validate` surfaces). When ``strict=True``
        any validation issue -- including those coercions -- raises
        :class:`ValueError`.
        """
        if not isinstance(data, Mapping):
            raise TypeError(
                f"ProtocolSpec.from_mapping expects a mapping, got {type(data).__name__}"
            )
        data = dict(data)
        extra = {key: value for key, value in data.items() if key not in _PROTOCOL_KEYS}

        # Lenient handling of malformed sub-blocks (fix F): a non-mapping
        # ``autogluon``/``hpo``/``exports`` value (or ``method_overrides``, or an
        # individual method override) must NOT raise here -- coerce it to empty
        # and record an issue that ``validate``/``require_valid`` surface.
        parse_issues: list[str] = []
        autogluon_data = _coerce_subblock(data.get("autogluon"), "autogluon", parse_issues)
        hpo_data = _coerce_subblock(data.get("hpo"), "hpo", parse_issues)
        exports_data = _coerce_subblock(data.get("exports"), "exports", parse_issues)

        # Nested: ``hpo.method_overrides`` (a mapping of method -> override
        # mapping). Scrub non-mapping overrides before HpoSpec.from_mapping so it
        # never chokes; report each coercion.
        if isinstance(hpo_data, Mapping):
            raw_overrides = hpo_data.get("method_overrides")
            if raw_overrides is not None and not isinstance(raw_overrides, Mapping):
                parse_issues.append(
                    "hpo.method_overrides: expected a mapping, got "
                    f"{type(raw_overrides).__name__}; treated as empty (lenient parse)"
                )
                hpo_data = {**hpo_data, "method_overrides": {}}
            elif isinstance(raw_overrides, Mapping):
                scrubbed: dict[str, Any] = {}
                changed = False
                for name, payload in raw_overrides.items():
                    if payload is None or isinstance(payload, Mapping):
                        scrubbed[name] = payload
                    else:
                        parse_issues.append(
                            f"hpo.method_overrides.{name}: expected a mapping, got "
                            f"{type(payload).__name__}; treated as empty (lenient parse)"
                        )
                        scrubbed[name] = {}
                        changed = True
                if changed:
                    hpo_data = {**hpo_data, "method_overrides": scrubbed}

        spec = cls(
            benchmark_id=data.get("benchmark_id", ""),
            task_type=data.get("task_type", EXPECTED_TASK_TYPE),
            split_strategy=data.get("split_strategy", "repeated_nested_cv"),
            outer_folds=data.get("outer_folds", 5),
            outer_repeats=data.get("outer_repeats", 3),
            inner_folds=data.get("inner_folds", 3),
            seeds=_as_tuple(data.get("seeds")),
            primary_metric=data.get("primary_metric", ""),
            profile=data.get("profile", ""),
            comparison_modes=_as_tuple(data.get("comparison_modes")),
            secondary_metrics=_as_tuple(data.get("secondary_metrics")),
            autogluon=AutoGluonSpec.from_mapping(autogluon_data),
            hpo=HpoSpec.from_mapping(hpo_data),
            time_horizons_quantiles=_as_tuple(data.get("time_horizons_quantiles")),
            exports=ExportsSpec.from_mapping(exports_data),
            datasets=_as_tuple(data.get("datasets")),
            methods=_as_tuple(data.get("methods")),
            notes=data.get("notes", ""),
            protocol_version=data.get("protocol_version", PROTOCOL_VERSION),
            raw_extra=_freeze(extra),
            parse_issues=tuple(parse_issues),
        )
        if strict:
            spec.require_valid()
        return spec

    @classmethod
    def from_yaml(cls, path: str | Any, *, strict: bool = False) -> ProtocolSpec:
        """Load a spec from a YAML file via ``yaml.safe_load``."""
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_mapping(data or {}, strict=strict)

    # -- serialisation --------------------------------------------------------
    def to_mapping(self) -> dict[str, Any]:
        """Render back to a plain, YAML-safe ``dict`` (round-trips typed fields)."""
        payload: dict[str, Any] = {
            "benchmark_id": self.benchmark_id,
            "task_type": self.task_type,
            "split_strategy": self.split_strategy,
            "outer_folds": self.outer_folds,
            "outer_repeats": self.outer_repeats,
            "inner_folds": self.inner_folds,
            "seeds": list(self.seeds),
            "primary_metric": self.primary_metric,
            "profile": self.profile,
            "comparison_modes": list(self.comparison_modes),
            "secondary_metrics": list(self.secondary_metrics),
            "autogluon": self.autogluon.to_mapping(),
            "hpo": self.hpo.to_mapping(),
            "time_horizons_quantiles": list(self.time_horizons_quantiles),
            "exports": self.exports.to_mapping(),
            "datasets": list(self.datasets),
            "methods": list(self.methods),
            "notes": self.notes,
            "protocol_version": self.protocol_version,
        }
        payload.update(_thaw(self.raw_extra))
        return payload

    def to_yaml(self, path: str | Any) -> None:
        """Write the spec to a YAML file via ``yaml.safe_dump``."""
        import yaml

        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.to_mapping(),
                handle,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

    # -- validation -----------------------------------------------------------
    def validate(self) -> list[str]:
        """Return human-readable issues; an empty list means the spec is valid.

        Includes both hard errors (empty rosters, degenerate fold counts, bad
        comparison modes, out-of-range or unsorted quantiles) and warnings (a
        non-single-event ``task_type``, or preserved unknown keys).
        """
        issues: list[str] = []

        if not self.datasets:
            issues.append("datasets: must be a non-empty list of dataset ids")
        if not self.methods:
            issues.append("methods: must be a non-empty list of method ids")

        if not _is_int(self.outer_folds):
            issues.append(f"outer_folds: must be an integer (got {self.outer_folds!r})")
        elif self.outer_folds < 2:
            issues.append(f"outer_folds: must be >= 2 (got {self.outer_folds})")

        if not _is_int(self.outer_repeats):
            issues.append(f"outer_repeats: must be an integer (got {self.outer_repeats!r})")
        elif self.outer_repeats < 1:
            issues.append(f"outer_repeats: must be >= 1 (got {self.outer_repeats})")

        if not _is_int(self.inner_folds):
            issues.append(f"inner_folds: must be an integer (got {self.inner_folds!r})")
        elif self.inner_folds < 2:
            issues.append(f"inner_folds: must be >= 2 (got {self.inner_folds})")

        if not self.seeds:
            issues.append("seeds: must be a non-empty list of integers")
        else:
            bad_seeds = [seed for seed in self.seeds if not _is_int(seed)]
            if bad_seeds:
                issues.append(f"seeds: all entries must be integers (offending: {bad_seeds})")

        if not (isinstance(self.primary_metric, str) and self.primary_metric.strip()):
            issues.append("primary_metric: must be a non-empty string")

        if not self.comparison_modes:
            issues.append("comparison_modes: must be a non-empty list")
        else:
            unknown_modes = [m for m in self.comparison_modes if m not in ALLOWED_COMPARISON_MODES]
            if unknown_modes:
                allowed = sorted(ALLOWED_COMPARISON_MODES)
                issues.append(
                    f"comparison_modes: unknown modes {unknown_modes}; allowed={allowed}"
                )

        quantiles = self.time_horizons_quantiles
        if quantiles:
            if any((not _is_number(q)) or not (0.0 < q < 1.0) for q in quantiles):
                issues.append(
                    "time_horizons_quantiles: all values must be in the open interval (0, 1) "
                    f"(got {list(quantiles)})"
                )
            elif list(quantiles) != sorted(set(quantiles)):
                issues.append(
                    "time_horizons_quantiles: must be sorted and unique "
                    f"(got {list(quantiles)})"
                )

        # Warning (still an "issue"): v1 scope is single-event right-censored.
        if self.task_type != EXPECTED_TASK_TYPE:
            issues.append(
                f"task_type: expected {EXPECTED_TASK_TYPE!r} for the v1 single-event scope "
                f"(got {self.task_type!r})"
            )

        # Warning: unknown top-level keys were preserved but not schema-checked.
        if self.raw_extra:
            issues.append(
                "raw_extra: unrecognized top-level keys preserved but not validated: "
                f"{sorted(self.raw_extra)}"
            )

        # Parse-time problems (e.g. malformed sub-blocks leniently coerced to
        # empty under the "lenient by default" contract) are surfaced here so the
        # strict path (require_valid) rejects them too (fix F).
        issues.extend(self.parse_issues)

        return issues

    def is_valid(self) -> bool:
        return not self.validate()

    def require_valid(self) -> ProtocolSpec:
        """Raise :class:`ValueError` if :meth:`validate` reports any issue."""
        issues = self.validate()
        if issues:
            joined = "\n  - ".join(issues)
            label = self.benchmark_id or "<unnamed>"
            raise ValueError(f"Invalid ProtocolSpec ({label}):\n  - {joined}")
        return self
