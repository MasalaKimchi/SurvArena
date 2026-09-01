"""Standalone tests for ``survarena.core.protocol`` (no pytest required).

Run with either::

    PYTHONPATH=. python3 tests/test_core_protocol.py

or simply ``python3 tests/test_core_protocol.py`` (the repo root is added to
``sys.path`` below). Exits non-zero if any check fails.

These tests deliberately avoid ``import pytest`` / parametrize / fixtures so the
new ``core`` kernel stays verifiable on a bare stdlib+PyYAML environment.
"""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from survarena.core.protocol import (  # noqa: E402  (after sys.path shim)
    PROTOCOL_VERSION,
    AutoGluonSpec,
    ExportsSpec,
    HpoSpec,
    MethodOverride,
    ProtocolSpec,
)

_BENCH_DIR = os.path.join(_ROOT, "configs", "benchmark")
_MANUSCRIPT_V1 = os.path.join(_BENCH_DIR, "manuscript_v1.yaml")

_PASS = 0
_FAIL = 0


def _ok(message: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  PASS  {message}")


def _bad(message: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"  FAIL  {message}")


def _check(condition: bool, message: str) -> None:
    _ok(message) if condition else _bad(message)


def _expect(exc, fn, *args, **kwargs) -> None:
    """Assert that ``fn(*args, **kwargs)`` raises ``exc``."""
    try:
        result = fn(*args, **kwargs)
    except exc:
        _ok(f"{getattr(fn, '__name__', fn)} raised {exc.__name__} as expected")
    except BaseException as unexpected:  # noqa: BLE001 - report wrong exception type
        _bad(
            f"{getattr(fn, '__name__', fn)} raised "
            f"{type(unexpected).__name__} ({unexpected}); expected {exc.__name__}"
        )
    else:
        _bad(
            f"{getattr(fn, '__name__', fn)} returned {result!r}; expected {exc.__name__}"
        )


def test_real_yaml_load() -> None:
    print("test_real_yaml_load")
    spec = ProtocolSpec.from_yaml(_MANUSCRIPT_V1)

    issues = spec.validate()
    _check(issues == [], f"manuscript_v1 validate() == [] (got {issues})")

    _check(spec.benchmark_id == "manuscript_v1", "benchmark_id == 'manuscript_v1'")
    _check(spec.task_type == "right_censored_survival", "task_type == 'right_censored_survival'")
    _check(spec.split_strategy == "repeated_nested_cv", "split_strategy == 'repeated_nested_cv'")
    _check(spec.outer_folds == 5, "outer_folds == 5")
    _check(spec.outer_repeats == 3, "outer_repeats == 3")
    _check(spec.inner_folds == 3, "inner_folds == 3")
    _check(list(spec.seeds) == [11, 22, 33, 44, 55], "seeds == [11, 22, 33, 44, 55]")
    _check(spec.primary_metric == "uno_c", "primary_metric == 'uno_c'")
    _check(spec.profile == "manuscript", "profile == 'manuscript'")
    _check(list(spec.comparison_modes) == ["no_hpo"], "comparison_modes == ['no_hpo']")
    _check(len(spec.datasets) == 7, f"len(datasets) == 7 (got {len(spec.datasets)})")
    _check(len(spec.methods) == 27, f"len(methods) == 27 (got {len(spec.methods)})")
    _check(
        list(spec.time_horizons_quantiles) == [0.25, 0.5, 0.75],
        "time_horizons_quantiles == [0.25, 0.5, 0.75]",
    )

    # protocol_version defaults even though the YAML predates the field.
    _check(spec.protocol_version == PROTOCOL_VERSION, f"protocol_version == {PROTOCOL_VERSION!r}")
    _check(spec.raw_extra == {}, "raw_extra empty for a fully-known config")

    # Typed sub-specs.
    _check(isinstance(spec.autogluon, AutoGluonSpec), "autogluon is AutoGluonSpec")
    _check(spec.autogluon.enabled is True, "autogluon.enabled is True")
    _check(spec.autogluon.time_limit_seconds == 120, "autogluon.time_limit_seconds == 120")
    _check(spec.autogluon.presets is None, "autogluon.presets is None")
    _check(isinstance(spec.hpo, HpoSpec), "hpo is HpoSpec")
    _check(spec.hpo.enabled is False, "hpo.enabled is False")
    _check(spec.hpo.sampler == "random", "hpo.sampler == 'random'")
    _check(
        "tabpfn_survival" in spec.hpo.method_overrides,
        "hpo.method_overrides contains 'tabpfn_survival'",
    )
    override = spec.hpo.method_overrides.get("tabpfn_survival")
    _check(isinstance(override, MethodOverride), "method override is a MethodOverride")
    _check(
        override is not None and override.default_params.get("device") == "cpu",
        "tabpfn_survival default_params.device == 'cpu'",
    )
    _check(isinstance(spec.exports, ExportsSpec), "exports is ExportsSpec")
    _check(spec.exports.profile == "core_csv", "exports.profile == 'core_csv'")


def test_round_trip() -> None:
    print("test_round_trip")
    spec = ProtocolSpec.from_yaml(_MANUSCRIPT_V1)

    # In-memory round trip must be exact on every typed field (deep equality).
    rebuilt = ProtocolSpec.from_mapping(spec.to_mapping())
    _check(rebuilt == spec, "from_mapping(to_mapping(spec)) == spec")
    _check(rebuilt.autogluon == spec.autogluon, "autogluon sub-spec round-trips")
    _check(rebuilt.hpo == spec.hpo, "hpo sub-spec (with method_overrides) round-trips")
    _check(rebuilt.exports == spec.exports, "exports sub-spec round-trips")

    # YAML file round trip: structural fields survive (notes may re-fold, so we
    # assert on the typed geometry/roster rather than byte-identical notes).
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "roundtrip.yaml")
        spec.to_yaml(out_path)
        reloaded = ProtocolSpec.from_yaml(out_path)
        _check(reloaded.validate() == [], "reloaded YAML validate() == []")
        _check(reloaded.benchmark_id == spec.benchmark_id, "YAML round trip: benchmark_id")
        _check(reloaded.outer_folds == spec.outer_folds, "YAML round trip: outer_folds")
        _check(reloaded.seeds == spec.seeds, "YAML round trip: seeds")
        _check(reloaded.datasets == spec.datasets, "YAML round trip: datasets")
        _check(reloaded.methods == spec.methods, "YAML round trip: methods")
        _check(reloaded.hpo.method_overrides.keys() == spec.hpo.method_overrides.keys(),
               "YAML round trip: method_overrides keys")


def test_immutability() -> None:
    print("test_immutability")
    spec = ProtocolSpec.from_yaml(_MANUSCRIPT_V1)

    def _mutate_field() -> None:
        spec.outer_folds = 99  # type: ignore[misc]

    _expect(Exception, _mutate_field)

    def _mutate_nested() -> None:
        spec.hpo.method_overrides["tabpfn_survival"].default_params["device"] = "gpu"

    _expect(Exception, _mutate_nested)


def test_unknown_keys_preserved() -> None:
    print("test_unknown_keys_preserved")
    payload = ProtocolSpec.from_yaml(_MANUSCRIPT_V1).to_mapping()
    payload["experimental_flag"] = True
    payload["extends"] = "base.yaml"

    spec = ProtocolSpec.from_mapping(payload)  # lenient: must not raise
    _check("experimental_flag" in spec.raw_extra, "unknown key kept in raw_extra")
    _check("extends" in spec.raw_extra, "second unknown key kept in raw_extra")

    issues = spec.validate()
    _check(any("raw_extra" in issue for issue in issues), "validate() warns about unknown keys")

    # Even with unknown keys, the typed fields still round-trip (extras re-emit).
    rebuilt = ProtocolSpec.from_mapping(spec.to_mapping())
    _check(rebuilt == spec, "spec with raw_extra round-trips")

    # Strict parse rejects the unknown keys.
    _expect(ValueError, ProtocolSpec.from_mapping, payload, strict=True)


def test_invalid_configs() -> None:
    print("test_invalid_configs")
    base = ProtocolSpec.from_yaml(_MANUSCRIPT_V1).to_mapping()

    def _with(**overrides) -> dict:
        data = dict(base)
        data.update(overrides)
        return data

    # 1. Empty datasets.
    empty_datasets = ProtocolSpec.from_mapping(_with(datasets=[]))
    _check(
        any("datasets" in issue for issue in empty_datasets.validate()),
        "empty datasets -> validate() issue",
    )
    _expect(ValueError, ProtocolSpec.from_mapping, _with(datasets=[]), strict=True)

    # 2. Empty methods.
    _expect(ValueError, ProtocolSpec.from_mapping, _with(methods=[]), strict=True)

    # 3. Degenerate outer_folds.
    one_fold = ProtocolSpec.from_mapping(_with(outer_folds=1))
    _check(
        any("outer_folds" in issue for issue in one_fold.validate()),
        "outer_folds=1 -> validate() issue",
    )
    _expect(ValueError, ProtocolSpec.from_mapping, _with(outer_folds=1), strict=True)

    # 4. inner_folds too small.
    _expect(ValueError, ProtocolSpec.from_mapping, _with(inner_folds=1), strict=True)

    # 5. Bad comparison mode.
    bad_mode = ProtocolSpec.from_mapping(_with(comparison_modes=["turbo"]))
    _check(
        any("comparison_modes" in issue for issue in bad_mode.validate()),
        "comparison_modes=['turbo'] -> validate() issue",
    )
    _expect(ValueError, ProtocolSpec.from_mapping, _with(comparison_modes=["turbo"]), strict=True)

    # 6. Unsorted quantiles.
    unsorted_q = ProtocolSpec.from_mapping(_with(time_horizons_quantiles=[0.75, 0.25, 0.5]))
    _check(
        any("time_horizons_quantiles" in issue for issue in unsorted_q.validate()),
        "unsorted quantiles -> validate() issue",
    )
    _expect(
        ValueError,
        ProtocolSpec.from_mapping,
        _with(time_horizons_quantiles=[0.75, 0.25, 0.5]),
        strict=True,
    )

    # 7. Out-of-range quantiles.
    _expect(
        ValueError,
        ProtocolSpec.from_mapping,
        _with(time_horizons_quantiles=[0.25, 0.5, 1.5]),
        strict=True,
    )

    # 8. Duplicate (non-unique) quantiles.
    _expect(
        ValueError,
        ProtocolSpec.from_mapping,
        _with(time_horizons_quantiles=[0.25, 0.25, 0.5]),
        strict=True,
    )

    # 9. Empty / non-int seeds.
    _expect(ValueError, ProtocolSpec.from_mapping, _with(seeds=[]), strict=True)
    non_int = ProtocolSpec.from_mapping(_with(seeds=[11, "22", 33]))
    _check(
        any("seeds" in issue for issue in non_int.validate()),
        "non-int seeds -> validate() issue",
    )

    # 10. Empty primary_metric.
    _expect(ValueError, ProtocolSpec.from_mapping, _with(primary_metric=""), strict=True)

    # 11. Non-mapping input to from_mapping.
    _expect(TypeError, ProtocolSpec.from_mapping, ["not", "a", "mapping"])


def test_task_type_warning() -> None:
    print("test_task_type_warning")
    base = ProtocolSpec.from_yaml(_MANUSCRIPT_V1).to_mapping()
    base["task_type"] = "competing_risks"
    spec = ProtocolSpec.from_mapping(base)
    _check(
        any("task_type" in issue for issue in spec.validate()),
        "non-single-event task_type -> validate() warning",
    )


def test_defaults_and_generality() -> None:
    print("test_defaults_and_generality")
    # Default construction is coherent (empty rosters aside) and versioned.
    blank = ProtocolSpec()
    _check(blank.protocol_version == PROTOCOL_VERSION, "default protocol_version set")
    _check(not blank.is_valid(), "empty ProtocolSpec is not valid (needs datasets/methods)")

    # Generality: every real benchmark YAML present must parse and validate clean.
    if os.path.isdir(_BENCH_DIR):
        yamls = sorted(f for f in os.listdir(_BENCH_DIR) if f.endswith((".yaml", ".yml")))
        _check(len(yamls) >= 1, f"found {len(yamls)} benchmark config(s) to check")
        for name in yamls:
            path = os.path.join(_BENCH_DIR, name)
            spec = ProtocolSpec.from_yaml(path)
            issues = spec.validate()
            # Every real config must be structurally sound. A couple of
            # experimental configs legitimately carry an extra top-level key
            # (e.g. ``validation_diagnostics``), which the schema preserves in
            # ``raw_extra`` and reports as an advisory warning -- that is
            # expected, so only *non*-raw_extra issues are treated as failures.
            hard_errors = [issue for issue in issues if not issue.startswith("raw_extra:")]
            _check(hard_errors == [], f"{name}: no structural errors (got {hard_errors})")
            rebuilt = ProtocolSpec.from_mapping(spec.to_mapping())
            _check(rebuilt == spec, f"{name}: from_mapping(to_mapping(spec)) == spec")


def test_malformed_subblocks_are_lenient() -> None:
    print("test_malformed_subblocks_are_lenient")
    base = ProtocolSpec.from_yaml(_MANUSCRIPT_V1).to_mapping()

    def _with(**overrides) -> dict:
        data = dict(base)
        data.update(overrides)
        return data

    # 1. Non-mapping autogluon / hpo / exports must NOT raise on the lenient path.
    bad_blocks = _with(autogluon="yes", hpo=123, exports=[1, 2, 3])
    spec = ProtocolSpec.from_mapping(bad_blocks)  # must not raise
    _ok("malformed autogluon/hpo/exports parsed leniently (no raise)")
    # Malformed sub-blocks fell back to defaults instead of crashing.
    _check(
        isinstance(spec.autogluon, AutoGluonSpec) and spec.autogluon.enabled is False,
        "malformed autogluon coerced to default AutoGluonSpec",
    )
    _check(
        isinstance(spec.hpo, HpoSpec) and spec.hpo.max_trials == 0,
        "malformed hpo coerced to default HpoSpec",
    )
    _check(
        isinstance(spec.exports, ExportsSpec) and spec.exports.profile == "",
        "malformed exports coerced to default ExportsSpec",
    )
    # validate() surfaces each malformed sub-block.
    issues = spec.validate()
    _check(any("autogluon" in issue for issue in issues), "validate() reports malformed autogluon")
    _check(any(issue.startswith("hpo:") for issue in issues), "validate() reports malformed hpo")
    _check(any("exports" in issue for issue in issues), "validate() reports malformed exports")
    # Strict parse rejects the coercions.
    _expect(ValueError, ProtocolSpec.from_mapping, bad_blocks, strict=True)

    # 2. Nested: an individual non-mapping method override (method_overrides.foo).
    nested = _with(hpo={"enabled": False, "method_overrides": {"foo": "str"}})
    spec2 = ProtocolSpec.from_mapping(nested)  # must not raise
    _ok("malformed method override parsed leniently (no raise)")
    _check(
        isinstance(spec2.hpo.method_overrides.get("foo"), MethodOverride),
        "malformed method override coerced to a default MethodOverride",
    )
    _check(
        any("method_overrides.foo" in issue for issue in spec2.validate()),
        "validate() reports malformed method override",
    )
    _expect(ValueError, ProtocolSpec.from_mapping, nested, strict=True)

    # 3. Nested: a non-mapping method_overrides block as a whole.
    nested_block = _with(hpo={"enabled": True, "method_overrides": "not-a-map"})
    spec3 = ProtocolSpec.from_mapping(nested_block)  # must not raise
    _check(dict(spec3.hpo.method_overrides) == {}, "non-mapping method_overrides -> empty mapping")
    _check(
        any("method_overrides" in issue for issue in spec3.validate()),
        "validate() reports non-mapping method_overrides block",
    )
    _expect(ValueError, ProtocolSpec.from_mapping, nested_block, strict=True)


_TESTS = [
    test_real_yaml_load,
    test_round_trip,
    test_immutability,
    test_unknown_keys_preserved,
    test_invalid_configs,
    test_task_type_warning,
    test_defaults_and_generality,
    test_malformed_subblocks_are_lenient,
]


def main() -> int:
    for test in _TESTS:
        test()
    print(f"\n{_PASS} passed, {_FAIL} failed")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
