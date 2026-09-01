"""Tests for the Phase-1a model capability contract.

This file is written to run BOTH under pytest and as a plain script (there is no
pytest available in every environment). It uses only plain ``assert`` statements
and a local ``_expect`` helper for expected-exception cases -- no ``import
pytest``, no parametrize. The ``__main__`` block runs every ``test_*`` function,
prints PASS/FAIL per test and a summary, and exits non-zero on any failure.
"""

from __future__ import annotations

import dataclasses

from survarena.core.models import ModelCapabilities, SurvivalModel
from survarena.core.models import contract as _contract
from survarena.methods.base import BaseSurvivalMethod

# The six capability dimensions, in declaration order.
_EXPECTED_FIELDS = (
    "uses_validation",
    "supports_early_stopping",
    "refit_on_full_train",
    "native_categoricals",
    "supports_gpu",
    "deterministic_given_seed",
)


def _expect(exc, fn, *a, **k):
    """Assert that ``fn(*a, **k)`` raises ``exc``; return the raised exception."""
    try:
        fn(*a, **k)
    except exc as e:  # noqa: B902 - explicit expected-exception assertion
        return e
    name = getattr(fn, "__name__", repr(fn))
    raise AssertionError(f"expected {exc!r} from {name}, but nothing was raised")


# --- concrete throwaway adapters (BaseSurvivalMethod is abstract) -------------


class _DummyBase(BaseSurvivalMethod):
    """Minimal concrete adapter: inherits every capability default."""

    def fit(self, X_train, time_train, event_train, X_val=None, time_val=None, event_val=None):
        return self

    def predict_risk(self, X):
        return X

    def predict_survival(self, X, times):
        return X


class _DummyVal(_DummyBase):
    consumes_validation = True


class _DummyCat(_DummyBase):
    native_categoricals = True


# --- ModelCapabilities dataclass ---------------------------------------------


def test_modelcapabilities_has_six_fields():
    names = tuple(f.name for f in dataclasses.fields(ModelCapabilities))
    assert names == _EXPECTED_FIELDS, names
    assert len(names) == 6


def test_modelcapabilities_defaults():
    cap = ModelCapabilities(uses_validation=False)
    assert cap.uses_validation is False
    assert cap.supports_early_stopping is False
    assert cap.refit_on_full_train is False
    assert cap.native_categoricals is False
    assert cap.supports_gpu is False
    # Conservative default: assume reproducibility.
    assert cap.deterministic_given_seed is True


def test_modelcapabilities_uses_validation_is_required():
    # `uses_validation` has no default -- constructing without it must fail.
    _expect(TypeError, ModelCapabilities)


def test_modelcapabilities_is_frozen():
    cap = ModelCapabilities(uses_validation=True)
    # frozen=True -> assignment to an existing field raises FrozenInstanceError.
    _expect(dataclasses.FrozenInstanceError, setattr, cap, "uses_validation", False)
    # value is unchanged after the rejected assignment
    assert cap.uses_validation is True


def test_modelcapabilities_equality_and_immutability_semantics():
    a = ModelCapabilities(uses_validation=True)
    b = ModelCapabilities(uses_validation=True)
    # frozen dataclasses get value equality and are hashable.
    assert a == b
    assert hash(a) == hash(b)
    assert a != ModelCapabilities(uses_validation=False)


# --- BaseSurvivalMethod.capabilities() ---------------------------------------


def test_base_default_capabilities():
    # Callable on the abstract base itself (classmethod, no instantiation).
    cap = BaseSurvivalMethod.capabilities()
    assert isinstance(cap, ModelCapabilities)
    assert cap.uses_validation is False
    assert cap.deterministic_given_seed is True
    # `uses_validation` is sourced from the Phase-0 flag.
    assert cap.uses_validation == BaseSurvivalMethod.consumes_validation


def test_subclass_consumes_validation_propagates():
    # As a classmethod on the subclass...
    assert _DummyVal.capabilities().uses_validation is True
    # ...and via an instance.
    assert _DummyVal().capabilities().uses_validation is True
    # A sibling that leaves the default keeps uses_validation False.
    assert _DummyBase.capabilities().uses_validation is False


def test_subclass_native_categoricals_propagates():
    cap = _DummyCat.capabilities()
    assert cap.native_categoricals is True
    # Overriding one dimension leaves the others at their conservative defaults.
    assert cap.uses_validation is False
    assert cap.supports_gpu is False
    assert cap.deterministic_given_seed is True


# --- package wiring / Protocol ------------------------------------------------


def test_package_reexports_are_identical():
    # `core.models` must re-export the exact objects from `core.models.contract`.
    assert ModelCapabilities is _contract.ModelCapabilities
    assert SurvivalModel is _contract.SurvivalModel


def test_survivalmodel_is_runtime_checkable_name_level():
    # runtime_checkable only verifies attribute/method *presence* (not signatures);
    # this documents that the Phase-1a adapters already expose the right names.
    dummy = _DummyVal()
    assert isinstance(dummy, SurvivalModel)


if __name__ == "__main__":
    import sys
    import traceback

    tests = sorted(
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    )
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:  # noqa: BLE001 - standalone harness reports every failure
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {name}")

    print(f"\n{passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(1 if failed else 0)
