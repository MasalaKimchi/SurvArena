"""Regression guard for the Elo-bootstrap `present` hoist (perf optimization).

Runs WITHOUT pytest: plain ``assert``\\ s and a ``__main__`` harness that prints
PASS/FAIL per test and exits non-zero on any failure.

    PYTHONPATH=. python3 tests/test_ratings_bootstrap_identity.py

The optimization precomputes each cluster's method-id set once and derives each
bootstrap draw's ``present`` list as a UNION over the drawn clusters, instead of
rescanning the concatenated ``boot_matches`` every iteration. This module locks
in the invariant that makes that rewrite output-preserving:

    sorted(set().union(*(cluster_to_methods[c] for c in drawn)))
        == sorted({m for left, right, _ in boot_matches for m in (left, right)})

for every possible bootstrap draw, plus the downstream properties (determinism,
point-estimate invariance to n_bootstrap) that the change must not disturb.
"""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd

from survarena.evaluation._ratings import _clustered_match_lists, elo_ratings


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def _ragged_frame() -> pd.DataFrame:
    """Datasets covering DIFFERENT method subsets so a draw's ``present`` varies.

    ds0: a,b,c   ds1: a,b   ds2: b,c   ds3: c only (a NO-match cluster -> its
    method set is a singleton but it contributes zero matches, exercising the
    empty-``boot_matches`` skip when a draw picks only ds3).
    """
    coverage = {"ds0": ["a", "b", "c"], "ds1": ["a", "b"], "ds2": ["b", "c"], "ds3": ["c"]}
    skill = {"a": 0.70, "b": 0.74, "c": 0.66}
    rng = np.random.default_rng(99)
    rows = []
    for d, methods in coverage.items():
        for s in range(2):
            for m in methods:
                rows.append(
                    {
                        "benchmark_id": "benchR",
                        "dataset_id": d,
                        "split_id": f"fold{s}",
                        "seed": s,
                        "method_id": m,
                        "uno_c": float(np.clip(skill[m] + rng.normal(0, 0.04), 0.0, 1.0)),
                    }
                )
    return pd.DataFrame(rows)


def _full_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2024)
    skill = {"cox": 0.70, "rsf": 0.74, "deepsurv": 0.66}
    rows = []
    for benchmark_id in ("benchA", "benchB"):
        for d in range(3):
            for s in range(3):
                for m, sk in skill.items():
                    rows.append(
                        {
                            "benchmark_id": benchmark_id,
                            "dataset_id": f"ds{d}",
                            "split_id": f"fold{s}",
                            "seed": s,
                            "method_id": m,
                            "uno_c": float(np.clip(sk + rng.normal(0, 0.05), 0.0, 1.0)),
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_present_union_equals_rescan_over_all_draws():
    """The hoisted union ``present`` equals the old per-draw rescan, every draw.

    This is the exact equivalence the perf rewrite relies on: because
    ``boot_matches`` is the concatenation of the drawn clusters' match lists, the
    set of method ids in ``boot_matches`` equals the union of those clusters'
    precomputed method sets.
    """
    frame = _ragged_frame()
    cluster_to_matches, _methods = _clustered_match_lists(frame, metric="uno_c")
    cluster_to_methods = {
        cluster: {m for a, b, _ in matches for m in (a, b)}
        for cluster, matches in cluster_to_matches.items()
    }
    clusters = sorted(cluster_to_matches)
    rng = np.random.default_rng(0)

    distinct_present: set[tuple[str, ...]] = set()
    for _ in range(5000):
        idx = rng.integers(0, len(clusters), size=len(clusters))
        boot_matches: list[tuple[str, str, float]] = []
        for j in idx:
            boot_matches.extend(cluster_to_matches[clusters[int(j)]])
        rescan = sorted({m for left, right, _ in boot_matches for m in (left, right)})
        union = sorted(set().union(*(cluster_to_methods[clusters[int(j)]] for j in idx)))
        assert union == rescan, (idx.tolist(), union, rescan)
        distinct_present.add(tuple(rescan))

    # The guard is only meaningful if `present` actually varies across draws;
    # a ragged frame must yield more than one distinct present-set (and include
    # the empty set from an all-ds3 draw).
    assert len(distinct_present) > 1, distinct_present
    assert () in distinct_present, distinct_present


def test_elo_ratings_deterministic_for_fixed_seed():
    """Same inputs + same seed -> byte-identical result frame (ratings AND CIs)."""
    frame = _full_frame()
    first = elo_ratings(frame, metric="uno_c", n_bootstrap=64, seed=7)
    second = elo_ratings(frame, metric="uno_c", n_bootstrap=64, seed=7)
    assert first.equals(second)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_point_estimates_invariant_to_n_bootstrap():
    """elo_rating point estimates do not depend on n_bootstrap (only the CIs do)."""
    frame = _full_frame()
    base = (
        elo_ratings(frame, metric="uno_c", n_bootstrap=0, seed=7)
        .set_index(["benchmark_id", "method_id"])["elo_rating"]
        .sort_index()
    )
    for nb in (16, 64, 200):
        got = (
            elo_ratings(frame, metric="uno_c", n_bootstrap=nb, seed=7)
            .set_index(["benchmark_id", "method_id"])["elo_rating"]
            .sort_index()
        )
        assert np.array_equal(base.to_numpy(), got.to_numpy()), nb


def test_bootstrap_ci_present_and_well_ordered():
    """With bootstrap on, CIs bracket the point estimate; with n_bootstrap=0 -> NaN."""
    frame = _full_frame()
    res = elo_ratings(frame, metric="uno_c", n_bootstrap=64, seed=3)
    assert (res["elo_rating_ci95_low"] <= res["elo_rating"]).all()
    assert (res["elo_rating"] <= res["elo_rating_ci95_high"]).all()
    zero = elo_ratings(frame, metric="uno_c", n_bootstrap=0, seed=3)
    assert zero["elo_rating_ci95_low"].isna().all()
    assert zero["elo_rating_ci95_high"].isna().all()


def test_ragged_frame_bootstrap_runs_and_is_reproducible():
    """A ragged frame (incl. a no-match cluster) fits and reproduces exactly."""
    frame = _ragged_frame()
    a = elo_ratings(frame, metric="uno_c", n_bootstrap=64, seed=5)
    b = elo_ratings(frame, metric="uno_c", n_bootstrap=64, seed=5)
    assert a.equals(b)
    assert set(a["method_id"]) == {"a", "b", "c"}


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------
def _all_tests():
    return [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]


def main():
    tests = _all_tests()
    failures = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {fn.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
