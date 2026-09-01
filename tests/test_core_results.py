"""Standalone tests for ``survarena.core.results`` (Phase-1a).

Runs WITHOUT pytest: plain ``assert``\\ s, a local ``_expect`` helper, and a
``__main__`` harness that prints PASS/FAIL per test and exits non-zero on any
failure. Uses an in-memory SQLite database (and one tmpfile round-trip).

    PYTHONPATH=. python3 tests/test_core_results.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from survarena.core.results import RunResult, ResultStore
from survarena.core.results.schema import KNOWN_METRIC_COLUMNS


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _expect(exc, fn, *args, **kwargs):
    """Assert that calling ``fn(*args, **kwargs)`` raises ``exc``."""
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception as other:  # noqa: BLE001
        raise AssertionError(
            f"expected {exc.__name__}, got {type(other).__name__}: {other}"
        )
    raise AssertionError(f"expected {exc.__name__}, but nothing was raised")


def _mk(method_id, seed, metrics, *, status="success", comparison_ineligible=False,
        protocol_id="protoA", dataset_id="ds1", hpo_mode="no_hpo", split_id="fold0",
        **extra):
    return RunResult(
        protocol_id=protocol_id,
        benchmark_id="benchX",
        dataset_id=dataset_id,
        method_id=method_id,
        hpo_mode=hpo_mode,
        split_id=split_id,
        seed=seed,
        status=status,
        comparison_ineligible=comparison_ineligible,
        metrics=metrics,
        **extra,
    )


def _close(a, b, tol=1e-9):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=tol)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_roundtrip_append_load_frame():
    """Scalars + metrics survive append -> load_frame, incl. odd strings."""
    store = ResultStore(":memory:")
    r1 = _mk("cox", 0, {"harrell_c": 0.72, "ibs": 0.18},
             ineligible_reason="O'Brien; DROP TABLE runs;--",  # injection-safety
             fit_seconds=1.5, predict_seconds=0.25, peak_rss_mb=512.0)
    r2 = _mk("rsf", 1, {"harrell_c": 0.80, "uno_c": 0.79})
    n = store.append([r1, r2])
    assert n == 2, n

    frame = store.load_frame()
    assert len(frame) == 2, len(frame)
    # scalar identity columns present
    for col in ("protocol_id", "benchmark_id", "dataset_id", "method_id",
                "hpo_mode", "split_id", "seed", "status", "comparison_ineligible"):
        assert col in frame.columns, col
    # metric columns pivoted in
    for col in ("harrell_c", "ibs", "uno_c"):
        assert col in frame.columns, col

    by_method = frame.set_index("method_id")
    assert _close(by_method.loc["cox", "harrell_c"], 0.72)
    assert _close(by_method.loc["cox", "ibs"], 0.18)
    assert _close(by_method.loc["cox", "fit_seconds"], 1.5)
    assert _close(by_method.loc["cox", "predict_seconds"], 0.25)
    assert _close(by_method.loc["cox", "peak_rss_mb"], 512.0)
    assert by_method.loc["cox", "ineligible_reason"] == "O'Brien; DROP TABLE runs;--"
    assert _close(by_method.loc["rsf", "harrell_c"], 0.80)
    assert _close(by_method.loc["rsf", "uno_c"], 0.79)
    # cox has no uno_c -> NaN cell after pivot
    assert pd.isna(by_method.loc["cox", "uno_c"])
    # comparison_ineligible surfaces as a real bool
    assert frame["comparison_ineligible"].dtype == bool
    store.close()


def test_idempotent_append_updates_not_duplicates():
    """Re-appending the same PRIMARY_KEY updates values without duplicating."""
    store = ResultStore(":memory:")
    first = _mk("cox", 3, {"harrell_c": 0.70}, fit_seconds=1.0)
    store.append([first])
    # same PK (protocol/dataset/method/hpo_mode/split/seed), new values
    second = _mk("cox", 3, {"harrell_c": 0.91}, fit_seconds=2.0, status="success")
    assert first.run_id == second.run_id, "same PK must yield same run_id"
    store.append([second])

    frame = store.load_frame()
    assert len(frame) == 1, f"expected 1 row after idempotent re-append, got {len(frame)}"
    assert _close(frame.iloc[0]["harrell_c"], 0.91), "metric value must be updated"
    assert _close(frame.iloc[0]["fit_seconds"], 2.0), "scalar must be updated"

    # a stale metric that no longer exists in the new record must be gone
    store.append([_mk("cox", 3, {"harrell_c": 0.95, "ibs": 0.2})])
    store.append([_mk("cox", 3, {"harrell_c": 0.96})])  # drops ibs
    frame = store.load_frame()
    assert len(frame) == 1
    assert _close(frame.iloc[0]["harrell_c"], 0.96)
    assert "ibs" not in frame.columns or pd.isna(frame.iloc[0]["ibs"]), "stale metric not replaced"
    store.close()


def _leaderboard_dataset():
    # A: means over eligible seeds = (0.70+0.80)/2 = 0.75
    # B: means over eligible seeds = (0.60+0.66)/2 = 0.63
    # plus one failed A row and one comparison_ineligible B row (must be ignored)
    return [
        _mk("A", 0, {"harrell_c": 0.70}, split_id="f0"),
        _mk("A", 1, {"harrell_c": 0.80}, split_id="f1"),
        _mk("B", 0, {"harrell_c": 0.60}, split_id="f0"),
        _mk("B", 1, {"harrell_c": 0.66}, split_id="f1"),
        _mk("A", 2, {"harrell_c": 0.99}, split_id="f2", status="failed"),
        _mk("B", 2, {"harrell_c": 0.01}, split_id="f2", comparison_ineligible=True),
    ]


def _pandas_leaderboard_reference(results, metric):
    df = pd.DataFrame([r.to_row() for r in results])
    elig = df[(df["status"] == "success") & (~df["comparison_ineligible"].astype(bool))]
    elig = elig[elig[metric].notna()]
    ref = (
        elig.groupby("method_id")[metric]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "value_mean", "count": "n"})
    )
    ref = ref.sort_values(["value_mean", "method_id"], ascending=[False, True]).reset_index(drop=True)
    return ref


def test_leaderboard_sql_matches_pandas_reference():
    store = ResultStore(":memory:")
    results = _leaderboard_dataset()
    store.append(results)

    got = store.leaderboard("harrell_c")
    ref = _pandas_leaderboard_reference(results, "harrell_c")

    assert list(got.columns) == ["method_id", "n", "value_mean"], list(got.columns)
    assert got["method_id"].tolist() == ref["method_id"].tolist(), (
        got["method_id"].tolist(), ref["method_id"].tolist())
    assert got["n"].tolist() == ref["n"].tolist(), (got["n"].tolist(), ref["n"].tolist())
    for a, b in zip(got["value_mean"].tolist(), ref["value_mean"].tolist()):
        assert _close(a, b), (a, b)

    # sanity: eligible counts are 2 per method (the 3rd row of each was excluded)
    assert got.set_index("method_id").loc["A", "n"] == 2
    assert got.set_index("method_id").loc["B", "n"] == 2
    # order: A (0.75) before B (0.63)
    assert got["method_id"].tolist() == ["A", "B"]

    # unsupported aggregate is rejected clearly
    _expect(ValueError, store.leaderboard, "harrell_c", agg="median")
    store.close()


def test_eligibility_filtering():
    """Failed + comparison_ineligible rows are excluded from eligible views."""
    store = ResultStore(":memory:")
    store.append(_leaderboard_dataset())

    # load_frame() keeps everything (6 rows), eligible_only keeps 4
    assert len(store.load_frame()) == 6
    elig = store.load_frame(eligible_only=True)
    assert len(elig) == 4, len(elig)
    present = set(zip(elig["method_id"], elig["seed"]))
    assert (("A", 2) not in present), "failed row leaked into eligible_only"
    assert (("B", 2) not in present), "comparison_ineligible row leaked into eligible_only"
    assert present == {("A", 0), ("A", 1), ("B", 0), ("B", 1)}, present

    # leaderboard n never counts the excluded rows
    lb = store.leaderboard("harrell_c").set_index("method_id")
    assert lb.loc["A", "n"] == 2 and lb.loc["B", "n"] == 2
    store.close()


def test_from_fold_row_realistic():
    """from_fold_row maps a realistic runner ``record`` dict faithfully."""
    row = {
        "benchmark_id": "bench_main",
        "dataset_id": "veterans",
        "method_id": "deepsurv",
        "hpo_mode": "hpo",
        "split_id": "repeat_0_fold_1",
        "seed": np.int64(7),
        "status": "success",
        "comparison_ineligible": False,
        "ineligible_reason": "",
        # metrics (CORE + MANUSCRIPT + dynamic)
        "validation_score": 0.71,
        "uno_c": 0.688,
        "harrell_c": 0.702,
        "ibs": 0.171,
        "td_auc_25": 0.75,
        "td_auc_50": 0.73,
        "brier_50": 0.19,
        "calibration_slope_abs_error_50": 0.12,
        "net_benefit_50": 0.05,
        "td_auc_75": np.nan,  # NaN metric -> dropped
        # timing (mapped to scalar fields / dropped)
        "tuning_time_sec": 3.0,
        "runtime_sec": 12.5,
        "fit_time_sec": 8.0,
        "infer_time_sec": 0.4,
        "peak_memory_mb": 1024.0,
        # non-metric numeric columns -> must NOT become metrics
        "retry_attempt": 0,
        "realized_trial_count": 5,
    }
    rr = RunResult.from_fold_row(row, protocol_id="protocol_v1")

    assert rr.protocol_id == "protocol_v1"
    assert rr.benchmark_id == "bench_main"
    assert rr.dataset_id == "veterans"
    assert rr.method_id == "deepsurv"
    assert rr.hpo_mode == "hpo"
    assert rr.split_id == "repeat_0_fold_1"
    assert rr.seed == 7 and isinstance(rr.seed, int)
    assert rr.status == "success"
    assert rr.comparison_ineligible is False
    # timing mapping
    assert _close(rr.fit_seconds, 8.0)
    assert _close(rr.predict_seconds, 0.4)
    assert _close(rr.peak_rss_mb, 1024.0)
    # metric collection
    assert _close(rr.metrics["uno_c"], 0.688)
    assert _close(rr.metrics["harrell_c"], 0.702)
    assert _close(rr.metrics["ibs"], 0.171)
    assert _close(rr.metrics["td_auc_25"], 0.75)
    assert _close(rr.metrics["brier_50"], 0.19)
    assert _close(rr.metrics["calibration_slope_abs_error_50"], 0.12)
    assert _close(rr.metrics["net_benefit_50"], 0.05)
    assert _close(rr.metrics["validation_score"], 0.71)
    # NaN metric dropped
    assert "td_auc_75" not in rr.metrics
    # timing / bookkeeping numerics excluded from metrics
    for bad in ("tuning_time_sec", "runtime_sec", "fit_time_sec", "infer_time_sec",
                "peak_memory_mb", "retry_attempt", "realized_trial_count", "seed"):
        assert bad not in rr.metrics, bad
    # metrics mapping is immutable (item assignment is unsupported)
    def _mutate():
        rr.metrics["x"] = 1.0  # type: ignore[index]
    _expect(TypeError, _mutate)

    # tolerant defaults for a near-empty row
    sparse = RunResult.from_fold_row({"method_id": "m"}, protocol_id="p")
    assert sparse.hpo_mode == "no_hpo"     # export_coverage_matrix fallback
    assert sparse.status == "unknown"      # -> not eligible
    assert sparse.seed == 0
    assert math.isnan(sparse.fit_seconds)
    assert dict(sparse.metrics) == {}

    # from_fold_row output stores/reloads through the store unchanged
    store = ResultStore(":memory:")
    store.append([rr])
    loaded = store.load_frame()
    assert len(loaded) == 1
    assert _close(loaded.iloc[0]["harrell_c"], 0.702)
    assert loaded.iloc[0]["method_id"] == "deepsurv"
    store.close()


def test_empty_store_returns_expected_columns():
    store = ResultStore(":memory:")
    frame = store.load_frame()
    assert len(frame) == 0, len(frame)
    for col in ("run_id", "protocol_id", "benchmark_id", "dataset_id", "method_id",
                "hpo_mode", "split_id", "seed", "status", "comparison_ineligible"):
        assert col in frame.columns, col
    # leaderboard on an empty store is an empty, well-formed frame
    lb = store.leaderboard("harrell_c")
    assert len(lb) == 0
    assert list(lb.columns) == ["method_id", "n", "value_mean"]
    store.close()


def test_file_backed_roundtrip():
    """A real .db file round-trips independently of export capabilities."""
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "results.db")
        store = ResultStore(path)
        store.append([_mk("cox", 0, {"harrell_c": 0.7}), _mk("rsf", 0, {"harrell_c": 0.8})])
        store.close()

        reopened = ResultStore(path)
        frame = reopened.load_frame()
        assert len(frame) == 2, len(frame)
        assert set(frame["method_id"]) == {"cox", "rsf"}
        reopened.close()


def test_export_parquet_round_trips_when_pyarrow_is_available():
    store = ResultStore(":memory:")
    store.append([_mk("cox", 0, {"harrell_c": 0.7}), _mk("rsf", 0, {"harrell_c": 0.8})])
    with tempfile.TemporaryDirectory() as temp_dir:
        export_dir = Path(temp_dir) / "parquet"
        store.export_parquet(export_dir)

        assert list(export_dir.rglob("*.parquet"))
        exported = pd.read_parquet(export_dir)
        assert len(exported) == 2
        assert set(exported["method_id"]) == {"cox", "rsf"}
    store.close()


def test_export_parquet_reports_forced_pyarrow_import_failure():
    store = ResultStore(":memory:")
    with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
        with tempfile.TemporaryDirectory() as temp_dir:
            _expect(RuntimeError, store.export_parquet, Path(temp_dir) / "parquet")

    store.close()


def test_known_metric_columns_cover_runner_metrics():
    """Guard: the metric allow-list still covers the runner's canonical metrics."""
    for m in ("uno_c", "harrell_c", "ibs", "td_auc_25", "td_auc_50", "td_auc_75",
              "brier_50", "calibration_slope_abs_error_50",
              "calibration_intercept_abs_error_50", "net_benefit_50", "validation_score"):
        assert m in KNOWN_METRIC_COLUMNS, m


def test_leaderboard_lower_is_better_ordering():
    """higher_is_better=False ranks LOWER aggregates first (ibs/brier/etc.)."""
    store = ResultStore(":memory:")
    store.append([
        _mk("A", 0, {"ibs": 0.10}, split_id="f0"),
        _mk("A", 1, {"ibs": 0.20}, split_id="f1"),   # mean 0.15 (better, lower)
        _mk("B", 0, {"ibs": 0.30}, split_id="f0"),
        _mk("B", 1, {"ibs": 0.40}, split_id="f1"),   # mean 0.35 (worse, higher)
    ])
    # default higher_is_better=True -> DESC -> worse(B, 0.35) leads
    hib = store.leaderboard("ibs")
    assert hib["method_id"].tolist() == ["B", "A"], hib["method_id"].tolist()
    # lower-is-better -> ASC -> best(A, 0.15) leads; same values, different order
    lib = store.leaderboard("ibs", higher_is_better=False)
    assert lib["method_id"].tolist() == ["A", "B"], lib["method_id"].tolist()
    assert list(lib.columns) == ["method_id", "n", "value_mean"], list(lib.columns)
    by_method = lib.set_index("method_id")
    assert _close(by_method.loc["A", "value_mean"], 0.15)
    assert _close(by_method.loc["B", "value_mean"], 0.35)
    assert by_method.loc["A", "n"] == 2 and by_method.loc["B", "n"] == 2
    store.close()


def test_protocol_id_filter_on_load_frame_and_leaderboard():
    """protocol_id filters both load_frame() and leaderboard()."""
    store = ResultStore(":memory:")
    store.append([
        _mk("cox", 0, {"harrell_c": 0.70}, protocol_id="protoA", split_id="f0"),
        _mk("cox", 1, {"harrell_c": 0.72}, protocol_id="protoA", split_id="f1"),  # A mean 0.71
        _mk("cox", 0, {"harrell_c": 0.90}, protocol_id="protoB", split_id="f0"),  # B mean 0.90
    ])
    # load_frame filter
    assert len(store.load_frame()) == 3
    a_rows = store.load_frame(protocol_id="protoA")
    assert len(a_rows) == 2 and set(a_rows["protocol_id"]) == {"protoA"}
    b_rows = store.load_frame(protocol_id="protoB")
    assert len(b_rows) == 1 and set(b_rows["protocol_id"]) == {"protoB"}

    # leaderboard filter
    lb_all = store.leaderboard("harrell_c").set_index("method_id")
    assert lb_all.loc["cox", "n"] == 3
    lb_a = store.leaderboard("harrell_c", protocol_id="protoA").set_index("method_id")
    assert lb_a.loc["cox", "n"] == 2 and _close(lb_a.loc["cox", "value_mean"], 0.71)
    lb_b = store.leaderboard("harrell_c", protocol_id="protoB").set_index("method_id")
    assert lb_b.loc["cox", "n"] == 1 and _close(lb_b.loc["cox", "value_mean"], 0.90)
    store.close()


def test_multi_metric_idempotent_replacement():
    """Re-appending a PK with a smaller metric set replaces the set wholesale."""
    store = ResultStore(":memory:")
    store.append([_mk("cox", 5, {"harrell_c": 0.70, "uno_c": 0.68, "ibs": 0.20})])
    frame = store.load_frame()
    assert len(frame) == 1
    row = frame.iloc[0]
    for m in ("harrell_c", "uno_c", "ibs"):
        assert m in frame.columns and not pd.isna(row[m]), m

    # re-append the SAME PK carrying only {harrell_c} -> uno_c/ibs must vanish
    store.append([_mk("cox", 5, {"harrell_c": 0.99})])
    frame = store.load_frame()
    assert len(frame) == 1, f"expected exactly one run, got {len(frame)}"
    row = frame.iloc[0]
    assert _close(row["harrell_c"], 0.99), "surviving metric must be updated"
    for gone in ("uno_c", "ibs"):
        assert gone not in frame.columns or pd.isna(row[gone]), f"stale metric {gone} not cleared"
    store.close()


def test_intra_batch_duplicate_pk_dedup():
    """Duplicate PKs within ONE append() collapse to the LAST occurrence."""
    store = ResultStore(":memory:")
    dup_a = _mk("cox", 7, {"harrell_c": 0.10, "ibs": 0.30}, fit_seconds=1.0)
    dup_b = _mk("cox", 7, {"harrell_c": 0.99}, fit_seconds=2.0)  # same PK, later wins
    other = _mk("rsf", 7, {"harrell_c": 0.55})
    assert dup_a.run_id == dup_b.run_id, "duplicates must share a run_id"
    assert dup_a.run_id != other.run_id, "distinct PK must differ"

    n = store.append([dup_a, dup_b, other])
    # return value counts DISTINCT rows persisted (2), NOT len(input) == 3
    assert n == 2, f"expected 2 distinct rows persisted, got {n}"

    frame = store.load_frame()
    assert len(frame) == 2, len(frame)
    cox = frame.set_index("method_id").loc["cox"]
    assert _close(cox["harrell_c"], 0.99), "last occurrence's metric must win"
    assert _close(cox["fit_seconds"], 2.0), "last occurrence's scalar must win"
    # dup_a's ibs must not survive (dup_b, the last write, had no ibs)
    assert "ibs" not in frame.columns or pd.isna(cox["ibs"]), "stale metric from earlier dup leaked"
    store.close()


def _reference_load_frame_via_join(store, *, protocol_id=None, eligible_only=False):
    """Reference implementation of load_frame that ALWAYS fetches metrics via the
    ``run_metrics JOIN runs`` query (the pre-optimization behavior).

    load_frame() now skips that join on the fully-unfiltered path (a perf win);
    this helper reproduces the OLD join-based read so a test can assert the new
    unfiltered frame is byte-identical to what the join produced.
    """
    from survarena.core.results.store import _IDENTITY_COLUMNS, _RUN_COLUMNS

    clauses, params = store._runs_filter(protocol_id, eligible_only)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    cur = store._conn.execute(f"SELECT * FROM runs{where}", params)
    runs = pd.DataFrame([dict(r) for r in cur.fetchall()], columns=list(_RUN_COLUMNS))
    if runs.empty:
        empty = pd.DataFrame(columns=list(_IDENTITY_COLUMNS))
        empty["comparison_ineligible"] = empty["comparison_ineligible"].astype(bool)
        return empty
    runs["comparison_ineligible"] = runs["comparison_ineligible"].fillna(0).astype(bool)

    mclauses, mparams = store._runs_filter(protocol_id, eligible_only, alias="r")
    mwhere = (" WHERE " + " AND ".join(mclauses)) if mclauses else ""
    mcur = store._conn.execute(
        "SELECT rm.run_id AS run_id, rm.metric AS metric, rm.value AS value "
        "FROM run_metrics rm JOIN runs r ON r.run_id = rm.run_id" + mwhere,
        mparams,
    )
    metric_records = [dict(r) for r in mcur.fetchall()]
    if metric_records:
        metrics_long = pd.DataFrame(metric_records, columns=["run_id", "metric", "value"])
        wide_metrics = metrics_long.pivot(index="run_id", columns="metric", values="value")
        wide_metrics.columns.name = None
        merged = runs.merge(wide_metrics, left_on="run_id", right_index=True, how="left")
        metric_cols = sorted(wide_metrics.columns.tolist())
    else:
        merged = runs
        metric_cols = []
    return merged.reindex(columns=list(_IDENTITY_COLUMNS) + metric_cols)


def test_load_frame_unfiltered_join_elision_is_identical():
    """Perf regression guard: the unfiltered load_frame (no runs-join metric read)
    is byte-identical to the reference join-based read, and the join is retained
    whenever a filter is present (so a metric carried only by an EXCLUDED row does
    not leak in as a spurious column)."""
    store = ResultStore(":memory:")
    store.append([
        _mk("cox", 0, {"harrell_c": 0.72, "uno_c": 0.70, "ibs": 0.18},
            protocol_id="P1", dataset_id="d1", split_id="f0"),
        _mk("rsf", 0, {"harrell_c": 0.80, "uno_c": 0.79},
            protocol_id="P1", dataset_id="d1", split_id="f0"),
        _mk("deepsurv", 0, {"harrell_c": 0.66}, protocol_id="P1", dataset_id="d1", split_id="f0"),
        _mk("cox", 1, {"harrell_c": 0.71, "ibs": 0.19}, protocol_id="P1", dataset_id="d2", split_id="f1"),
        # A FAILED row carrying a metric NO eligible row has (adversarial):
        _mk("cox", 2, {"harrell_c": 0.99, "weird_metric": 0.5}, protocol_id="P1",
            dataset_id="d2", split_id="f2", status="failed"),
        # A comparison_ineligible row:
        _mk("rsf", 2, {"harrell_c": 0.01}, protocol_id="P1", dataset_id="d2", split_id="f2",
            comparison_ineligible=True),
        _mk("cox", 0, {"harrell_c": 0.60, "brier_50": 0.2}, protocol_id="P2", dataset_id="d1", split_id="f0"),
    ])

    # The unfiltered path (new no-join read) must equal the old join-based read.
    got = store.load_frame()
    ref = _reference_load_frame_via_join(store)
    assert list(got.columns) == list(ref.columns), (list(got.columns), list(ref.columns))
    assert list(got.dtypes) == list(ref.dtypes)
    assert got.equals(ref), "unfiltered load_frame diverged from the join reference"
    # The failed row's exclusive metric IS present on the unfiltered frame (that
    # row is included when eligible_only=False), as an all-else-NaN column.
    assert "weird_metric" in got.columns

    # Every filtered combination must ALSO match the join reference (the code keeps
    # the join there) -- crucially, eligible_only drops 'weird_metric' entirely.
    for protocol_id in (None, "P1", "P2"):
        for eligible_only in (False, True):
            g = store.load_frame(protocol_id=protocol_id, eligible_only=eligible_only)
            r = _reference_load_frame_via_join(store, protocol_id=protocol_id, eligible_only=eligible_only)
            assert list(g.columns) == list(r.columns), (protocol_id, eligible_only, list(g.columns), list(r.columns))
            assert g.equals(r), (protocol_id, eligible_only)
    # eligible_only removes the metric that only the excluded rows carried.
    elig = store.load_frame(eligible_only=True)
    assert "weird_metric" not in elig.columns, "excluded-only metric leaked into eligible view"
    store.close()


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
