"""Queryable results store for :class:`RunResult` records.

Reference backend is stdlib :mod:`sqlite3` -- deliberately *not* duckdb/pyarrow
so the store runs on a bare Python install (see ``survarena.core`` docstring).
Only :mod:`sqlite3` and :mod:`pandas` are imported at module load; the optional
Parquet export lazily imports ``pyarrow`` and degrades with a clear error.

Data model (two tables, third-normal-form-ish)
-----------------------------------------------
``runs``          one row per evaluated unit (the scalar identity/status/cost
                  columns of :class:`RunResult`), keyed by ``run_id`` with a
                  ``UNIQUE`` constraint on the natural primary key so an UPSERT
                  is idempotent.
``run_metrics``   a *narrow* (run_id, metric, value) triple store -- sparse, so
                  a run that lacks a metric simply has no row (matching
                  ``eligible_frame``'s ``notna`` semantics). Pivoted to wide
                  columns on read.

Efficiency notes (the maintainer cares about time & space)
----------------------------------------------------------
* **Batch writes in one transaction.** :meth:`append` wraps the whole iterable in
  a single ``with connection:`` block -> one commit / one fsync for N rows, and
  ``executemany`` for the metric triples.
* **Indexes for query pushdown.** PRIMARY KEY on ``run_id`` + ``UNIQUE`` natural
  key give O(log n) UPSERT; ``idx_runs_protocol_method`` serves the
  ``protocol_id`` filter and the per-``method_id`` leaderboard GROUP BY;
  ``idx_run_metrics_metric`` serves the metric-name join/filter.
* **Aggregate in SQL, not pandas.** :meth:`leaderboard` computes
  ``AVG/COUNT/... GROUP BY method_id`` inside SQLite, so only the small
  per-method result set is materialised in Python.
* **Row factory.** ``sqlite3.Row`` gives dict-like access without per-column
  index bookkeeping.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import math
import sqlite3

import pandas as pd

from survarena.core.results.schema import RunResult

__all__ = ["ResultStore"]

# Scalar columns of the ``runs`` table, in canonical order (matches RunResult).
_RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "schema_version",
    "protocol_id",
    "benchmark_id",
    "dataset_id",
    "dataset_version",
    "dataset_sha256",
    "method_id",
    "method_version",
    "hpo_mode",
    "split_id",
    "seed",
    "status",
    "comparison_ineligible",
    "ineligible_reason",
    "fit_seconds",
    "predict_seconds",
    "peak_rss_mb",
    "env_fingerprint",
    "created_at",
)

# Column groups used to order the wide load_frame() output.
_IDENTITY_COLUMNS = _RUN_COLUMNS  # already in the shape we want to lead with

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    schema_version        INTEGER,
    protocol_id           TEXT,
    benchmark_id          TEXT,
    dataset_id            TEXT,
    dataset_version       TEXT,
    dataset_sha256        TEXT,
    method_id             TEXT,
    method_version        TEXT,
    hpo_mode              TEXT,
    split_id              TEXT,
    seed                  INTEGER,
    status                TEXT,
    comparison_ineligible INTEGER,
    ineligible_reason     TEXT,
    fit_seconds           REAL,
    predict_seconds       REAL,
    peak_rss_mb           REAL,
    env_fingerprint       TEXT,
    created_at            TEXT,
    UNIQUE(protocol_id, dataset_id, method_id, hpo_mode, split_id, seed)
)
"""

_CREATE_METRICS = """
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id TEXT,
    metric TEXT,
    value  REAL,
    PRIMARY KEY(run_id, metric),
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
)
"""

_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_runs_protocol_method ON runs(protocol_id, method_id)",
    "CREATE INDEX IF NOT EXISTS idx_run_metrics_metric ON run_metrics(metric)",
)

# SQL aggregate function per requested ``agg`` (all computed inside SQLite).
_AGG_FUNCS = {
    "mean": "AVG",
    "avg": "AVG",
    "min": "MIN",
    "max": "MAX",
    "sum": "TOTAL",  # TOTAL returns 0.0 (not NULL) for empty groups
    "count": "COUNT",
}


def _none_if_nan(value: float | None) -> float | None:
    """Store NaN scalars as SQL NULL (cleaner than a stored IEEE NaN)."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


class ResultStore:
    """A SQLite-backed, queryable store of :class:`RunResult` rows.

    Accepts a filesystem ``path`` or ``":memory:"``. A single long-lived
    connection is held for the store's lifetime (required for ``":memory:"``,
    where each connection is a distinct database). Usable as a context manager.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        # Enforce the run_metrics -> runs referential integrity (off by default
        # in SQLite). We delete child rows before REPLACE-ing a parent, below.
        self._conn.execute("PRAGMA foreign_keys = ON")
        # Wait (rather than immediately raising SQLITE_BUSY) when another writer
        # holds the lock: the CI/community model has concurrent shard writers
        # appending to one .db, so a short busy timeout absorbs their contention.
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._create_schema()

    # -- lifecycle ---------------------------------------------------------
    def _create_schema(self) -> None:
        with self._conn:
            self._conn.execute(_CREATE_RUNS)
            self._conn.execute(_CREATE_METRICS)
            for stmt in _CREATE_INDEXES:
                self._conn.execute(stmt)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ------------------------------------------------------------
    def append(self, results: Iterable[RunResult]) -> int:
        """Idempotently UPSERT ``results``; returns the number of DISTINCT rows written.

        Re-appending a :class:`RunResult` with the same PRIMARY_KEY does *not*
        duplicate the row -- it replaces the ``runs`` row and swaps in the new
        metric set (updated values win). The whole batch runs in one
        transaction. All values are bound as query parameters (no string
        interpolation -> injection-safe).

        Intra-batch de-duplication: if the same PRIMARY_KEY/``run_id`` appears
        more than once *within a single call*, the LAST occurrence wins (matching
        the cross-batch UPSERT semantics), and the return value counts DISTINCT
        rows persisted -- not ``len(results)``.
        """
        # Collapse duplicate run_ids to their last occurrence (last-wins). ``dict``
        # keeps first-seen insertion order while overwriting the value, so the row
        # we persist is always the final one supplied for that key.
        deduped: dict[str, RunResult] = {}
        for result in results:
            deduped[result.run_id] = result
        rows = list(deduped.values())
        if not rows:
            return 0
        insert_run = (
            "INSERT OR REPLACE INTO runs (" + ", ".join(_RUN_COLUMNS) + ") "
            "VALUES (" + ", ".join(["?"] * len(_RUN_COLUMNS)) + ")"
        )
        with self._conn:  # single transaction -> one commit for the whole batch
            cur = self._conn.cursor()
            for result in rows:
                run_id = result.run_id
                # Delete children first so the subsequent INSERT OR REPLACE on the
                # parent cannot trip the FOREIGN KEY (no ON DELETE CASCADE).
                cur.execute("DELETE FROM run_metrics WHERE run_id = ?", (run_id,))
                cur.execute(
                    insert_run,
                    (
                        run_id,
                        int(result.schema_version),
                        result.protocol_id,
                        result.benchmark_id,
                        result.dataset_id,
                        result.dataset_version,
                        result.dataset_sha256,
                        result.method_id,
                        result.method_version,
                        result.hpo_mode,
                        result.split_id,
                        int(result.seed),
                        result.status,
                        1 if result.comparison_ineligible else 0,
                        result.ineligible_reason,
                        _none_if_nan(result.fit_seconds),
                        _none_if_nan(result.predict_seconds),
                        _none_if_nan(result.peak_rss_mb),
                        result.env_fingerprint,
                        result.created_at,
                    ),
                )
                metric_rows = [
                    (run_id, str(metric), float(value))
                    for metric, value in result.metrics.items()
                    if value is not None
                    and not (isinstance(value, float) and math.isnan(value))
                ]
                if metric_rows:
                    cur.executemany(
                        "INSERT OR REPLACE INTO run_metrics (run_id, metric, value) VALUES (?, ?, ?)",
                        metric_rows,
                    )
        return len(rows)

    # -- internal query helpers -------------------------------------------
    @staticmethod
    def _runs_filter(
        protocol_id: str | None,
        eligible_only: bool,
        alias: str = "",
    ) -> tuple[list[str], list[object]]:
        """Build a parameterised WHERE fragment list for the ``runs`` table.

        ``alias`` qualifies the column names (e.g. ``"r"`` inside a join). The
        eligibility predicate mirrors ``evaluation._eligibility.eligible_frame``
        with ``metric=None``: ``status == "success"`` AND NOT comparison_ineligible.
        """
        prefix = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list[object] = []
        if protocol_id is not None:
            clauses.append(f"{prefix}protocol_id = ?")
            params.append(protocol_id)
        if eligible_only:
            clauses.append(f"{prefix}status = 'success'")
            clauses.append(f"{prefix}comparison_ineligible = 0")
        return clauses, params

    # -- reads -------------------------------------------------------------
    def load_frame(
        self,
        *,
        protocol_id: str | None = None,
        eligible_only: bool = False,
    ) -> pd.DataFrame:
        """Return a WIDE DataFrame (one row per run, metrics pivoted to columns).

        Reproduces the runner's ``fold_results`` shape: scalar identity/status/
        cost columns first, then one column per metric name. ``eligible_only``
        applies ``status == "success" AND comparison_ineligible == 0`` (the
        metric-present half of eligibility is inherent -- an absent metric is a
        NaN cell after the pivot). An empty store yields an empty frame that
        still carries the scalar identity columns.
        """
        clauses, params = self._runs_filter(protocol_id, eligible_only)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur = self._conn.execute(f"SELECT * FROM runs{where}", params)
        run_records = [dict(row) for row in cur.fetchall()]
        runs = pd.DataFrame(run_records, columns=list(_RUN_COLUMNS))

        if runs.empty:
            # Expected-columns contract for an empty store: scalar identity cols.
            empty = pd.DataFrame(columns=list(_IDENTITY_COLUMNS))
            empty["comparison_ineligible"] = empty["comparison_ineligible"].astype(bool)
            return empty

        # comparison_ineligible is stored 0/1 -> present as a real bool.
        runs["comparison_ineligible"] = runs["comparison_ineligible"].fillna(0).astype(bool)

        # Pull the metrics for exactly the selected runs (same filter, aliased),
        # then pivot the narrow triples to wide columns.
        mclauses, mparams = self._runs_filter(protocol_id, eligible_only, alias="r")
        mwhere = (" WHERE " + " AND ".join(mclauses)) if mclauses else ""
        if mwhere:
            mcur = self._conn.execute(
                "SELECT rm.run_id AS run_id, rm.metric AS metric, rm.value AS value "
                "FROM run_metrics rm JOIN runs r ON r.run_id = rm.run_id" + mwhere,
                mparams,
            )
        else:
            # Perf: with NO filter (protocol_id is None AND not eligible_only) the
            # runs join is pure overhead -- every run_metrics row has a parent run
            # (FK enforced) and run_id is UNIQUE in runs, so the join is a 1:1
            # pass-through returning exactly the same (run_id, metric, value)
            # triples. Output-preserving: row order is irrelevant downstream (the
            # pivot sorts index+columns, metric_cols is sorted, and the left merge
            # follows the runs row order), so the wide frame is byte-identical.
            # The join is KEPT whenever a filter is present (protocol_id and/or
            # eligible_only): dropping it there could surface a metric column that
            # only an excluded run carries, changing the column set.
            mcur = self._conn.execute("SELECT run_id, metric, value FROM run_metrics")
        metric_records = [dict(row) for row in mcur.fetchall()]

        if metric_records:
            metrics_long = pd.DataFrame(metric_records, columns=["run_id", "metric", "value"])
            wide_metrics = metrics_long.pivot(index="run_id", columns="metric", values="value")
            wide_metrics.columns.name = None
            merged = runs.merge(wide_metrics, left_on="run_id", right_index=True, how="left")
            metric_cols = sorted(wide_metrics.columns.tolist())
        else:
            merged = runs
            metric_cols = []

        ordered = list(_IDENTITY_COLUMNS) + metric_cols
        return merged.reindex(columns=ordered)

    def leaderboard(
        self,
        metric: str,
        *,
        protocol_id: str | None = None,
        agg: str = "mean",
        higher_is_better: bool = True,
    ) -> pd.DataFrame:
        """Per-method aggregate of ``metric`` over ELIGIBLE rows -- computed in SQL.

        Eligible == ``status == "success"`` AND NOT ``comparison_ineligible`` AND
        the run has a non-null value for ``metric``. Returns columns
        ``method_id, n, value_<agg>`` sorted so the BEST method leads, then
        ``method_id``. This is the SQL query that reproduces the current
        leaderboard's per-method ranking column.

        ``agg`` is one of ``mean``/``min``/``max``/``sum``/``count`` (all native
        SQLite aggregates).

        Metric *direction* is caller-driven via ``higher_is_better`` (default
        ``True``): ``True`` sorts the aggregate ``DESC`` (best-first for metrics
        like ``uno_c``/``harrell_c``), ``False`` sorts ``ASC`` (best-first for
        lower-is-better metrics such as ``ibs``, ``brier_*`` or
        ``calibration_*_abs_error_*``). The sort direction is chosen from a fixed
        ``"DESC"``/``"ASC"`` literal (never interpolated from user input).
        Direction is kept a caller concern on purpose: ``core`` must not import
        ``evaluation``'s ``metric_direction`` table, so the store stays
        self-contained (see ``survarena.core`` docstring).
        """
        key = agg.lower()
        func = _AGG_FUNCS.get(key)
        if func is None:
            raise ValueError(
                f"Unsupported agg {agg!r}. Supported (SQL-native): "
                f"{', '.join(sorted(_AGG_FUNCS))}."
            )
        value_col = f"value_{key}"
        # Fixed literal (not user input) -> safe to inline in the ORDER BY.
        direction = "DESC" if higher_is_better else "ASC"

        # ``rm.metric = ?`` lives in the JOIN, so ``metric`` is the FIRST param;
        # the optional protocol filter follows in WHERE order.
        params: list[object] = [metric]
        clauses = ["r.status = 'success'", "r.comparison_ineligible = 0", "rm.value IS NOT NULL"]
        if protocol_id is not None:
            clauses.append("r.protocol_id = ?")
            params.append(protocol_id)
        where = " AND ".join(clauses)

        sql = (
            f"SELECT r.method_id AS method_id, "
            f"       COUNT(rm.value) AS n, "
            f"       {func}(rm.value) AS {value_col} "
            f"FROM runs r "
            f"JOIN run_metrics rm ON rm.run_id = r.run_id AND rm.metric = ? "
            f"WHERE {where} "
            f"GROUP BY r.method_id "
            f"ORDER BY {value_col} {direction}, r.method_id ASC"
        )
        cur = self._conn.execute(sql, params)
        records = [dict(row) for row in cur.fetchall()]
        return pd.DataFrame(records, columns=["method_id", "n", value_col])

    # -- optional columnar export -----------------------------------------
    def export_parquet(self, out_dir: str | Path) -> None:
        """Export the wide frame to a partitioned Parquet dataset.

        Partitioned by ``protocol_id / dataset_id / method_id``. Requires
        ``pyarrow``; if unavailable this raises a clear, informative error rather
        than importing it at module load.
        """
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401
        except Exception as exc:  # ImportError (or a broken install)
            raise RuntimeError(
                "ResultStore.export_parquet requires the optional 'pyarrow' "
                "dependency, which is not installed. Install it (e.g. "
                "`pip install pyarrow`) to enable columnar Parquet export, or "
                "use load_frame()/CSV instead."
            ) from exc

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        frame = self.load_frame()
        partition_cols = [c for c in ("protocol_id", "dataset_id", "method_id") if c in frame.columns]
        # pandas writes a Hive-style partitioned directory dataset via pyarrow.
        frame.to_parquet(
            out_path,
            engine="pyarrow",
            index=False,
            partition_cols=partition_cols or None,
        )
