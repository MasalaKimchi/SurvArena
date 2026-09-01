# SurvArena — Code Quality & Efficiency Assessment

*Two independent read-only audits (programming practice for a scientific
benchmark; time/space efficiency of the hot paths), plus the optimizations
implemented and verified so far. 2026-07-15.*

## Verdict

This is a genuinely strong, above-average scientific-benchmark codebase, clearly
built by someone who understands both survival analysis and reproducibility
engineering. The reproducibility spine is the standout: content-fingerprinted
deterministic split caching with group-disjoint + stratification guards,
reproducibility-first HPO that refuses to let wall-clock silently truncate the
trial budget, failure-captured-as-data with full tracebacks and zero bare
excepts, IPCW-aware metrics with a dependency-free calibration solver, and a
new typed `core/` kernel. The weaknesses are the kind you get from a fast-moving
benchmark mid-refactor — they are what separate a "very good research repo" from
a "citable, living public benchmark," and all are fixable without architectural
upheaval.

## What's already excellent (keep / don't regress)

- **Deterministic split cache** (`data/splitters.py`): row-order-sensitive,
  NaN-safe content fingerprint of X/time/event; group-disjoint CV via
  `StratifiedGroupKFold` with graceful fallback; size-aware stratification
  tolerance; refuses silent regeneration on manifest mismatch. Textbook.
- **Reproducibility-first HPO** (`benchmark/tuning.py:327`): Optuna driven by a
  fixed `n_trials`, wall-clock only as a non-binding cap, warns if it truncates.
- **Failure-as-data** (`runner.py:500-588`, `api/predictor.py:334`): typed
  failure rows + full tracebacks; 0 bare `except:`.
- **Careful metrics** (`evaluation/metrics.py`): Harrell-C on the full test set,
  Uno-C on the IPCW-estimable subset; torch imported *inside* functions so
  evaluation stays import-light.
- **Process isolation for determinism** (`runner.py`): processes (not threads)
  so concurrent `set_global_seed` can't race the global RNG — correct call.
- **`core/` kernel (Phase-1a)**: immutable `ProtocolSpec`, SQLite `ResultStore`
  with parameterized SQL, idempotent UPSERT, SQL-side aggregation. Best-
  documented code in the repo.

## Highest-leverage improvements (practices)

Priority M = must for the citable-benchmark goal, S = should. `[VN]` = verifiable
in a numpy/pandas/stdlib sandbox; `[ENV]` = needs the full stack.

1. **[M][VN] One source of truth for the metric set.** The canonical metric-name
   list is re-declared in ~7 places (`runner.py:555`, `api/predictor.py:1064`,
   `logging/export.py:39`, `evaluation/metrics.py:8`, `core/results/schema.py:81`,
   `logging/export_shared.py`, `evaluation/_metric_stats.py`). As the leaderboard
   grows, this is how a new metric silently goes missing on failure rows or in an
   export column. Derive all of them from one `MetricBundle`.
2. **[M][VN] Ship a hash-pinned lockfile + complete env capture.** No resolved
   lock today; the run manifest's package list is hardcoded and *omits
   torch/torchsurv/scipy* (`logging/manifest.py:34`). A citable benchmark must let
   a reader reproduce a number from code+env — record the full `pip freeze` and a
   `git_dirty` flag per run.
3. **[M][VN] Centralize determinism and prove it.** `utils/seeds.py:9` seeds only
   `random`+`numpy`; torch determinism is decentralized to each deep adapter via
   `set_torch_seed`. A new adapter that forgets it is silently non-reproducible,
   and `ModelCapabilities.deterministic_given_seed` is declared but never
   verified. Add one seed-everything choke point + a test asserting two runs of a
   `deterministic_given_seed` method match bit-for-bit.
4. **[S][VN] Finish or gate the strangler-fig.** The typed `core/` and the
   dict-based runner now coexist with two validators (`spec.py:477` vs
   `runner.py:88`) that can diverge. Either wire the runner onto
   `ProtocolSpec`/`ResultStore` (Phase-1b) or add a test asserting the two agree.
5. **[S][VN] De-brittle edges + decompose `evaluate_split`.** It's ~440 lines and
   its failure branch hand-mirrors a 40-key schema; and `predictor.py:1021`
   branches on an exception *message string* — replace with a typed exception.
6. **[S][VN] Make ruff suppressions real.** `pyproject.toml` sets no
   `[tool.ruff.lint] select`, so the 7 `# noqa: BLE001` are inert. Add an explicit
   rule set so blind-except (and import order, pyupgrade, etc.) are actually
   enforced.

## Efficiency findings (prioritized)

Rank by expected impact on constrained hardware (laptop + free CI) and wide
genomics data. `[VN]`/`[ENV]` as above.

| # | Area | Location | Change | Impact | Risk | |
|---|------|----------|--------|--------|------|---|
| 1 | **BLAS/thread oversubscription** | pool at `runner.py:962`; `rsf.py:45`, `extra_survival_trees.py:48`, `tabular_boosting.py:129` use `n_jobs/thread_count=-1` | Pin `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` + `torch.set_num_threads(1)` in a pool-worker `initializer`; resolve model intra-op parallelism to 1 when pooled | **1.5–4×** on tree/boosting runs | Low | [ENV] to benchmark |
| 2 | **Per-unit X pickling** | `runner.py:52,1159,962` — each unit carries full `X`; `executor.map` pickles per unit | Send only indices; hand the worker X once via initializer / shared memory | Large for wide genomics × many units | Med | [ENV] |
| 3 | **Eager robustness copies** | `data/robustness.py:64` `X.copy(deep=True)` per track, built eagerly `runner.py:1112` | Apply perturbation lazily in the worker from base + recipe | Large memory when enabled | Med | [ENV] |
| 4 | **Breslow O(N²)** | `methods/survival_utils.py:20` — `np.sum(exp_risk[time>=t])` per event time | Sort once; risk-set sums via reverse `cumsum` → O(N log N) | Large for classical Cox on big/low-event data | Low | [VN]¹ |
| 5 | **Pairwise significance rescan** | `evaluation/_significance.py:160` — per method-pair boolean-mask ×2 + merge, O(k²·n) | Pivot stratum once to unit×method wide; deltas by column subtraction | Med–large when #methods large | Med | [VN]¹ |
| 6 | **Elo bootstrap `present` rescan** | `evaluation/_ratings.py` bootstrap loop | Precompute per-cluster method sets; union over picked clusters | Small–med | Low | ✅ **DONE** |
| 7 | **`ResultStore.load_frame` join** | `core/results/store.py` unfiltered path did a redundant `runs` join | Drop the join when fully unfiltered | Med on large stores | Low | ✅ **DONE** |
| 8 | **float32 for dense one-hot/scaled** | `methods/preprocessing.py:43` `.to_numpy()`→float64 | Emit float32 for dense matrices | ~2× memory on wide genomics | Med | [ENV] |
| 9 | **Repeated column typing** | `data/preprocess.py:22`, `feature_roles.py:20` re-run `nunique` per fit | Cache numeric/categorical partition at dataset load | Med on wide genomics | Med | [ENV] |

¹ [VN] in principle, but `survival_utils`/`_significance` need scipy, which is
absent in the current sandbox — so implement + benchmark these on the Mac.

## Implemented and verified now (output byte/value-identical)

Both were snapshotted before/after and proven identical across many inputs, with
regression tests, then independently re-verified:

- **#6 — Elo bootstrap `present` hoist** (`evaluation/_ratings.py`). Removes a
  per-iteration O(#matches) rescan; ratings, CIs, and match counts unchanged.
  New test `tests/test_ratings_bootstrap_identity.py` (5/5). Point estimates
  remain invariant to `n_bootstrap`; ratings+CI order-independent.
- **#7 — `ResultStore.load_frame` join elision** (`core/results/store.py`).
  Drops the redundant `runs` join on the common fully-unfiltered read; gated on
  `not mwhere` so `eligible_only` semantics stay identical. `test_core_results`
  13/13.

## Ready-to-apply heavy-path patches (do on the Mac, then benchmark)

The three biggest real wins are all in the parallel-execution layer and need the
full stack to measure. #1 is the highest-leverage and lowest-risk:

**#1 — pin low-level threads in a pool worker initializer** (sketch):

```python
# survarena/benchmark/runner.py — where ProcessPoolExecutor is created
def _pool_worker_init() -> None:
    import os
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch; torch.set_num_threads(1)
    except Exception:
        pass

executor = ProcessPoolExecutor(max_workers=n_jobs, initializer=_pool_worker_init)
# and resolve model n_jobs/thread_count to 1 whenever n_jobs > 1
```

Rationale: with `n_jobs = cores`, an RSF fit spawns `cores` threads inside each
of `cores` processes → ~cores² threads on `cores` cores. Pinning removes the
thrash; commonly 1.5–4× wall-clock on tree/boosting-heavy runs. Determinism-safe
(fewer threads only). Validate with `pytest` + a small timed benchmark.

**#4 — Breslow via cumsum** (`methods/survival_utils.py`): sort by time once,
compute risk-set sums with a reverse `cumsum` instead of a per-event-time
`np.sum(mask)`; O(N log N) vs O(N²). Guard with a numerical-equivalence test.

## What NOT to optimize (premature)

The Elo epoch loop (sequentially dependent, can't vectorize without changing
numerics), SQLite→DuckDB (aggregates return tiny result sets; stdlib backend is
deliberate), `metric_direction`/sort flips (called once per stratum), and
per-dataset load/profiling (paid once, not per fold). Focus stays on the parallel
execution layer and the classical-Cox Breslow path.
