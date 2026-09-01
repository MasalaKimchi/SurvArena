# Stack Research

**Domain:** Reproducible tabular survival-analysis benchmarking
**Researched:** 2026-08-31
**Confidence:** HIGH for the verification kernel; MEDIUM for future public platform work

## Recommended Stack

### Core Technologies

| Technology | Version policy | Purpose | Why Recommended |
|------------|----------------|---------|-----------------|
| CPython | 3.10–3.12 tested; one canonical 3.11 release image | Benchmark runtime | Matches the package contract while allowing one frozen citable runtime. |
| NumPy / pandas / SciPy / scikit-learn | Fully resolved in a hashed Linux lock | Numeric, tabular, statistics, splitting | Existing code is built around these stable Python interfaces. |
| scikit-survival | Pinned and used as an independent test oracle | Reference survival metrics and classical models | Provides established Harrell, Uno/IPCW, Brier/IBS, AUC, and estimator implementations. |
| PyTorch / pycox / torchsurv | Fully pinned, optional method tier | Deep survival methods and primary metric implementation | Retains the current ecosystem while differential tests guard semantics. |
| SQLite | Runtime version captured in the environment fingerprint | Canonical result collection | Transactional, compact, queryable, portable, and already represented by `ResultStore`. |
| OCI/Docker image | Linux/amd64 base pinned by digest | Citable execution environment | Provides the environment boundary required for reproducible reference results. |

### Supporting Libraries

| Library | Version policy | Purpose | When to Use |
|---------|----------------|---------|-------------|
| `uv` or `pip-tools` | Pinned release tool | Fully resolved hashed lock | Generate and verify the canonical Linux dependency graph. |
| `hypothesis` | Pinned dev dependency | Property/metamorphic testing | Split invariants, prediction contracts, ranking pairing, serialization, and metric bounds. |
| `mypy` or `pyright` | One pinned checker | Static type gate | Enforce typed protocol/model/result boundaries incrementally. |
| `psutil` | Existing pin | Child-process telemetry | Measure per-run RSS/CPU rather than process-lifetime `ru_maxrss`. |
| `pyarrow` | Existing pin; export-only | Parquet interchange | Derived export from SQLite, not an alternative source of truth. |
| `tabulate` | Avoid if possible | Markdown tables | Prefer an internal renderer for the strict audit; otherwise declare it explicitly. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Ruff | Lint and formatting policy | Keep the current Python 3.10 target and 120-character limit. |
| pytest | Unit, differential, property, integration, and E2E tests | Separate fast kernel tests from optional/heavy adapter tests. |
| GitHub Actions | Canonical CI and sharded reference execution | Assert semantic output, not only process exit status. |
| Docker/BuildKit | Build canonical image | Verify image digest, wheel installation, and smoke benchmark in CI. |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| SQLite canonical store | DuckDB | Consider after schema/provenance semantics are stable and analytical query volume justifies it. |
| Local subprocess isolation | Ray/Celery/Kubernetes | Only after a verified kernel exists and distributed scheduling is the actual bottleneck. |
| Dataclasses with explicit validation | Pydantic | Useful if external submission/config APIs later require JSON schema. |
| Frozen lock + image digest | Direct-dependency pins only | Direct pins are acceptable for local development, not citable execution. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Python thread timeouts for model fits | Running Python/native work cannot be safely killed and can leak resources into later runs. | Killable subprocess per run unit. |
| CSV files as canonical state | Weak schema, no transactional conflict detection, duplicated artifacts, and fragile provenance. | SQLite result collection plus derived exports. |
| Process-pool submission of full DataFrames per fold/method | Repeated serialization dominates high-dimensional datasets. | Worker initializers/shared dataset context and index-only run units. |
| Unbounded `n_jobs=-1` inside multiple benchmark workers | Nested parallelism makes budgets and timings incomparable. | Explicit worker/thread caps captured in the recipe. |
| Mutable image tags as release identity | Rebuilds can silently resolve a different base and transitive stack. | Image digest plus hashed dependency lock. |

## Stack Patterns by Variant

**Fast pull-request verification:** core dependencies, synthetic fixtures, one classical E2E cell, and no optional foundation weights.

**Citable reference execution:** locked Linux/amd64 image, frozen datasets, explicit thread/process limits, and append-only result collection.

**Optional frontier methods:** separately identified hardware tier; never mix hardware/runtime claims with the CPU core tier without explicit normalization.

## Version Compatibility

| Package family | Compatibility rule | Notes |
|----------------|--------------------|-------|
| Python / torch / torchvision | Resolve together in the canonical Linux lock | Optional foundation extras need a separate matrix. |
| NumPy / scikit-survival / scikit-learn | Use the resolved canonical-platform wheel set | Differential tests detect semantic drift. |
| AutoGluon / foundation adapters | Pin backbone and adapter versions together | Record foundation weight/model revision in every result. |
| SQLite / Python | Record Python and SQLite runtime versions | SQLite is part of the artifact writer. |

## Sources

- https://scikit-survival.readthedocs.io/en/stable/api/metrics.html — reference survival metric APIs.
- https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html — IPCW support and censoring assumptions.
- https://arxiv.org/abs/2506.16791 — maintained benchmark, validation, time-budget, and ensembling design.
- Repository implementation and verification audit executed 2026-08-31.

---
*Stack research for: SurvArena v2 verified benchmark kernel*
*Researched: 2026-08-31*
