# Phase 1: Truthful Green Baseline - Context

**Gathered:** 2026-08-31
**Status:** Ready for planning
**Source:** Approved 2026-08-31 expert audit, v2 requirements, and roadmap

<domain>
## Phase Boundary

Restore a trustworthy verification baseline without changing the scientific comparison protocol. Phase 1 resolves the current red suite, makes the strict release audit self-contained, provides explicit legacy split-cache compatibility guidance, adds a declared incremental static-type gate, and corrects maintained status documentation. Statistical pairing/ranking, group-safe execution, timeout isolation, typed-runner integration, dataset expansion, and new benchmark evidence belong to later phases.

</domain>

<decisions>
## Implementation Decisions

### Verification Baseline
- D-01: All four currently failing tests must be resolved by clarifying and testing intended behavior, not by weakening assertions or broadly skipping environment-dependent paths.
- D-02: `ruff check survarena tests scripts`, the declared mypy scope, the full pytest suite, `compileall`, and a clean-root CoxPH/WHAS500 smoke are the Phase 1 baseline gates.
- D-03: Static typing begins with authoritative/lightweight kernel and changed baseline modules; its exact scope must be declared in `pyproject.toml`, documentation, and CI rather than implying whole-package coverage.

### Current Failing Contracts
- D-04: Model-artifact persistence behavior must be explicit: `hpo_mode` participates in artifact identity and a requested artifact write that cannot serialize fitted state returns a structured failure or raises the documented error consistently across code and tests.
- D-05: Resume identity includes comparison arm. A legacy success row lacking `hpo_mode` may satisfy only a `no_hpo` unit and must not suppress a configured `hpo` run.
- D-06: Parallel execution tests must verify `ProcessPoolExecutor` worker configuration and results without depending on monkeypatch state mutation inside child processes.
- D-07: Parquet tests must cover both installed and unavailable `pyarrow` branches deterministically rather than assuming the ambient environment lacks the dependency.

### Release Audit and Split Cache
- D-08: The strict publication audit must not rely on pandas' optional `tabulate` dependency; use a small deterministic internal Markdown-table renderer.
- D-09: Split-cache mismatch remains fail-closed. The error/report must identify mismatched manifest fields, the cache path, and the explicit `--regenerate-splits` recovery command; regeneration remains opt-in.
- D-10: Legacy manifests are never silently rewritten or treated as citable under new split semantics.

### Documentation Truthfulness
- D-11: `PROJECT_STATE.md`, `docs/test_status.md`, and `docs/ci_and_reproducibility.md` must describe current executed evidence, distinguish uncommitted/scaffolded infrastructure from live verification, and retain the statement that historical benchmark matrices require regeneration.
- D-12: No new benchmark results, datasets, model adapters, leaderboard claims, or manuscript conclusions are produced in Phase 1.

### the agent's Discretion
- Exact helper names and placement for Markdown rendering and split-manifest diff formatting.
- Whether ProcessPool behavior is tested through an injectable executor factory or a deterministic top-level worker fixture.
- Exact initial mypy module allow-list, provided it covers `survarena/core`, the Phase 1 touched modules, and has a documented expansion path.
- Organization of smoke-test helper fixtures and documentation tables.

</decisions>

<canonical_refs>
## Canonical References

### Milestone contracts
- `.planning/PROJECT.md` — core value, constraints, and v2 decisions.
- `.planning/REQUIREMENTS.md` — BASE-01 through BASE-04 acceptance scope.
- `.planning/ROADMAP.md` — Phase 1 boundary and success criteria.
- `.planning/research/SUMMARY.md` — audit synthesis and phase ordering.

### Current verification and release state
- `PROJECT_STATE.md` — historical evidence invalidation and remaining work.
- `docs/test_status.md` — stale constrained-environment verification claims to replace.
- `docs/ci_and_reproducibility.md` — current CI/container caveats.
- `pyproject.toml` — dependencies and tool configuration.
- `.github/workflows/ci.yml` — current proposed CI gates.

### Failing contracts
- `tests/test_benchmark.py` — artifact, resume, and process-parallel tests.
- `tests/test_core_results.py` — environment-dependent Parquet test.
- `survarena/benchmark/runner.py` — artifact identity, resume, and process execution behavior.
- `survarena/core/results/store.py` — Parquet export branch.

### Audit and split compatibility
- `scripts/audit_manuscript_publishability.py` — strict release audit and optional `to_markdown` dependency.
- `survarena/data/splitters.py` — split-manifest mismatch handling.
- `survarena/run_benchmark.py` and `survarena/cli.py` — `--regenerate-splits` user entry points.

</canonical_refs>

<specifics>
## Specific Ideas

- Current targeted failure baseline: four failures in approximately 16 seconds; full suite baseline: 4 failed, 246 passed, 6 skipped.
- Mypy 2.3.1 is the current release as of 2026-08-31; pin the chosen development version until the canonical lock is created in Phase 6.
- A clean-root CoxPH/WHAS500 one-seed smoke already succeeds and should become a stable semantic assertion gate.

</specifics>

<deferred>
## Deferred Ideas

- Matched pairwise statistics, common-support rankings, and IPCW/horizon changes → Phase 2.
- Killable subprocess workers, group-safe inner validation, and per-run telemetry → Phase 3.
- Authoritative `ProtocolSpec`/`ResultStore` integration → Phase 4.
- Dataset/model roster expansion and tuned/ensemble evidence → Phase 5.
- Locked container and independent reference reproduction → Phase 6.

</deferred>

---
*Phase: 01-truthful-green-baseline*
*Context gathered: 2026-08-31 from approved audit and milestone contracts*
