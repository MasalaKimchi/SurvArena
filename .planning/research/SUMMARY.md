# Project Research Summary

**Project:** SurvArena Benchmark Modernization
**Domain:** Reproducible tabular survival-analysis benchmarking
**Researched:** 2026-08-31
**Confidence:** HIGH

## Executive Summary

SurvArena should be built as a small trusted benchmark kernel surrounded by replaceable adapters and reporting layers. The repository already has a strong survival metric base and broad Python model coverage, but current evidence is blocked by incorrect pairing/ranking, fold-level pseudo-replication, conditional-on-success selection, partially group-aware validation, non-killable timeouts, and provenance identities that can overwrite materially different runs.

The recommended path is a strangler migration through one verified vertical sentinel: compile a strict protocol, load a frozen dataset with groups separated from features, create the full split hierarchy, execute one disposable resource-bounded worker, validate predictions, score against verified metric semantics, persist an immutable provenance-complete `RunResult`, and derive common-support performance plus reliability reports. Once that path is authoritative, port the broader adapter roster and freeze the representative protocol.

The main risk is expanding models, datasets, HPO, or public infrastructure before the kernel is correct. That would increase compute and public surface area while making the existing evidence more expensive to invalidate and regenerate.

## Key Findings

### Recommended Stack

- Retain the current Python scientific stack, but resolve it in one hashed Linux/amd64 release lock.
- Use scikit-survival as an independent metric oracle and Hypothesis for invariants.
- Promote SQLite `ResultStore` to the canonical artifact; treat CSV/Parquet as derived exports.
- Replace thread timeouts with supervised disposable processes and explicit native thread limits.
- Add a pinned static type checker, wheel-install verification, and a container build/smoke gate.

### Expected Features

**Must have:** strict release recipe, frozen datasets, group-safe validation, verified metrics, dataset-level statistics, common-support performance, reliability reporting, killable budgets, immutable provenance, and automated clean-environment reproduction.

**Competitive after correctness:** balanced default/tuned arms, IBS primary plus Uno C co-primary, OOF portfolio ensemble, and runtime/quality/reliability Pareto reporting.

**Deferred:** public leaderboard/submissions, DOI/governance automation, and broader survival task types.

### Architecture Approach

The architecture has five authoritative boundaries: frozen specs, split/run planning, disposable execution workers, verified comparison kernel, and canonical result collection. Reports are pure queries over the store. Adapters implement declared capabilities but do not decide benchmark policy.

### Critical Pitfalls

1. Fold-level pseudo-replication and Cartesian pair comparisons.
2. Survivorship bias from dropping failed cells before defining support.
3. Group leakage in inner CV/early stopping and group IDs retained as features.
4. Thread-based timeouts that leave fits running.
5. Run identities that omit material provenance.
6. Per-fold horizon drift and incomplete IPCW support policy.
7. Single-horizon classifiers presented as validated full survival distributions.

## Implications for Roadmap

### Phase 0: Truthful Green Baseline

Resolve red tests, make the strict audit self-contained, migrate split caches explicitly, and remove stale claims before changing protocol behavior.

### Phase 1: Scientific Comparison Kernel

Fix matching, ranks, dataset-level effects/tests, complete common support, reliability views, horizon/support policy, prediction contracts, and differential/property tests.

### Phase 2: Execution and Data Contracts

Implement group-safe validation at every level, killable workers, separate budgets, per-run telemetry, deterministic RNG domains, and explicit capability enforcement.

### Phase 3: Authoritative Protocol and Result Collection

Promote typed specs and `RunResult`/`ResultStore` into the live runner, introduce full recipe identity and conflict rejection, and derive reports from the collection.

### Phase 4: Representative Protocol Pilot

Freeze the medium dataset suite/core model roster, support IBS HPO, define default/tuned parity, add the OOF ensemble, and run sentinel then staged pilots.

### Phase 5: Release Reproduction

Lock and digest the environment, verify wheel/container execution in CI, complete reference runs, reproduce them independently, and publish `survbench-1.0-rc1` artifacts.

### Phase Ordering Rationale

- Statistical correctness comes before new compute.
- Execution/data safety precedes claims about tuning or deep-model fairness.
- Protocol/store integration precedes dataset freezing because release identity depends on both.
- The representative pilot precedes full reference execution so protocol defects remain cheap to correct.
- Public living-platform work begins only after the release candidate is independently reproducible.

### Research Flags

- **Scientific comparison kernel:** validate incomplete-block alternatives and the final multiple-comparison policy during planning.
- **Execution/data contracts:** prototype reliable process-tree termination and cross-platform resource telemetry before committing to an API.
- **Representative protocol:** research legally redistributable non-clinical survival datasets and predeclare curation criteria.
- **Foundation adapters:** validate each output formulation against its scientifically supported metric capabilities.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Builds on the existing stack; additions are standard verification/release tools. |
| Features | HIGH | Derived from executed audit findings and benchmark literature. |
| Architecture | HIGH | Strangler path aligns with existing additive core work and project constraints. |
| Pitfalls | HIGH | Most were demonstrated directly in source or deterministic fixtures. |

**Overall confidence:** HIGH

### Gaps to Address

- Final dataset roster and redistribution rights require dataset-specific curation work.
- Final HPO/runtime budgets require measured pilot data on the canonical hardware tier.
- D-calibration and any likelihood score need independent reference/golden fixtures before inclusion.
- Elo may remain a descriptive secondary view, but its formal role should be decided after primary dataset-level statistics are fixed.

## Sources

### Primary

- https://arxiv.org/abs/2506.16791 — TabArena living benchmark design.
- https://www.jmlr.org/papers/v7/demsar06a.html — multiple-dataset statistical comparison.
- https://scikit-survival.readthedocs.io/en/stable/api/metrics.html — survival metric semantics.
- https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html — IPCW assumptions/support.
- https://jmlr.csail.mit.edu/papers/v21/18-772.html — survival-distribution evaluation and D-calibration.
- SurvArena source, tests, smoke run, differential fixtures, and publication gate audited 2026-08-31.

---
*Research completed: 2026-08-31*
*Ready for roadmap: yes*
