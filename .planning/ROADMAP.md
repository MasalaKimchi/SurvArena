# Roadmap: SurvArena Benchmark Modernization

## Overview

Milestone v2.0 converts the current manuscript-oriented benchmark into the verified kernel behind `survbench-1.0-rc1`. Work starts by restoring truthful green verification, then fixes the scientific unit of comparison before changing execution semantics. Typed protocol/result contracts become authoritative only after statistics and worker/data boundaries are tested. The final stages freeze a medium representative suite, pilot default/tuned/ensemble behavior, and prove clean-environment reproduction before any public living-platform work begins.

## Milestone

- [ ] **v2.0 Verified Benchmark Kernel** — Phases 1–6; target `survbench-1.0-rc1`.

## Phases

- [x] **Phase 1: Truthful Green Baseline** — Restore executable, honest development and release gates. (completed 2026-09-01)
- [ ] **Phase 2: Scientific Comparison Kernel** — Make pairing, ranks, support, inference, metrics, and prediction eligibility scientifically defensible.
- [ ] **Phase 3: Execution and Data Safety** — Enforce group-safe validation and killable, resource-bounded, deterministic run units.
- [ ] **Phase 4: Authoritative Protocol and Results** — Route the live runner through strict typed contracts and one immutable provenance-complete collection.
- [ ] **Phase 5: Representative Protocol Pilot** — Freeze the curated core suite/roster and validate default, tuned, and ensemble protocol semantics on sentinel runs.
- [ ] **Phase 6: Release Reproduction** — Lock the environment, enforce CI/release gates, regenerate evidence, and independently reproduce the release candidate.

## Phase Details

### Phase 1: Truthful Green Baseline

**Goal:** Establish a trustworthy baseline where every documented verification command runs and accurately reports current evidence status.

**Depends on:** Nothing.

**Requirements:** BASE-01, BASE-02, BASE-03, BASE-04

**Success Criteria:**

1. A contributor can run the declared lint, static-type, test, and core smoke commands with no failures.
2. The strict manuscript/release audit runs from declared dependencies and returns an actionable verdict.
3. Legacy split caches produce an explicit compatibility/migration path rather than blocking normal operation without guidance.
4. Maintained status documentation no longer presents invalidated historical results or constrained-environment checks as current release evidence.

**Plans:** 3/3 plans complete

- Wave 1 — `01-01`: Repair artifact/resume/process/Parquet contracts and establish incremental mypy.
- Wave 1 — `01-02`: Make the release audit self-contained and split-cache mismatch diagnostics actionable.
- Wave 2 — `01-03`: Align CI/documentation and execute the integrated baseline plus semantic smoke.

### Phase 2: Scientific Comparison Kernel

**Goal:** Produce statistically valid, censoring-aware comparisons whose support, pairing, inference unit, and prediction eligibility are explicit and verified.

**Depends on:** Phase 1.

**Requirements:** STAT-01, STAT-02, STAT-03, STAT-04, STAT-05, STAT-06, STAT-07, METR-01, METR-02, METR-03, METR-04, METR-05

**Success Criteria:**

1. Constructed pairing counterexamples return exactly the matched win counts/ranks and cannot create Cartesian cross-split comparisons.
2. Pairwise and multiple-method inference reports datasets as the experimental sample and fails closed when common-support/block assumptions do not hold.
3. Performance, coverage, fallback, invalid-output, timeout, and failure views are separately queryable and a failure cannot improve headline eligibility.
4. Metric edge fixtures agree with independent references within declared tolerances and every result records its evaluation window/IPCW support.
5. Invalid or unsupported prediction bundles are rejected before scoring and cannot enter full-distribution headline tables.

**Plans:** 1/3 plans executed

- Wave 1 — `02-01`: Establish exact comparison cells, complete common support, dataset ranks/wins, reliability, and equal-weight ratings.
- Wave 2 — `02-02`: Make pairwise and multiple-method inference dataset-level, complete-block, and equally weighted.
- Wave 3 — `02-03`: Enforce prediction contracts, fixed evaluation support, independent metric references, and runner eligibility.

### Phase 3: Execution and Data Safety

**Goal:** Execute each benchmark cell under fair, killable budgets with no group leakage, hidden oversubscription, or cross-run resource contamination.

**Depends on:** Phase 2.

**Requirements:** EXEC-01, EXEC-02, EXEC-03, EXEC-04, EXEC-05, DATA-01, DATA-02, DATA-03, MODL-01

**Success Criteria:**

1. A deliberately hanging model and its descendants are terminated within the configured budget/grace period, and later cells run in clean resources.
2. Groups are absent from model features and disjoint across outer CV, inner HPO CV, and early-stopping validation.
3. Split manifests detect any material dataset/group/row-order/algorithm change and reject folds with inadequate events or censoring.
4. Run telemetry reflects only the individual cell, while explicit worker/native-thread limits prevent nested oversubscription.
5. Every maintained adapter passes a capability conformance suite covering validation, refit, predictions, determinism, fallback, and metric eligibility.

**Plans:** TBD during `$gsd-plan-phase 3`.

### Phase 4: Authoritative Protocol and Results

**Goal:** Make typed release recipes and the canonical result collection the only authoritative path through execution, resume, merge, comparison, and reporting.

**Depends on:** Phase 3.

**Requirements:** CORE-01, CORE-02, CORE-03, STORE-01, STORE-02, STORE-03, STORE-04, STORE-05

**Success Criteria:**

1. Ambiguous or incoherent configurations fail before any dataset/model work, and the runner no longer relies on parallel legacy policy flags.
2. Material changes to code, data, methods, splits, environment, hardware tier, or protocol change the release recipe/run identity.
3. Identical result appends are idempotent and provenance conflicts are rejected without overwriting prior evidence.
4. Resume/retry/merge and every table/figure operate from one canonical SQLite collection containing success and failure records.
5. A generated report identifies its source collection digest, schema, support query, and release recipe.

**Plans:** TBD during `$gsd-plan-phase 4`.

### Phase 5: Representative Protocol Pilot

**Goal:** Freeze and validate the medium balanced dataset/model protocol that will become the first release candidate.

**Depends on:** Phase 4.

**Requirements:** SUIT-01, SUIT-02, SUIT-03, PROT-01, PROT-02, PROT-03, PROT-04, PROT-05, PROT-06

**Success Criteria:**

1. Every core dataset has a curation decision, frozen checksum/version, license, domain, event/follow-up definition, group structure, and related-task declaration.
2. The 12–18 task suite spans at least four domains and relevant size/dimensionality/censoring regimes without counting related cohort variants as independent inference units.
3. The balanced core roster has capability-verified default/tuned arm semantics and explicit comparable compute/refit policies.
4. HPO selects the declared censoring-aware primary score without outer-test access, and Uno C remains a separate co-primary dimension.
5. Sentinel runs across representative model families meet coverage, validity, determinism, budget, OOF-ensemble, storage, and report-regeneration gates.

**Plans:** TBD during `$gsd-plan-phase 5`.

### Phase 6: Release Reproduction

**Goal:** Demonstrate that `survbench-1.0-rc1` can be built, executed, regenerated, and independently reproduced from a clean frozen environment.

**Depends on:** Phase 5.

**Requirements:** REPR-01, REPR-02, REPR-03, REPR-04, REPR-05, REPR-06

**Success Criteria:**

1. CI builds the hashed-lock, digest-pinned Linux/amd64 environment and installs/tests a non-editable wheel inside it.
2. Pull-request and release pipelines enforce the declared fast/heavy verification tiers, including semantic benchmark and report assertions.
3. Two clean executions reproduce recipe, split, and result identities and satisfy declared metric/determinism tolerances.
4. All release tables, figures, coverage claims, and manifests regenerate only from the new canonical collection.
5. An independent clean runner reproduces at least one complete dataset-method matrix and records the matching or tolerance-valid collection digest.

**Plans:** TBD during `$gsd-plan-phase 6`.

## Release Gates

`survbench-1.0-rc1` is not releasable until all are true:

- All 48 v2.0 requirements are verified complete.
- No known HIGH correctness, leakage, provenance, or reproducibility finding remains open.
- Headline tables use a declared complete common-support dataset block.
- Every configured core cell has attempt evidence and every excluded score has a machine-readable reason.
- Historical pre-fix evidence is absent from the release collection and reports.
- The canonical environment, protocol, datasets, methods, splits, and result collection have stable digests.
- Clean and independent reproduction gates pass.

## Progress

**Execution order:** Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Truthful Green Baseline | v2.0 | 3/3 | Complete    | 2026-09-01 |
| 2. Scientific Comparison Kernel | v2.0 | 1/3 | In Progress|  |
| 3. Execution and Data Safety | v2.0 | 0/TBD | Not started | - |
| 4. Authoritative Protocol and Results | v2.0 | 0/TBD | Not started | - |
| 5. Representative Protocol Pilot | v2.0 | 0/TBD | Not started | - |
| 6. Release Reproduction | v2.0 | 0/TBD | Not started | - |

---
*Roadmap created: 2026-08-31 for milestone v2.0*
