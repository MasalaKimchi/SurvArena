# Roadmap: SurvArena Benchmark Modernization

## Overview

This roadmap delivers a practitioner-trustworthy survival benchmark by sequencing work from deterministic execution foundations to fair dual-mode evaluation, manuscript-grade statistical inference, robust global ranking, and a canonical reproducibility-first artifact contract.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [ ] **Phase 1: Deterministic Execution Foundation** - Lock profile-tier execution and resumable run behavior.
- [ ] **Phase 2: Fair Dual-Mode HPO Governance** - Enforce no-HPO/HPO parity under a uniform budget policy.
- [ ] **Phase 3: Statistical Inference Pipeline** - Produce corrected pairwise statistical outputs for publication-quality claims.
- [ ] **Phase 4: Global Ranking Robustness** - Add complete matchup-derived ELO with trust diagnostics.
- [ ] **Phase 5: Canonical Artifact and Quality Hardening** - Finalize compact artifact contract, schema compatibility, and quality gates.

## Phase Details

### Phase 1: Deterministic Execution Foundation
**Goal**: Users can run deterministic benchmark tiers and safely resume interrupted collections without losing completed work.
**Depends on**: Nothing (first phase)
**Requirements**: EXEC-01, EXEC-04
**Success Criteria** (what must be TRUE):
  1. User can run `smoke`, `standard`, and `manuscript` profiles with deterministic split governance.
  2. User can resume an interrupted collection and preserve already completed model/dataset results.
  3. User can inspect structured failure records for failed runs without discarding successful outputs.
**Plans**: 2 plans
Plans:
- [x] 01-01-PLAN.md — Enforce deterministic profile-tier and split manifest governance.
- [x] 01-02-PLAN.md — Implement strict resume eligibility and failure-preserving retries.

### Phase 2: Fair Dual-Mode HPO Governance
**Goal**: Users can compare no-HPO and HPO results fairly because every model follows one explicit budget policy.
**Depends on**: Phase 1
**Requirements**: EXEC-02, EXEC-03
**Success Criteria** (what must be TRUE):
  1. User can run every selected model in both no-HPO and HPO modes within the same benchmark profile.
  2. User can inspect requested versus realized HPO budget usage per model run.
  3. User can trust that a single uniform budget policy is enforced across all model runs in the collection.
**Plans**: 3 plans
Plans:
- [ ] 02-01-PLAN.md — Define dual-mode parity and budget-governance test contracts.
- [ ] 02-02-PLAN.md — Implement deterministic dual-mode runner and uniform budget telemetry.
- [ ] 02-03-PLAN.md — Enforce parity-gated comparative exports and canonical governance reporting.

### Phase 3: Statistical Inference Pipeline
**Goal**: Users can make statistically valid comparative claims with corrected pairwise outputs and uncertainty context.
**Depends on**: Phase 2
**Requirements**: STAT-01, STAT-02, STAT-03, STAT-04
**Success Criteria** (what must be TRUE):
  1. User can generate full pairwise model comparisons on aligned evaluation units across datasets.
  2. User can view multiplicity-corrected pairwise significance outputs with adjusted p-values.
  3. User can inspect effect sizes and uncertainty intervals for comparative claims.
  4. User can export manuscript-grade statistical summary artifacts suitable for publication review.
**Plans**: TBD

### Phase 4: Global Ranking Robustness
**Goal**: Users can evaluate a global model ranking with explicit robustness checks before trusting leaderboard claims.
**Depends on**: Phase 3
**Requirements**: RANK-01, RANK-02
**Success Criteria** (what must be TRUE):
  1. User can compute a global ELO leaderboard derived from complete pairwise matchup outcomes.
  2. User can inspect ELO robustness diagnostics (coverage and sensitivity checks) tied to the leaderboard output.
  3. User can identify when ranking claims are strong enough to trust based on published diagnostics.
**Plans**: TBD

### Phase 5: Canonical Artifact and Quality Hardening
**Goal**: Users and contributors can rely on one compact, versioned, reproducible artifact while benchmark code quality stays enforceable.
**Depends on**: Phase 4
**Requirements**: RANK-03, QUAL-01, QUAL-02, QUAL-03, QUAL-04
**Success Criteria** (what must be TRUE):
  1. User can save one canonical comprehensive results artifact per experiment collection without redundant duplicates.
  2. User can read and write result artifacts through versioned schemas with compatibility checks.
  3. User can retrieve a reproducibility bundle containing config, seeds, environment, and command lineage for each collection.
  4. Contributor can pass lint/type/test quality gates for touched benchmark code before merge.
  5. Contributor can remove dead code in benchmark pathways without breaking benchmark behavior.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Deterministic Execution Foundation | 0/TBD | Not started | - |
| 2. Fair Dual-Mode HPO Governance | 0/TBD | Not started | - |
| 3. Statistical Inference Pipeline | 0/TBD | Not started | - |
| 4. Global Ranking Robustness | 0/TBD | Not started | - |
| 5. Canonical Artifact and Quality Hardening | 0/TBD | Not started | - |
