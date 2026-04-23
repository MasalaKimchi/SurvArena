# Requirements: SurvArena Benchmark Modernization

**Defined:** 2026-04-23
**Core Value:** A practitioner can trust one benchmark run to produce fair, statistically robust, and compactly stored model comparisons across diverse survival datasets.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Benchmark Execution

- [ ] **EXEC-01**: User can run benchmark profile tiers (`smoke`, `standard`, `manuscript`) with deterministic split governance.
- [ ] **EXEC-02**: User can run every selected model in both no-HPO and HPO modes under the same benchmark profile.
- [ ] **EXEC-03**: User can enforce a uniform HPO budget policy and inspect realized budget usage per model run.
- [ ] **EXEC-04**: User can resume interrupted benchmark runs with structured failure records instead of losing completed progress.

### Statistical Inference

- [ ] **STAT-01**: User can generate full pairwise model comparisons on aligned evaluation units across datasets.
- [ ] **STAT-02**: User can view multiplicity-corrected significance results (adjusted p-values) for pairwise comparisons.
- [ ] **STAT-03**: User can inspect effect sizes and uncertainty intervals for comparative claims.
- [ ] **STAT-04**: User can export manuscript-grade statistical summary artifacts suitable for publication review.

### Ranking and Results

- [ ] **RANK-01**: User can compute a global ELO leaderboard derived from complete pairwise matchup outcomes.
- [ ] **RANK-02**: User can inspect ELO robustness diagnostics (coverage and sensitivity checks) before trusting rank claims.
- [ ] **RANK-03**: User can save one canonical comprehensive results artifact per experiment collection without redundant duplicate outputs.

### Reproducibility and Quality

- [ ] **QUAL-01**: Contributor can pass lint/type/test quality gates for touched benchmark code before merge.
- [ ] **QUAL-02**: Contributor can remove dead functions/dead code in benchmark pathways without breaking behavior.
- [ ] **QUAL-03**: User can rely on versioned result schemas with compatibility checks for artifact readers/writers.
- [ ] **QUAL-04**: User can retrieve a reproducibility bundle containing config, seeds, environment, and command lineage for each benchmark collection.

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Ecosystem Expansion

- **ECO-02**: User can benchmark optional deep/foundation-heavy tracks with dedicated runtime budgets.

### Productization

- **PROD-01**: User can inspect benchmark outputs through an interactive dashboard/service layer.
- **PROD-02**: User can apply predefined decision policies (for example speed-first, calibration-first, robustness-first) directly in report generation.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| R and other non-Python package execution | Project scope is exclusively Python package ecosystems |
| Hosted multi-tenant benchmark service in v1 | Core benchmark validity and artifact contract are higher priority |
| Unlimited or model-specific custom HPO budgets | Breaks fairness and comparability across methods |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| EXEC-01 | Phase 1 | Pending |
| EXEC-02 | Phase 2 | Pending |
| EXEC-03 | Phase 2 | Pending |
| EXEC-04 | Phase 1 | Pending |
| STAT-01 | Phase 3 | Pending |
| STAT-02 | Phase 3 | Pending |
| STAT-03 | Phase 3 | Pending |
| STAT-04 | Phase 3 | Pending |
| RANK-01 | Phase 4 | Pending |
| RANK-02 | Phase 4 | Pending |
| RANK-03 | Phase 5 | Pending |
| QUAL-01 | Phase 5 | Pending |
| QUAL-02 | Phase 5 | Pending |
| QUAL-03 | Phase 5 | Pending |
| QUAL-04 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-23*
*Last updated: 2026-04-23 after roadmap creation*
