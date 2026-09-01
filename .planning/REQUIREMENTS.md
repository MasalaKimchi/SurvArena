# Requirements: SurvArena Benchmark Modernization

**Defined:** 2026-08-31
**Milestone:** v2.0 Verified Benchmark Kernel
**Release target:** `survbench-1.0-rc1`
**Core Value:** A practitioner can trust one benchmark run to produce fair, statistically robust, provenance-complete, and compactly stored model comparisons across a representative survival dataset suite.

## v2.0 Requirements

### Truthful Baseline

- [x] **BASE-01**: A contributor can run the declared lint, type, unit, integration, and core end-to-end checks with zero failures in the supported development environment.
- [x] **BASE-02**: A maintainer can run the strict manuscript/release audit using only declared project dependencies and receive an actionable pass/fail report rather than an import/rendering crash.
- [x] **BASE-03**: A user encountering a legacy split cache receives an explicit compatibility report and can deliberately migrate or regenerate it without silent split changes.
- [x] **BASE-04**: A reader can distinguish current verified behavior, historical invalidated evidence, and remaining release blockers from the maintained project/status documentation.

### Statistical Comparison

- [x] **STAT-01**: A user receives pairwise win rates computed only from exactly matched dataset, split, seed, and comparison-arm cells.
- [x] **STAT-02**: A user receives method ranks on a method-within-dataset or method-within-matched-split scale, never a pooled fold-times-method scale.
- [ ] **STAT-03**: Cross-dataset pairwise significance tests use one paired aggregate effect per dataset and report the dataset count as the inferential sample size.
- [x] **STAT-04**: Every headline comparison declares and enforces a complete common-support dataset block before methods are ranked or tested.
- [ ] **STAT-05**: Friedman/Nemenyi or replacement post-hoc analysis is only applied when its block assumptions hold; incomplete blocks fail closed or use a documented appropriate method.
- [x] **STAT-06**: A user can inspect attempt coverage, success, invalid prediction, fallback, timeout, and failure rates separately from conditional performance.
- [x] **STAT-07**: Dataset-level effect estimates and uncertainty give equal declared weight to benchmark datasets unless an alternative weighting policy is explicitly versioned.

### Metric and Prediction Verification

- [ ] **METR-01**: Harrell C, Uno C, Brier/IBS, and cumulative/dynamic AUC pass deterministic differential tests against an independent reference implementation across normal and edge-case fixtures.
- [ ] **METR-02**: Every dataset release declares fixed evaluation horizons/windows and an IPCW support policy based on estimable censoring support; used support is recorded per result.
- [ ] **METR-03**: Risk and survival predictions are rejected or marked ineligible when shape, finiteness, time grid, orientation, probability bounds, or survival monotonicity contracts fail.
- [ ] **METR-04**: Any calibration metric included in the release has an independent reference/golden test and explicitly supports censored observations; D-calibration is available for full-distribution-capable methods.
- [ ] **METR-05**: A method is only scored on metrics supported by its declared prediction capabilities; single-horizon or fallback outputs cannot silently enter full-distribution headline metrics.

### Execution and Data Safety

- [ ] **EXEC-01**: A hanging or over-budget model fit runs in a disposable process whose entire process tree is terminated within a bounded grace period.
- [ ] **EXEC-02**: Execution timeout, HPO budget, adapter-native time limit, memory limit, benchmark worker count, and native thread count are separate validated protocol fields.
- [ ] **EXEC-03**: Split generation, HPO sampling, model initialization, and robustness perturbation use distinct recorded RNG domains while preserving paired method comparisons.
- [ ] **EXEC-04**: Fit time, prediction time, CPU use, and peak RSS are measured for the individual run unit rather than inherited from a reused worker's lifetime.
- [ ] **EXEC-05**: Parallel execution avoids repeatedly serializing a complete wide dataset for every method/split cell and prevents nested all-core oversubscription.
- [ ] **DATA-01**: Dataset groups and stable row identifiers are stored separately from feature columns and cannot be passed to model preprocessing as predictors.
- [ ] **DATA-02**: Outer CV, inner HPO CV, and early-stopping validation are group-disjoint whenever a dataset declares groups.
- [ ] **DATA-03**: Split manifests include dataset, target, group, row-order, split-algorithm, and geometry fingerprints plus hard minimum event/censoring checks.
- [ ] **MODL-01**: Every maintained adapter declares validated capabilities for validation use, early stopping, full-refit policy, prediction outputs, determinism, and supported metrics.

### Protocol and Canonical Results

- [ ] **CORE-01**: The live benchmark runner compiles every configuration through one strict `ProtocolSpec` and rejects ambiguous booleans, unknown semantics, duplicate rosters, or incoherent arm/HPO/budget settings before execution.
- [ ] **CORE-02**: Dataset and method references resolve to versioned specifications whose normalized content contributes to a stable release-recipe digest.
- [ ] **CORE-03**: The runner consumes the typed model capability contract rather than parallel legacy flags or method-name policy branches.
- [ ] **STORE-01**: A run identity includes protocol recipe, code revision, dataset version/checksum, method/weight version, split digest, arm, seed, environment, and relevant hardware tier.
- [ ] **STORE-02**: Re-appending an identical result is idempotent, while a materially different payload under the same identity is rejected as a provenance conflict rather than overwritten.
- [ ] **STORE-03**: One SQLite result collection is the comprehensive canonical artifact for an experiment collection; CSV, Parquet, tables, and figures are derived views.
- [ ] **STORE-04**: Resume, retry, merge, and report regeneration operate from canonical store state and preserve both success and failure evidence.
- [ ] **STORE-05**: Every generated report records the source collection digest, schema version, query/support policy, and release recipe.

### Representative Protocol

- [ ] **SUIT-01**: A maintainer can apply a predeclared curation policy covering leakage, duplicates, event definition, follow-up, censoring, groups, licenses, provenance, and task redundancy.
- [ ] **SUIT-02**: The core release contains 12–18 frozen, checksummed tasks spanning at least four application domains and multiple sample-size, dimensionality, and censoring regimes.
- [ ] **SUIT-03**: Variants derived from the same cohort are identified as related tasks and cannot silently count as independent datasets in inference.
- [ ] **PROT-01**: The release declares a balanced 12–16 method core roster with baseline, classical, tree, boosting, deep, and eligible foundation/AutoML coverage; extended methods are reported separately.
- [ ] **PROT-02**: Default and tuned arms have explicit parity semantics, method eligibility, validation/refit behavior, and comparable versioned compute budgets.
- [ ] **PROT-03**: Inner validation can select hyperparameters using the declared censoring-aware primary score, proposed as IBS, without accessing outer-test outcomes.
- [ ] **PROT-04**: Uno C is reported as the co-primary discrimination score, while calibration, runtime, memory, and reliability remain separate decision dimensions.
- [ ] **PROT-05**: A portfolio ensemble is selected only from cached out-of-fold predictions and never uses outer-test labels for member or weight selection.
- [ ] **PROT-06**: A sentinel pilot across representative datasets and model families meets prediction-validity, coverage, budget, determinism, and report-regeneration gates before the full reference matrix begins.

### Reproducibility and Release

- [ ] **REPR-01**: The canonical Linux/amd64 environment is defined by a fully resolved hashed dependency lock and an OCI base/image digest.
- [ ] **REPR-02**: CI builds and installs a non-editable wheel in a clean environment, builds the canonical container, and runs semantic smoke assertions inside it.
- [ ] **REPR-03**: Pull-request and release CI enforce lint, static types, unit, property, differential, integration, store migration, packaging, container, publishability, and sentinel E2E gates at appropriate tiers.
- [ ] **REPR-04**: Two clean executions of the same recipe reproduce protocol/split/result identities and metrics within declared deterministic or numerical tolerances.
- [ ] **REPR-05**: All citable tables, figures, coverage statements, and manifests are regenerated from the release collection after behavior-changing fixes, with no historical artifact mixed into the release.
- [ ] **REPR-06**: An independent clean-machine or CI-shard reproduction verifies at least one complete dataset-method matrix and records the resulting collection digest.

## Future Requirements

### Living Platform

- **LIVE-01**: Users can browse a public versioned leaderboard generated from validated result collections.
- **LIVE-02**: External contributors can validate and submit model shards without editing the core repository.
- **LIVE-03**: Maintainers can publish DOI-backed protocol/dataset releases with governance, deprecation, and review policies.
- **LIVE-04**: Hardware-specific community tiers can contribute results without being conflated with the canonical CPU core tier.

### Extended Survival Tasks

- **TASK-01**: Users can benchmark competing-risk outcomes under a separately versioned protocol.
- **TASK-02**: Users can benchmark time-varying covariates, recurrent events, or multi-state outcomes under separate task contracts.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Non-Python model ecosystems | Explicit project ecosystem constraint. |
| Public leaderboard/submission service in v2.0 | Scientific and provenance contracts must be stable first. |
| Unbounded 30–50 dataset launch matrix | Conflicts with the medium balanced suite and wall-clock constraints. |
| Generic cross-dataset clinical decision-curve headline | Threshold/horizon utility is application-specific. |
| Conditional-on-success headline ranking | Creates survivorship bias. |
| Automatic silent migration of legacy result/split evidence | Citable identities must change explicitly when semantics change. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BASE-01 | Phase 1 | Complete |
| BASE-02 | Phase 1 | Complete |
| BASE-03 | Phase 1 | Complete |
| BASE-04 | Phase 1 | Complete |
| STAT-01 | Phase 2 | Complete |
| STAT-02 | Phase 2 | Complete |
| STAT-03 | Phase 2 | Pending |
| STAT-04 | Phase 2 | Complete |
| STAT-05 | Phase 2 | Pending |
| STAT-06 | Phase 2 | Complete |
| STAT-07 | Phase 2 | Complete |
| METR-01 | Phase 2 | Pending |
| METR-02 | Phase 2 | Pending |
| METR-03 | Phase 2 | Pending |
| METR-04 | Phase 2 | Pending |
| METR-05 | Phase 2 | Pending |
| EXEC-01 | Phase 3 | Pending |
| EXEC-02 | Phase 3 | Pending |
| EXEC-03 | Phase 3 | Pending |
| EXEC-04 | Phase 3 | Pending |
| EXEC-05 | Phase 3 | Pending |
| DATA-01 | Phase 3 | Pending |
| DATA-02 | Phase 3 | Pending |
| DATA-03 | Phase 3 | Pending |
| MODL-01 | Phase 3 | Pending |
| CORE-01 | Phase 4 | Pending |
| CORE-02 | Phase 4 | Pending |
| CORE-03 | Phase 4 | Pending |
| STORE-01 | Phase 4 | Pending |
| STORE-02 | Phase 4 | Pending |
| STORE-03 | Phase 4 | Pending |
| STORE-04 | Phase 4 | Pending |
| STORE-05 | Phase 4 | Pending |
| SUIT-01 | Phase 5 | Pending |
| SUIT-02 | Phase 5 | Pending |
| SUIT-03 | Phase 5 | Pending |
| PROT-01 | Phase 5 | Pending |
| PROT-02 | Phase 5 | Pending |
| PROT-03 | Phase 5 | Pending |
| PROT-04 | Phase 5 | Pending |
| PROT-05 | Phase 5 | Pending |
| PROT-06 | Phase 5 | Pending |
| REPR-01 | Phase 6 | Pending |
| REPR-02 | Phase 6 | Pending |
| REPR-03 | Phase 6 | Pending |
| REPR-04 | Phase 6 | Pending |
| REPR-05 | Phase 6 | Pending |
| REPR-06 | Phase 6 | Pending |

**Coverage:**

- v2.0 requirements: 48 total
- Mapped to phases: 48
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-31*
*Last updated: 2026-09-01 after Phase 1 verification*
