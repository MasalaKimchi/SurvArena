# SurvArena Benchmark Modernization

## What This Is

SurvArena is a Python benchmark toolkit for comparing single-event, right-censored tabular survival methods under shared, reproducible protocols. It is being modernized from a manuscript-oriented collection of benchmark scripts into a verified benchmark kernel that practitioners and researchers can use to make defensible model comparisons under realistic compute constraints.

## Core Value

A practitioner can trust one benchmark run to produce fair, statistically robust, provenance-complete, and compactly stored model comparisons across a representative survival dataset suite.

## Current Milestone: v2.0 Verified Benchmark Kernel

**Goal:** Ship the first citable `survbench-1.0-rc1` protocol whose metrics, statistical comparisons, execution isolation, dataset provenance, and result artifacts are independently verifiable.

**Target features:**
- A scientifically correct comparison kernel using matched cells, dataset-level inference, complete common support, and explicit reliability reporting.
- A typed protocol and capability contract that is authoritative across splitting, tuning, fitting, prediction validation, metric computation, and export.
- A killable, resource-bounded execution path that produces one immutable, provenance-complete result collection.
- A frozen medium-sized dataset suite and balanced core model roster with default and tuned comparison arms.
- Automated release gates covering numerical reference agreement, leakage, determinism, packaging, containers, end-to-end execution, and report regeneration.

## Requirements

### Validated

- ✓ Users can define benchmark datasets, methods, seeds, folds, repeats, metrics, and HPO settings in YAML-backed configurations — existing v0.1 behavior.
- ✓ Users can evaluate classical, tree, boosting, deep-learning, foundation, and AutoML-backed survival adapters through a shared fit/predict interface — existing v0.1 behavior.
- ✓ Users can reuse persisted outer splits and resume or retry benchmark work — existing v0.1 behavior.
- ✓ Users can compute Harrell C, Uno C, IPCW Brier/IBS, and cumulative/dynamic AUC; deterministic differential fixtures agree with scikit-survival to numerical tolerance — verified during the 2026-08-31 audit.
- ✓ Users can export fold-level metrics, manifests, coverage summaries, ranks, significance tables, Elo-style ratings, and manuscript reports — existing v0.1 behavior, pending v2 correctness changes.
- ✓ Statistical comparisons use exact matched cells, complete common support, dataset-level weighting/inference, and fail-closed post-hoc assumptions — Phase 2 verified 2026-09-01.
- ✓ Prediction bundles are validated before scoring; fixed dataset horizons, censoring support, metric eligibility, and independent metric references are executable contracts — Phase 2 verified 2026-09-01.

### Active

- [ ] Group identifiers are structurally separated from features and honored by outer CV, inner CV, and early-stopping validation.
- [ ] Every fit runs under a killable, explicit resource budget with separate execution, HPO, and adapter-native limits.
- [ ] The typed protocol and model capability contracts are the runner's single source of truth.
- [ ] The canonical result collection is immutable or conflict-rejecting and includes code, environment, config, data, split, method, and recipe provenance.
- [ ] Dataset releases are frozen, versioned, checksummed, licensed, and accompanied by event/follow-up/group metadata.
- [ ] The core protocol compares balanced default and tuned arms, tunes the declared primary metric without leakage, and includes a cheap out-of-fold ensemble baseline.
- [ ] A clean locked container can reproduce the reference run and regenerate all published artifacts from the canonical result collection.

### Out of Scope

- Competing risks, recurrent events, multi-state outcomes, and time-varying covariates — v2 remains deliberately scoped to single-event right-censored tabular survival.
- A public leaderboard, community submission service, DOI automation, and long-term governance platform — deferred until `survbench-1.0-rc1` is scientifically and operationally verified.
- Non-Python model ecosystems — excluded by the project ecosystem constraint.
- An unbounded 30–50 dataset launch matrix — v2 targets a medium balanced suite that fits the available wall-clock budget.
- Treating conditional-on-success scores as the headline leaderboard — this would reward brittle methods through survivorship bias.

## Context

Phases 1 and 2 are now verified. The clean committed snapshot passes 286 tests with 6 optional skips, while the comparison kernel uses exact cells and datasets as independent units, and the metric path validates predictions with fixed, support-aware estimands. Historical evidence remains invalidated because these behavior-changing corrections have not yet been regenerated under a frozen release recipe.

The next critical path is operational integrity: disposable process trees and per-cell telemetry, explicit thread/memory/worker budgets, structural group/row-ID separation through nested validation, complete split fingerprints, and a conformance-tested model capability contract. Canonical typed protocol/result storage follows in Phase 4.

The authoritative expert audit that initiated this milestone was completed on 2026-08-31. It should be treated as the baseline for requirements and phase planning; previous claims that all ranking/statistical logic was already correct are superseded.

## Constraints

- **Ecosystem:** Python-only package coverage — all maintained methods must run through Python adapters.
- **Runtime budget:** Wall-clock time is the primary operational constraint — protocol breadth, HPO, and repetitions must be tiered and explicitly budgeted.
- **Benchmark scope:** Medium balanced dataset suite — representative enough for external validity without requiring an unbounded compute matrix.
- **Quality gate:** All touched-code lint, type, unit, differential, integration, packaging, and end-to-end checks must pass.
- **Storage contract:** One comprehensive, non-redundant result artifact per experiment collection, with derived reports generated from it.
- **Scientific scope:** Single-event right-censored survival only for this milestone.
- **Evidence:** No historical benchmark result may be cited after a behavior-changing protocol revision unless regenerated under the frozen release recipe.
- **Environment:** Citable results come from a locked canonical Linux/amd64 environment; developer-machine results are diagnostic only.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Software milestone is `SurvArena v2.0`; first citable protocol is `survbench-1.0-rc1` | Separates package evolution from protocol versioning | — Pending |
| Fix scientific correctness before adding datasets or models | Invalid aggregation makes a larger matrix more expensive to regenerate, not more credible | — Pending |
| IBS is the proposed primary score and Uno C the co-primary discrimination score | IBS evaluates full survival distributions; Uno C provides censoring-aware ranking performance | — Pending validation |
| Dataset is the cross-dataset inference unit | Repeated CV folds within one dataset are correlated | ✓ Implemented and verified in Phase 2 |
| Headline tables require complete common support | Prevents methods from benefiting by failing on difficult datasets | ✓ Implemented and verified in Phase 2 |
| Reliability is reported separately from conditional performance | Failures and fallbacks are practitioner-relevant outcomes, not missing data to hide | ✓ Implemented and verified in Phase 2 |
| Public leaderboard is deferred until the benchmark kernel is verified | A living platform should distribute trusted evidence, not amplify invalid evidence | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition:**
1. Requirements invalidated? → Move to Out of Scope with reason.
2. Requirements validated? → Move to Validated with phase reference.
3. New requirements emerged? → Add to Active.
4. Decisions to log? → Add to Key Decisions.
5. "What This Is" still accurate? → Update if drifted.

**After each milestone:**
1. Review all sections.
2. Confirm the Core Value remains the correct priority.
3. Re-audit Out of Scope decisions.
4. Update Context with verified state and evidence.

---
*Last updated: 2026-09-01 after verifying Phase 2*
