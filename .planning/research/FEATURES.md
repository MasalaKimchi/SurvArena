# Feature Research

**Domain:** Reproducible tabular survival-analysis benchmarking
**Researched:** 2026-08-31
**Confidence:** HIGH

## Feature Landscape

### Table Stakes

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Frozen protocol identity | Published numbers must have stable meaning | MEDIUM | Include geometry, seeds, metrics, budgets, roster, eligibility, and recipe digest. |
| Frozen dataset releases | Live upstream loaders cannot anchor a citation | HIGH | Version, checksum, license, event definition, follow-up, grouping, and curation decision. |
| Shared leakage-free validation | Comparison is invalid if splits or preprocessing differ | HIGH | Group constraints apply to outer, inner, and early-stop partitions. |
| Censoring-aware metrics | Survival outcomes are incompletely observed | HIGH | IBS, Uno C, dynamic AUC, and calibration need explicit support policies. |
| Paired dataset-level statistics | Repeated folds are not independent datasets | MEDIUM | Matched cells, common support, dataset-level effects, corrected comparisons. |
| Reliability and coverage reporting | Failed/time-limited methods are operationally important | MEDIUM | Separate from conditional-on-success performance. |
| Killable resource budgets | Runtime is a primary practitioner constraint | HIGH | Distinct fit, HPO, adapter, memory, and thread/process budgets. |
| Canonical provenance-complete store | Results must be compact, queryable, and reproducible | HIGH | Immutable/conflict-rejecting identities and derived reports. |
| Automated verification gates | A living benchmark must detect semantic regressions | HIGH | Reference, property, leakage, packaging, container, E2E, and report gates. |

### Differentiators

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Balanced default versus tuned arms | Shows practical defaults and attainable tuned performance | HIGH | Arm eligibility and budgets must be explicit per method. |
| Full-distribution primary score | Rewards useful survival curves, not ranking alone | MEDIUM | Proposed IBS primary with Uno C co-primary. |
| OOF portfolio ensemble | Exposes complementarity at low marginal training cost | HIGH | Use cached validation predictions without test-set selection. |
| Runtime/quality/reliability Pareto views | Helps practitioners choose under constraints | MEDIUM | Keep hardware tiers separate. |
| Independently reproducible result recipe | Converts a repo into a citable standard | HIGH | Clean image replay and artifact digest comparison. |

### Anti-Features

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Maximal model and dataset count at launch | Looks comprehensive | Multiplies invalid evidence before the kernel is stable | Medium balanced core plus extended tiers. |
| One Elo number over all metrics | Easy leaderboard | Hides score semantics and coverage differences | Metric-specific tables with primary/co-primary. |
| Conditional-on-success headline | Avoids missing values | Rewards methods that fail on hard datasets | Common-support headline plus reliability table. |
| Public submissions before protocol freeze | Creates momentum | Makes validation and compatibility unmanageable | Verify `survbench-1.0-rc1` first. |
| Generic decision-curve leaderboard | Appears clinically actionable | Thresholds/horizons are dataset-specific | Dataset-specific diagnostic only. |

## Feature Dependencies

```text
Correct matched statistics
    └──requires──> Explicit cell identity and common-support policy

Typed protocol authority
    ├──requires──> Strict config validation
    ├──enables──> Reproducible run recipe
    └──enables──> Canonical result identity

Tuned + ensemble headline
    ├──requires──> Leakage-free inner validation
    ├──requires──> IBS selection support
    └──requires──> Cached OOF predictions

Citable release
    ├──requires──> Frozen data + protocol + environment
    ├──requires──> Canonical result store
    └──requires──> Automated verification gates
```

## MVP Definition

### Launch With (`survbench-1.0-rc1`)

- [ ] Correct matched rankings and dataset-level inference.
- [ ] Complete common-support and reliability reporting.
- [ ] Group-safe split hierarchy and validated predictions.
- [ ] Killable resource-bounded execution.
- [ ] Authoritative typed protocol/capability contracts.
- [ ] Immutable provenance-complete result collection.
- [ ] Frozen medium dataset suite and balanced core roster.
- [ ] Default and tuned arms with IBS/Uno validation.
- [ ] Locked container and clean reproduction gates.

### Add After Validation

- [ ] OOF portfolio ensemble after prediction caching and primary-score selection are stable.
- [ ] Runtime/quality interactive explorer after the result schema is frozen.
- [ ] Additional frontier model tier after core coverage is complete.

### Future Consideration

- [ ] Public leaderboard and submission validation service.
- [ ] DOI-backed dataset/protocol releases and governance automation.
- [ ] Competing-risk or time-varying-covariate protocols.

## Feature Prioritization Matrix

| Feature | User Value | Cost | Priority |
|---------|------------|------|----------|
| Statistical correctness/common support | HIGH | MEDIUM | P1 |
| Group-safe validation and killable execution | HIGH | HIGH | P1 |
| Typed protocol and canonical store integration | HIGH | HIGH | P1 |
| Frozen dataset release | HIGH | HIGH | P1 |
| Tuned IBS arm | HIGH | HIGH | P1 |
| OOF ensemble | MEDIUM | HIGH | P2 |
| Public leaderboard | MEDIUM | HIGH | P3 |

## Sources

- https://arxiv.org/abs/2506.16791 — living benchmark, curation, validation, time budgets, and ensembling.
- https://www.jmlr.org/papers/v7/demsar06a.html — statistical comparison over multiple datasets.
- https://jmlr.csail.mit.edu/papers/v21/18-772.html — evaluation of individual survival distributions and D-calibration.
- SurvArena repository and 2026-08-31 verification audit.

---
*Feature research for: SurvArena v2 verified benchmark kernel*
*Researched: 2026-08-31*
