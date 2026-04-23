# Feature Research

**Domain:** Practitioner-focused survival model benchmark platform (production model selection)
**Researched:** 2026-04-23
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Dataset-balanced benchmark suite with fixed benchmark profiles (`smoke`, `research`, `manuscript`) | Practitioners need quick sanity checks plus progressively stronger evidence modes | MEDIUM | Dependency: standardized dataset metadata + profile-level runtime budgets + split policies |
| Reproducible split governance (persisted splits, manifests, hash checks) | Production decisions require reruns to produce equivalent conclusions | MEDIUM | Dependency: split storage contract and config/dataset fingerprinting across runs |
| No-HPO vs HPO parity for every model in scope | Teams expect to separate intrinsic model quality from tuning-budget advantage | HIGH | Dependency: shared compute budget policy, per-method HPO adapters, and paired reporting keyed by method + tuning mode |
| Multi-metric evaluation beyond C-index (discrimination, calibration, utility, efficiency) | Survival practitioners expect robust assessment under censoring and calibration-sensitive use cases | HIGH | Dependency: consistent metric engine and missing-metric handling policy |
| Pairwise model comparison table with significance-aware outcomes | Model selection needs direct "A vs B" evidence, not only absolute scores | HIGH | Dependency: fold-level aligned records, multiple-comparison correction, and confidence interval exports |
| Compact experiment artifact with complete lineage | Users need one portable output to archive, share, and re-analyze without reconstructing state | HIGH | Dependency: canonical schema, index/manifest metadata, and non-redundant serialization rules |
| CLI + Python API parity for benchmark execution and comparison | Benchmarking must fit both scripted research and production automation workflows | MEDIUM | Dependency: shared run orchestration layer and stable config schema |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| TabArena-style global ranking layer (pairwise battles + ELO with bootstrap CI) | Converts dense multi-dataset outcomes into an interpretable "who wins overall" signal with uncertainty | HIGH | Dependency: pairwise outcome matrix, robust bootstrap pipeline, and leaderboard calibration checks |
| Manuscript-grade robustness bundle (pairwise significance, multiple-comparison summaries, critical-difference outputs) | Supports publication-ready and regulator-facing evidence without custom analysis notebooks | HIGH | Dependency: repeated CV protocol, corrected hypothesis testing, and standardized statistical export schemas |
| Failure-rate and missing-metric transparency surfaced beside quality metrics | Enables safer production selection by exposing fragility, not just central performance | MEDIUM | Dependency: per-run status taxonomy and run-record normalization |
| Full run-record ledger with indexed compact storage for auditability | Keeps provenance, hyperparameter traces, and run context queryable at low storage overhead | MEDIUM | Dependency: append-safe record format + deterministic index generation |
| Practitioner decision views (leaderboard + rank summaries + pairwise risk framing) | Reduces time-to-decision for "which model to ship next" under realistic trade-offs | MEDIUM | Dependency: normalized score/rank features and consistent tie/uncertainty policy |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| "Single winning metric" benchmark (C-index-only or one-number score) | Easier communication and faster rankings | Masks calibration/utility failures and encourages overfitting to one target | Keep a constrained metric panel with explicit primary/secondary metrics and manuscript summaries |
| Unlimited HPO budgets or model-specific custom tuning loopholes | Teams want each model to "look its best" | Breaks fairness and makes rankings budget-dependent rather than method-dependent | Enforce budget parity tiers and report no-HPO + HPO side by side |
| Real-time dashboard-first scope in v1 | Stakeholders want interactive visibility immediately | Diverts effort from benchmark validity, reproducibility, and statistical rigor | Deliver disk-first artifacts + stable schema first; add UI after benchmark contract stabilizes |
| Fragmented per-metric/per-step output files without a canonical bundle | Feels modular and convenient for ad hoc scripts | Creates storage sprawl, schema drift, and hard-to-audit decision trails | Use one comprehensive artifact contract plus derived convenience exports |
| Automatic benchmark expansion to every available dataset/package by default | "More coverage" sounds more authoritative | Explodes runtime, increases maintenance burden, and weakens curation quality | Curate a medium-diverse suite with explicit inclusion criteria and controlled expansion gates |

## Feature Dependencies

```text
Reproducible split governance
    └──requires──> Dataset metadata + manifest hashing

No-HPO vs HPO parity
    └──requires──> Shared budget policy
                       └──requires──> Config schema + method adapter compliance

Pairwise significance analysis
    └──requires──> Fold-level aligned run records
                       └──requires──> Repeated CV protocol

ELO leaderboard with bootstrap CI
    └──requires──> Pairwise outcome matrix
                       └──requires──> Pairwise significance-ready data quality

Compact single-file experiment artifact
    └──requires──> Canonical result schema + index

Dashboard-first productization ──conflicts──> Manuscript-grade benchmark hardening in v1
Unlimited model-specific HPO ──conflicts──> Fair no-HPO vs HPO parity
```

### Dependency Notes

- **No-HPO vs HPO parity requires shared budget policy:** without budget normalization, tuned-vs-untuned comparisons become compute comparisons.
- **ELO leaderboard requires pairwise outcome matrix:** ELO is only meaningful when matchup generation is consistent across datasets and folds.
- **Pairwise significance requires fold-level aligned records:** significance tests are invalid when methods are not compared on matched evaluation units.
- **Compact artifact requires canonical schema + index:** a single file is only practical if it remains queryable and versioned.
- **Dashboard-first conflicts with benchmark hardening:** UI iteration pressure can freeze unstable schemas too early and slow scientific quality work.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [ ] No-HPO and HPO execution parity for selected survival model set — core fairness requirement for practitioner trust
- [ ] Pairwise comparison outputs with corrected significance summaries — enables defensible head-to-head model decisions
- [ ] ELO leaderboard with bootstrap confidence intervals — provides intuitive global ranking with uncertainty bounds
- [ ] Compact comprehensive experiment artifact with manifest/index — enforces portability, auditability, and storage discipline
- [ ] Manuscript-grade statistical export bundle (critical difference, multiple-comparison summary, rank summary) — supports publication-level rigor from a single run

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] Decision-policy presets (e.g., "speed-first", "calibration-first", "robustness-first") — add when users repeatedly request standardized selection lenses
- [ ] Enhanced robustness tracks (dataset shift stress tests, censoring sensitivity sweeps) — add after baseline benchmark throughput is stable
- [ ] Artifact introspection tooling (schema inspector + validation CLI) — add when external teams begin exchanging benchmark bundles

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] Hosted collaboration UI and scheduled benchmark service — defer until benchmark protocol and schemas are operationally stable
- [ ] Cross-language method execution (R/other ecosystems) — defer until Python-first benchmark quality plateaus and maintenance capacity expands
- [ ] Auto-ensemble benchmarking across all methods by default — defer until ranking leakage and budget fairness controls are mature

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| No-HPO vs HPO parity | HIGH | HIGH | P1 |
| Pairwise comparison + corrected significance | HIGH | HIGH | P1 |
| ELO leaderboard + bootstrap CI | HIGH | HIGH | P1 |
| Compact single-file experiment artifact | HIGH | HIGH | P1 |
| Manuscript-grade statistical exports | HIGH | HIGH | P1 |
| Failure/missing-metric transparency | HIGH | MEDIUM | P2 |
| Decision-policy presets | MEDIUM | MEDIUM | P2 |
| Hosted UI/service layer | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | Competitor A | Competitor B | Our Approach |
|---------|--------------|--------------|--------------|
| Pairwise + ELO leaderboard | TabArena uses task-weighted pairwise battles with MLE Elo and bootstrap CI | Typical survival benchmark papers often rely on rank/metric tables without a persistent leaderboard layer | Adopt TabArena-style global ranking adapted to survival metrics and censoring-aware evaluation units |
| Tuned vs default parity | TabArena explicitly studies default, tuned, and ensemble regimes | Many benchmark studies under-specify tuning budgets, reducing comparability | Require explicit no-HPO and HPO tracks for each method under shared budgets |
| Statistical robustness depth | Research papers vary; many over-index on C-index reporting | Broad ML benchmarking literature stresses variance-aware evaluation and replication concerns | Bake corrected pairwise testing, uncertainty reporting, and manuscript-ready summaries into core outputs |
| Result artifact portability | Public leaderboards often expose slices, not always full reproducible run ledgers | Ad hoc benchmark repos frequently spread outputs across many files | Standardize a compact canonical artifact plus derived exports for interoperability |

## Sources

- Internal project context: `/Users/justin/Documents/SurvArena/.planning/PROJECT.md` and `/Users/justin/Documents/SurvArena/docs/protocol.md`
- TabArena paper (living benchmark methodology, pairwise/ELO framing): [arXiv:2506.16791](https://arxiv.org/abs/2506.16791)
- TabArena results page (ELO + bootstrap CI leaderboard presentation): [TabArena Results](https://tabnetics.org/TABARENA_RESULTS.html)
- Survival metric caution against C-index-only evaluation: [arXiv:2506.02075](https://arxiv.org/abs/2506.02075)
- Variance-aware benchmark comparison principles: [arXiv:2103.03098](https://arxiv.org/abs/2103.03098), [MLSys 2021 abstract](https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html)

---
*Feature research for: comprehensive survival benchmark platform for practitioner model selection*
*Researched: 2026-04-23*
