# Pitfalls Research

**Domain:** Survival benchmark projects targeting manuscript-grade model comparisons
**Researched:** 2026-04-23
**Confidence:** MEDIUM-HIGH

## Critical Pitfalls

### Pitfall 1: Dataset Leakage Through Split/Preprocessing Violations

**What goes wrong:**
Models "win" because test information leaks into training via global preprocessing, fold leakage, or split reuse. Reported C-index/Brier gains vanish on truly unseen data.

**Why it happens:**
Benchmark pipelines mix data prep and modeling in one pass, or apply `fit_transform` before splitting/cross-validation. In survival settings, leakage is often subtle (feature engineering, imputation, censoring-related transforms).

**How to avoid:**
- Enforce split-first policy: generate train/val/test splits before any fitted transform.
- Require fold-local pipelines (fit on train fold only, transform on val/test fold).
- Store split manifests and hash them; fail the run if split IDs drift.
- Add leakage audits that compare performance with intentionally permuted labels or leakage probes.

**Warning signs:**
- Very high and unusually stable metrics across heterogeneous datasets.
- Performance collapses when rerunning with stricter nested CV.
- Train and test preprocessing statistics are identical in logs.
- No explicit split manifest or transform provenance in artifacts.

**Phase to address:**
**Phase 1 - Protocol and Data Hygiene.** This must be locked before large benchmark execution.

---

### Pitfall 2: Unfair Hyperparameter Optimization Budgets

**What goes wrong:**
Methods are compared under asymmetric search budgets (time, trials, resources, search-space breadth), so ranking reflects optimization spend instead of model quality.

**Why it happens:**
Teams "normalize" by defaults or tool-specific settings, but different frameworks consume budget differently; some silently exceed limits or fail under equivalent constraints.

**How to avoid:**
- Define one explicit budget policy per benchmark track (time, trials, CPU/GPU, parallelism).
- Run both no-HPO and HPO tracks for every model with matched constraints.
- Log realized budget consumption (not just requested limits), including overruns/failures.
- Treat budget violations as invalid runs, not hidden retries.

**Warning signs:**
- Different models use different wall-clock limits or fold counts.
- Missing "actual budget used" columns in result tables.
- HPO track improvements are extreme for only models with larger search spaces.
- Frequent timeout/exceeded-budget events without exclusion rules.

**Phase to address:**
**Phase 2 - Fair Execution and HPO Policy.** Budget contracts should be codified before final runs.

---

### Pitfall 3: Multiple-Comparison Errors in Pairwise Claims

**What goes wrong:**
Dozens/hundreds of pairwise p-values are reported as independent discoveries, inflating false positives and producing "significant" win/loss narratives that do not replicate.

**Why it happens:**
Manuscript workflows prioritize pairwise tables but skip omnibus testing and correction strategy, or rely on post-hoc mean-rank routines without validating assumptions.

**How to avoid:**
- Pre-register statistical analysis plan: omnibus test, post-hoc family, correction method, effect-size reporting.
- Separate confirmatory comparisons (primary hypotheses) from exploratory pairwise scans.
- Report adjusted p-values/confidence intervals and practical effect sizes, not significance flags alone.
- Add automated checks that block export when multiplicity correction metadata is absent.

**Warning signs:**
- Large pairwise matrix with no family-wise/FDR adjustment column.
- Claims based only on "p < 0.05" without effect-size magnitude.
- Significance conclusions changing when algorithm pool changes.
- No distinction between primary and exploratory endpoints.

**Phase to address:**
**Phase 3 - Statistical Inference Framework.** Must be implemented before leaderboard interpretation.

---

### Pitfall 4: ELO Misuse for Global Method Ranking

**What goes wrong:**
A single ELO leaderboard is treated as ground truth even when comparison graph is sparse, schedule-dependent, or non-stationary; rankings become volatile and overinterpreted.

**Why it happens:**
ELO is easy to explain and compresses pairwise outcomes into one number, but benchmark setups violate core assumptions (order effects, sparse pairings, unstable transitivity/reliability).

**How to avoid:**
- Position ELO as secondary summary, not sole decision criterion.
- Report sensitivity analyses across K-factors, match ordering, and bootstrap resamples.
- Require minimum matchup coverage per model before publishing ELO.
- Pair ELO with raw head-to-head win rates and uncertainty intervals.

**Warning signs:**
- Rank swings after small changes in comparison ordering.
- Large confidence overlap but definitive rank claims in text.
- ELO published without matchup matrix coverage diagnostics.
- Disagreement between ELO ranks and direct pairwise outcomes.

**Phase to address:**
**Phase 4 - Ranking Robustness and Reporting.** ELO gates should be enforced before manuscript claims.

---

### Pitfall 5: Reproducibility Gaps (Cannot Recreate the Paper Tables)

**What goes wrong:**
Benchmark outcomes cannot be exactly regenerated because seeds, environments, split manifests, or config versions are incomplete; "same config" reruns disagree.

**Why it happens:**
Pipelines capture results but not full provenance (code SHA, dependency lock, random seeds, hardware/runtime context, exact command lineage).

**How to avoid:**
- Make provenance mandatory in every run artifact: code SHA, config SHA, seed set, dependency lockfile, runtime environment.
- Version datasets and splits as immutable references with checksums.
- Add one-command reproduction scripts for leaderboard and manuscript tables.
- Enforce reproducibility CI that reruns a smoke subset and diffs key outputs.

**Warning signs:**
- Re-run drift with no code/config changes.
- Missing seed registry or dependency lock snapshots.
- "Manual post-processing" steps outside tracked pipeline.
- Tables/figures not traceable to a single experiment manifest.

**Phase to address:**
**Phase 5 - Reproducibility and Provenance.** Must be complete before freeze and release.

---

### Pitfall 6: Artifact Redundancy and Inconsistent Result Copies

**What goes wrong:**
The same metrics/predictions are written to multiple files/formats/locations, drifting over time and inflating storage; users cannot identify canonical truth.

**Why it happens:**
Teams optimize for convenience by exporting many summary variants without a contract for canonical artifacts and derivation lineage.

**How to avoid:**
- Define a canonical artifact contract: one authoritative comprehensive result file per experiment collection.
- Treat all secondary outputs as derived views generated from canonical source only.
- Add schema/version fields and content hashes to detect divergence.
- Deduplicate at export stage and fail on non-canonical duplicate writes.

**Warning signs:**
- Multiple leaderboard files disagree by metric value or model ordering.
- Unclear "source of truth" in docs and CLI help.
- Storage growth disproportionate to number of runs.
- Manual copy/merge scripts for final manuscript tables.

**Phase to address:**
**Phase 5 - Reproducibility and Provenance** (contract) and **Phase 6 - Export/Storage Hardening** (enforcement).

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Reusing one split for fast iteration and final claim | Faster turnaround | Hidden overfitting, inflated claims | Only in exploratory notebooks, never for final benchmark |
| Unbounded HPO retries "until stable" | Better point estimate | Unfair budget usage, irreproducible comparisons | Never for manuscript-grade claims |
| Exporting every intermediate table as a separate canonical file | Easy ad hoc consumption | Conflicting artifacts and storage bloat | Acceptable only if clearly marked as derived/non-canonical |

## Integration Gotchas

Common mistakes when connecting external benchmark components.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Method adapters | Different default preprocessing across adapters | Centralize preprocessing policy and enforce in runner |
| HPO engine + runner | Engine uses per-model hidden defaults for trial limits | Runner injects explicit, uniform budget policy |
| Reporting layer | ELO generated from filtered/incomplete matchup graph | Build ELO only from validated complete matchup snapshot |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Full nested CV + heavy HPO on every method/dataset | Runs miss deadlines; incomplete coverage | Stage tracks (smoke -> standard -> full), strict budget caps | Medium dataset suites with many methods |
| Recomputing derived summaries repeatedly from raw folds | Export step dominates runtime | Cache canonical aggregate once, derive all views from it | When pairwise matrix becomes large |
| Storing per-fold predictions in redundant formats | Disk I/O spikes, huge result directories | Canonical storage + on-demand derived exports | As benchmark count scales across tracks |

## Security Mistakes

Domain-specific security/reliability issues for benchmark artifact integrity.

| Mistake | Risk | Prevention |
|---------|------|------------|
| Mutable artifacts overwritten in-place | Silent tampering/drift in published results | Immutable run directories + content hashes |
| Untracked manual edits to result CSV/JSON | Paper tables do not match executable pipeline | Generated-only publication tables from pipeline |
| No integrity checks on downloaded datasets | Corrupted/changed data alters benchmark outcomes | Dataset checksum verification before run |

## UX Pitfalls

Common practitioner-facing mistakes in benchmark products.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| One-number leaderboard with no uncertainty | Overconfident model selection | Show uncertainty intervals and pairwise context |
| Hidden HPO budget details | Users cannot judge fairness of results | Display requested + realized budget in all summaries |
| Ambiguous artifact naming | Users consume stale/derived files accidentally | Explicit canonical vs derived naming convention |

## "Looks Done But Isn't" Checklist

- [ ] **Leakage controls:** Split manifests, fold-local preprocessing, and leakage audit report are present.
- [ ] **HPO fairness:** Every method has matched budget policy and realized-budget logs.
- [ ] **Statistical claims:** Multiplicity correction and effect-size reporting are included in exported summaries.
- [ ] **ELO robustness:** Sensitivity/coverage diagnostics pass before publishing global rankings.
- [ ] **Reproducibility:** Reproduction script regenerates key leaderboard/manuscript outputs from clean state.
- [ ] **Artifact contract:** Exactly one canonical comprehensive result artifact exists per experiment collection.

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Dataset leakage discovered post-run | HIGH | Invalidate affected runs, regenerate splits, rerun all impacted models, publish correction note |
| Unfair HPO budgets detected after ranking | HIGH | Rebaseline with uniform budget policy, rerun HPO track, archive old ranking as superseded |
| Multiplicity/statistical error in manuscript tables | MEDIUM-HIGH | Recompute tests with predeclared correction, update claims to effect-size-first narrative |
| ELO instability identified late | MEDIUM | Recompute with robustness protocol and downgrade ELO to secondary metric if unstable |
| Reproducibility failure during review | HIGH | Freeze environment, backfill provenance metadata, rerun smoke/full verification |
| Artifact redundancy with conflicting outputs | MEDIUM | Select canonical artifact, regenerate all derived files, enforce duplicate-write guards |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Dataset leakage | Phase 1 - Protocol and Data Hygiene | Leakage audit passes; split hashes stable; fold-local transforms verified |
| Unfair HPO budgets | Phase 2 - Fair Execution and HPO Policy | Requested vs realized budget parity report per method |
| Multiple-comparison errors | Phase 3 - Statistical Inference Framework | Omnibus + corrected post-hoc outputs present with effect sizes |
| ELO misuse | Phase 4 - Ranking Robustness and Reporting | Sensitivity analyses and matchup coverage thresholds satisfied |
| Reproducibility gaps | Phase 5 - Reproducibility and Provenance | Clean-room rerun reproduces key summary tables within tolerance |
| Artifact redundancy | Phase 6 - Export/Storage Hardening | Canonical artifact contract enforced; no divergent duplicate outputs |

## Sources

- [scikit-learn Common Pitfalls (data leakage/preprocessing)](https://scikit-learn.org/stable/common_pitfalls.html) — HIGH
- [Demsar (2006), Statistical Comparisons of Classifiers over Multiple Data Sets](https://jmlr.org/papers/v7/demsar06a.html) — HIGH
- [Benavoli et al. (2015), Should we really use post-hoc tests based on mean-ranks?](https://arxiv.org/abs/1505.02288) — MEDIUM-HIGH
- [Berrar (2022), Using p-values for the comparison of classifiers: pitfalls and alternatives](https://link.springer.com/article/10.1007/s10618-022-00828-1) — HIGH
- [Gijsbers et al. (2024), AMLB: an AutoML Benchmark](https://jmlr.org/papers/v25/22-0493.html) — HIGH
- [Boubdir et al. (2023), Elo Uncovered: Robustness and Best Practices in Language Model Evaluation](https://arxiv.org/abs/2311.17295) — MEDIUM
- [MLPerf Submission Rules (fairness/review governance context)](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc) — MEDIUM

---
*Pitfalls research for: survival benchmark modernization / manuscript-grade comparison system*
*Researched: 2026-04-23*
