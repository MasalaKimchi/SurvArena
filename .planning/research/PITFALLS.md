# Pitfalls Research

**Domain:** Reproducible tabular survival-analysis benchmarking
**Researched:** 2026-08-31
**Confidence:** HIGH

## Critical Pitfalls

### 1. Fold-Level Pseudo-Replication

**What goes wrong:** Correlated repeated-CV rows are treated as independent evidence, shrinking p-values and overstating effective sample size.

**Avoidance:** Pair exact split cells, aggregate to one method score/effect per dataset, and run cross-dataset inference on datasets. Enforce complete blocks for Friedman/Nemenyi or use an explicitly appropriate incomplete-block method.

**Warning signs:** `n_pairs` equals datasets x folds x repeats; ranks span folds x methods; pairwise code creates score matrices instead of joins.

**Phase:** Scientific comparison kernel.

### 2. Survivorship Bias from Dropped Failures

**What goes wrong:** A method fails on difficult datasets and ranks highly on the easy subset that remains.

**Avoidance:** Separate attempt coverage, reliability, and common-support performance. Require declared coverage for headline eligibility.

**Warning signs:** Different method-specific dataset counts in one headline table; failed rows disappear before support is computed.

**Phase:** Scientific comparison kernel.

### 3. Partial Group Leakage

**What goes wrong:** Outer CV is group-disjoint while inner HPO or early stopping reintroduces related subjects; group IDs may also remain predictive features.

**Avoidance:** Store groups separately from `X`, apply group constraints at every validation layer, fingerprint groups and split algorithm version, and test disjointness end to end.

**Warning signs:** `StratifiedKFold` or `train_test_split` appears after grouped outer splitting; `group_col` is selected from `X` without being dropped.

**Phase:** Execution/data contracts.

### 4. Fake Timeouts

**What goes wrong:** A timed-out thread continues fitting, contaminates later resource usage, and violates fair wall-clock limits.

**Avoidance:** Use an externally supervised subprocess and terminate the process tree on timeout/memory breach. Separate benchmark, HPO, and adapter budgets.

**Warning signs:** `ThreadPoolExecutor.result(timeout=...)`, `shutdown(wait=False)`, background native threads, identical process-lifetime peak memory values.

**Phase:** Execution/data contracts.

### 5. Provenance-Colliding Identities

**What goes wrong:** Changed code, data, method version, or environment overwrites an earlier row because identity excludes provenance.

**Avoidance:** Content-address the complete recipe and reject conflicting payloads. Preserve records; use explicit supersession instead of last-write-wins.

**Warning signs:** `run_id` includes only protocol/dataset/method/split/seed; append replaces rows without payload comparison.

**Phase:** Protocol and canonical storage.

### 6. Horizon and IPCW Semantic Drift

**What goes wrong:** Identically named metrics represent different absolute horizons, or IPCW is evaluated where censoring survival is unsupported.

**Avoidance:** Freeze dataset-level windows, check censoring KM support, store used support, and maintain differential tests against a second implementation.

**Warning signs:** Horizons are independently derived in each fold; estimability uses maximum event time alone; exceptions become silent NaNs.

**Phase:** Scientific comparison kernel.

### 7. Classification Adapter Presented as a Survival Distribution

**What goes wrong:** A single-horizon classifier is transformed into a full survival curve without a validated probabilistic model, yielding misleading IBS/calibration.

**Avoidance:** Use IPCW-aware multi-horizon/discrete-hazard learning or limit the adapter's declared metric capabilities.

**Warning signs:** Censored-before-horizon rows are dropped; one probability is reused as a proportional-hazards score for a full curve.

**Phase:** Protocol/model validation.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Loose YAML dictionaries in orchestration | Fast additions | Inconsistent defaults and unvalidated semantics | Only behind immediate strict compilation. |
| CSV bundle as canonical evidence | Easy inspection | Duplication, weak identity, non-transactional resume/reporting | Derived export only. |
| Silent legacy cache regeneration | Convenience | Split identity changes without a release event | Never for citable runs. |
| Broad roster before capability verification | Impressive coverage | Uneven fairness and invalid metrics | Experimental appendix only. |
| Process pool with full dataset per unit | Minimal scheduler refactor | Serialization and memory blow-up | Small sentinel fixtures only. |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Nested parallelism | CPU oversubscription and unstable runtime | Set native thread caps per worker | `n_jobs > 1` plus all-core models. |
| Repeated DataFrame pickling | High parent CPU/RSS before fitting | Dataset worker context + index-only units | Wide genomics data and many methods. |
| Process-lifetime RSS | Same peak memory across later folds | Sample child process per run | Any reused worker. |
| Recomputing predictions for reports | Slow, non-reproducible reports | Cache/store predictions or verified digests | Multi-metric/ensemble reporting. |

## "Looks Done But Isn't" Checklist

- [ ] **Metric verified:** Reference agreement includes censoring, ties, edge support, and invalid inputs.
- [ ] **Group-aware:** Outer, inner, early-stop, preprocessing, and features respect grouping.
- [ ] **Timed out:** Child process and descendants are gone and resources are reclaimed.
- [ ] **Reproducible:** Clean wheel/container replay yields the same recipe/split digests and bounded metric differences.
- [ ] **Complete leaderboard:** Published comparisons use an explicit shared dataset block and disclose coverage.
- [ ] **Canonical storage:** Every report derives from one collection; no directory search selects evidence.
- [ ] **Foundation method:** Weight revision, training regime, fallback state, and output semantics are recorded.

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Pseudo-replication | Statistical kernel | Constructed counterexamples and dataset-unit assertions. |
| Survivorship bias | Statistical kernel | Failure-on-hard-dataset fixture cannot improve headline eligibility. |
| Group leakage | Execution/data contracts | Groups are disjoint at all layers and absent from `X`. |
| Fake timeouts | Execution/data contracts | Hanging child is terminated within bounded grace. |
| Provenance collision | Protocol/storage | Changed recipe produces new identity or explicit conflict. |
| Horizon/IPCW drift | Metric protocol | Frozen support manifest plus differential tests. |
| Invalid survival adapter | Model validation | Capability gate prevents unsupported IBS/calibration claims. |

## Sources

- SurvArena source audit and executed counterexamples, 2026-08-31.
- https://www.jmlr.org/papers/v7/demsar06a.html — cross-dataset statistical comparisons.
- https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.concordance_index_ipcw.html — IPCW support assumptions.
- https://jmlr.csail.mit.edu/papers/v21/18-772.html — survival-distribution evaluation and calibration.

---
*Pitfalls research for: SurvArena v2 verified benchmark kernel*
*Researched: 2026-08-31*
