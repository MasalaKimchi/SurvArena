# Phase 2: Scientific Comparison Kernel - Context

**Gathered:** 2026-09-01
**Status:** Ready for planning
**Source:** Roadmap requirements plus the user's instruction to move directly into implementation

<domain>
## Phase Boundary

Make the comparison layer scientifically defensible before expanding the benchmark suite: exact matched-cell comparisons, dataset-level inference, complete common support, explicit reliability/eligibility views, independently checked censoring-aware metrics, fixed evaluation-window semantics, and fail-closed prediction validation.

</domain>

<decisions>
## Implementation Decisions

### Comparison population
- **D-01:** Headline ranks, win rates, ratings, and significance use an explicit complete common-support dataset block; conditional-on-success rankings are diagnostic only.
- **D-02:** A comparison cell is identified by benchmark, arm, dataset, split, and seed. Pairing must be an exact key join and must reject duplicate method/cell rows rather than constructing Cartesian comparisons.

### Inference
- **D-03:** The dataset is the experimental sample. Repeated folds/seeds are aggregated within dataset before cross-dataset inference, uncertainty, or multiple-method tests; datasets receive equal weight.

### Metrics and predictions
- **D-04:** Evaluation horizons are fixed once per dataset/protocol run, never recomputed independently inside each fold. Requested horizons, used support, and the IPCW policy are recorded with every result; unsupported metrics are missing with a reason, never silently moved to another horizon.
- **D-05:** Prediction bundles must pass shape, finiteness, time-grid, probability-bound, orientation, and survival-monotonicity checks before full-distribution metrics are computed.
- **D-06:** Harrell C, Uno C, Brier/IBS, cumulative/dynamic AUC, and calibration behavior require independent-reference or independently derived golden tests, including censoring and support-edge fixtures.

### the agent's Discretion
- Internal module boundaries and result-column names, provided they are typed, machine-queryable, compact, and backward-compatible only where that does not weaken the fail-closed contract.
- Exact nonparametric paired test and multiplicity correction, provided dataset count is reported as the inferential sample size and assumptions are explicit.

</decisions>

<canonical_refs>
## Canonical References

### Milestone contract
- `.planning/ROADMAP.md` — Phase 2 goal, success criteria, and downstream ordering.
- `.planning/REQUIREMENTS.md` — STAT-01 through STAT-07 and METR-01 through METR-05.
- `.planning/phases/01-truthful-green-baseline/01-REVIEW.md` — verified baseline boundary and deferred correctness findings.

### Live implementation
- `survarena/evaluation/_ranking.py` — current rank and win-rate logic, including the observed within-dataset Cartesian comparison defect.
- `survarena/evaluation/_significance.py` — current fold-level Wilcoxon/bootstrap/CD logic.
- `survarena/evaluation/_ratings.py` — paired-match Elo implementation and dataset bootstrap.
- `survarena/evaluation/metrics.py` — torchsurv-backed metric implementation and current per-fold horizon clipping.
- `survarena/benchmark/runner.py` — live prediction, metric, and result-record path.

</canonical_refs>

<specifics>
## Specific Ideas

- Use adversarial two-split examples where method A wins split 1 and loses split 2; exact pairing must return two comparisons, never four.
- Use unequal fold counts to prove equal dataset weighting.
- Use an incomplete method/dataset matrix to prove headline support fails closed and a failed method cannot improve rank by disappearing.
- Differential metric tests should use the declared scikit-survival dependency as the independent reference where it exposes the corresponding estimator.

</specifics>

<deferred>
## Deferred Ideas

- Killable process trees, group-safe inner validation, and exact split fingerprints remain Phase 3.
- Making the typed protocol and SQLite collection authoritative remains Phase 4.
- Freezing the final 12–18 dataset suite and release roster remains Phase 5.

</deferred>

---

*Phase: 02-scientific-comparison-kernel*
*Context gathered: 2026-09-01 from verified roadmap and user authorization*
