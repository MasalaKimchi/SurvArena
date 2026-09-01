# Phase 2: Scientific Comparison Kernel - Research

**Researched:** 2026-09-01
**Domain:** Repeated-resampling survival benchmark statistics and censoring-aware evaluation
**Confidence:** HIGH for code-path findings; MEDIUM-HIGH for the release policy until the Phase 5 protocol is frozen

## Summary

The current dirty worktree improves eligibility filtering, two-sided pair reporting, Nemenyi constants, and Elo bootstrap identity, but it is not yet safe for citable claims. `pairwise_win_rate` compares every score from one method with every score from the other method inside a dataset, so two matched folds become four comparisons. `add_dataset_ranks` ranks pooled method-fold rows, and `pairwise_significance` feeds fold/seed deltas directly to Wilcoxon. Those choices inflate the apparent sample size and make uncertainty depend on the number of resamples rather than the number of independent datasets.

The corrective architecture is a single comparison-population builder used by ranks, wins, Elo, bootstrap, and significance. It validates unique cell identities, retains reliability outcomes separately, derives exact matched method cells, and selects a complete common-support dataset block. Repeated cells are summarized within dataset, after which each dataset contributes one effect/rank/match. This makes the reported inferential sample size honest and prevents failure-driven survivorship bias.

Metric verification needs a second fail-closed boundary. Survival predictions must be validated before interpolation/scoring. Horizons must remain fixed for a dataset/protocol run; the current fold-specific event quantiles and silent clipping change the estimand across folds. The existing scikit-survival dependency provides an independent implementation for concordance, Brier/IBS, and cumulative/dynamic AUC differential tests. Calibration requires an independent numerical golden check, and D-calibration must be emitted only for valid full survival distributions.

**Primary recommendation:** Centralize comparison support and prediction validation first, then route every statistical/metric output through those contracts.

## Architectural Responsibility Map

| Capability | Primary module | Consumer | Rationale |
|---|---|---|---|
| Cell identity/common support | `survarena/evaluation/_comparison.py` | ranking, ratings, significance, reports | One population definition prevents divergent headline cohorts. |
| Eligibility/reliability | `survarena/evaluation/_eligibility.py` | all statistical views | Performance and failure views remain separately queryable. |
| Dataset-level ranks/wins | `survarena/evaluation/_ranking.py` | export/report layer | Exact joins and within-dataset aggregation belong together. |
| Dataset-level inference | `survarena/evaluation/_significance.py` | export/report layer | Wilcoxon/bootstrap/CD receive one observation per dataset. |
| Prediction contract | `survarena/evaluation/predictions.py` | runner and public predictor | Invalid outputs are stopped before metric-specific code. |
| Metric support metadata | `survarena/evaluation/metrics.py` | runner/result schema | Requested horizon and IPCW support define the estimand. |

## Existing Stack

No new dependency is required. NumPy/pandas implement deterministic transforms; SciPy supplies Wilcoxon, chi-square, and numerical reference checks; scikit-survival is the independent metric reference; torchsurv remains the production metric backend until differential tests justify it.

## Required Correctness Invariants

1. Natural cell key: `benchmark_id`, optional `hpo_mode`, `dataset_id`, optional `split_id`, optional `seed`, plus robustness/scenario identity when present.
2. At most one row per method and natural cell. Duplicate rows are an error, not an implicit average or many-to-many merge.
3. Pairwise comparisons inner-join exact natural cells. Missing counterparts reduce support and are reported.
4. Headline multi-method analyses operate on datasets where every required method has the complete expected cell set.
5. Repeated cells are aggregated to one method score or paired delta per dataset before inference.
6. The point estimate and bootstrap use equal dataset weights; folds/seeds only reduce within-dataset measurement noise.
7. Unsupported horizons remain the requested horizon with an ineligibility reason; they are never replaced with a clipped time while keeping the same column label.
8. Full-distribution metrics require finite, bounded, non-increasing survival curves on a finite strictly increasing grid with exact row alignment.

## Failure Modes to Test

- Two-by-two cross-product pairing and duplicated natural keys.
- A method missing only its difficult dataset and therefore appearing best conditionally.
- Unequal fold counts causing a large dataset to dominate mean/CI.
- Incomplete Friedman/Nemenyi blocks.
- Identical all-zero paired deltas and fewer than two/three datasets.
- NaN/Inf risk, transposed survival arrays, non-monotone survival, invalid probabilities, duplicate/descending time grids.
- Evaluation horizons outside censoring support, no training events, all-censored validation, and tied times.

## Validation Architecture

### Fast task tests

- `tests/test_evaluation.py`: constructed pairing, common-support, equal-weight, incomplete-block, prediction-contract, and differential metric fixtures.
- `tests/test_benchmark.py`: runner records fixed horizons/support/eligibility and rejects invalid bundles.

### Independent references

- `sksurv.metrics.concordance_index_censored`
- `sksurv.metrics.concordance_index_ipcw`
- `sksurv.metrics.brier_score` / `integrated_brier_score`
- `sksurv.metrics.cumulative_dynamic_auc`
- Independent weighted-logistic calibration optimum using SciPy numerical optimization.

### Phase gate

Ruff, bounded mypy, full pytest, compileall, constructed statistical counterexamples, and a clean WHAS500/CoxPH semantic smoke. No benchmark matrix regeneration occurs in this phase.

## Out of Scope

Process termination, group-safe inner validation, canonical-store conflict handling, final dataset curation, and clean-container reproduction remain Phases 3–6.

---

*Phase: 02-scientific-comparison-kernel*
*Research completed: 2026-09-01 (inline researcher fallback)*
