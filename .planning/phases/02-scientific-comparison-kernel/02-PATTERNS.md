# Phase 2: Scientific Comparison Kernel - Pattern Map

**Mapped:** 2026-09-01
**Scope:** Comparison populations, statistical aggregation, prediction contracts, and metric verification

## Pattern Assignments

| Changed responsibility | Closest analog | Preserve | Corrective deviation |
|---|---|---|---|
| Common support/eligibility | `survarena/evaluation/_eligibility.py`, `survarena/core/results/store.py::eligible_runs` | Small pure pandas transforms; explicit success/ineligibility filters | Add uniqueness, exact cell-key, finite-metric, roster, and complete-support contracts. |
| Matched wins/ranks | `survarena/evaluation/_ranking.py` | Metric-direction lookup and tidy DataFrame outputs | Replace within-dataset all-pairs matrices and pooled fold ranks with exact cell joins and one method score/rank per dataset. |
| Dataset inference | `survarena/evaluation/_significance.py` | Lazy SciPy import, symmetric directed rows, Holm/BH correction | Aggregate paired deltas by dataset before Wilcoxon; reject incomplete multiple-method blocks. |
| Ratings | `survarena/evaluation/_ratings.py::_extract_matches` | Deterministic method ordering and seeded update order | Build one dataset-level match per pair on common support so fold count cannot weight datasets. |
| Prediction validation | `survarena/core/models/contract.py` | Dataclass-like explicit typed contracts and early `ValueError` | Validate numeric array semantics before metric dispatch. |
| Metric references | `tests/test_evaluation.py` | Small deterministic arrays and direct assertions | Compare production values to scikit-survival/SciPy references within declared tolerances. |
| Runner wiring | `survarena/benchmark/runner.py::evaluate_split` | Structured failure rows instead of aborting the collection | Record invalid-output/support reasons and fixed requested horizons without silent clipping. |

## File Ownership

- Plan 02-01 owns comparison population, eligibility, ranks, wins, reliability, and Elo population semantics.
- Plan 02-02 owns dataset-level inference, bootstrap, complete-block multiple-method analysis, and statistical exports.
- Plan 02-03 owns prediction validation, metric support/reference behavior, and runner provenance wiring.

## Constraints

- Do not regenerate or bless historical result matrices.
- Do not silently drop failed methods from headline support.
- Do not treat folds/seeds as independent cross-dataset samples.
- Do not silently clip a requested horizon and report it under the original label.
- Do not add a new numerical dependency when declared NumPy/SciPy/scikit-survival cover the reference need.

---

*Phase: 02-scientific-comparison-kernel*
*Pattern map completed: 2026-09-01 (inline pattern-mapper fallback)*
