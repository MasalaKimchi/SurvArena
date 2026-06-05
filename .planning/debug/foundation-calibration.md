---
status: resolved
trigger: "Investigate why RealTabPFN and TabM appear miscalibrated in clinical benchmark metrics; determine whether training/evaluation logic has implementation errors; fix and update code review graph."
created: 2026-06-05
updated: 2026-06-05
---

# Debug Session: foundation-calibration

## Symptoms

- Expected behavior: AutoGluon-backed foundation adapters should produce event-risk scores and survival curves that are directionally consistent, monotone, and reasonably calibrated when evaluated by Brier/calibration/net-benefit metrics.
- Actual behavior: `realtabpfn_survival` and `tabm_survival` have competitive Uno C but poor Brier/IBS/net-benefit and extreme calibration slopes, especially on `aids` and `nwtco`.
- Error messages: none; all clinical no-HPO fold rows completed successfully.
- Timeline: observed in completed manuscript clinical no-HPO evidence bundle.
- Reproduction: aggregate `results/manuscript_grade/clinical_no_hpo/dataset_model/*/*/*_fold_results.csv` and inspect `calibration_slope_50`, `brier_50`, `ibs`, and `uno_c` for `realtabpfn_survival` and `tabm_survival`.

## Current Focus

- hypothesis: AutoGluon event-risk adapters may be training on raw event indicators without horizon/censoring awareness, then using a Cox/Breslow survival calibration path whose risk scale or direction is not compatible with binary event probabilities.
- test: inspect adapter training and survival conversion logic; compare predictions/artifacts where saved; add targeted tests for risk direction, monotone survival, and calibration metric semantics.
- expecting: either a training-label/scale bug, an evaluation direction bug, or an expected methodological limitation that should be exposed and guarded.
- next_action: gather initial evidence from adapter code and result artifacts.

## Evidence

- 2026-06-05: Clinical no-HPO artifacts show complete coverage but extreme calibration slopes for `tabm_survival` and `realtabpfn_survival`, especially `aids` and `nwtco`.
- 2026-06-05: `survarena/methods/automl/mitra_survival.py` trained AutoGluon-backed TabM/RealTabPFN on raw event indicators only, so censored rows were treated as non-events regardless of follow-up time.
- 2026-06-05: The same adapters passed bounded class probabilities directly into `fit_breslow_baseline_survival`, whose Cox-style formula exponentiates risk scores. That scale is appropriate for log-risk-like scores, not calibrated event probabilities.
- 2026-06-05: `survarena/evaluation/_metric_stats.py` treated `calibration_slope_50` as a maximize metric, so extreme slopes could be ranked as better by generic ranking/reporting helpers.
- 2026-06-05: Implemented censored-aware direct horizon classification for `tabm_survival` and `realtabpfn_survival`, matching the TabPFN/TabICL adapter contract: fit one classifier per horizon using known-at-horizon labels, fallback to KM event probability when unsupported, reconstruct monotone survival directly from horizon event probabilities.
- 2026-06-05: Added calibration slope/intercept absolute-error metrics and stopped treating raw calibration slope as higher-is-better.
- 2026-06-05: Verification passed: `pytest tests/test_methods.py tests/test_evaluation.py`, `pytest tests/test_foundation.py tests/test_benchmark.py`, and `ruff check survarena tests configs scripts`.
- 2026-06-05: Cleanup worker removed the obsolete local `TabICLSurvivalMethod` shim from `survarena/methods/automl/mitra_survival.py`; registry now uses the canonical direct-horizon TabICL adapter.
- 2026-06-05: Mitra feasibility check passed in the local environment: `scripts/check_environment.py --include-foundation` and `survarena foundation-check` both reported Mitra runtime readiness, and a bounded one-seed `whas500` run completed all 5 folds with mean `uno_c=0.7776`.
- 2026-06-05: Post-fix one-seed `whas500` calibration smoke completed all 10 TabM/RealTabPFN folds. Compared with old manuscript `whas500__baseline` means, RealTabPFN `calibration_slope_50` improved from `2.8950` to `0.9840`; TabM improved from `3.2534` to `2.1088`. RealTabPFN also improved `brier_50` from `0.1514` to `0.1386` and `ibs` from `0.1894` to `0.1778`.

## Eliminated

- hypothesis: The miscalibration was only a plotting/Elo artifact. Result: eliminated; the fold-level `brier_50`, `ibs`, `net_benefit_50`, and calibration slopes showed real survival-probability quality problems.
- hypothesis: The issue was failed benchmark rows or missing primary metrics. Result: eliminated; the clinical no-HPO matrix had 2,835 successful rows and no missing `uno_c`.

## Resolution

- root_cause: AutoGluon-backed TabM/RealTabPFN survival adapters used a plain event-ever binary target that ignored censoring/time horizons, then converted class probabilities to survival curves through a Cox/Breslow log-risk calibration path. Raw calibration slope was also incorrectly registered as maximize-able.
- fix: Reworked TabM/RealTabPFN to censored-aware horizon classifiers with direct monotone survival reconstruction; added horizon metadata and calibration error metrics; removed raw calibration slope from maximize metrics; updated method YAML contracts and regression tests.
- verification: `pytest tests/test_methods.py tests/test_evaluation.py`; `pytest tests/test_foundation.py tests/test_benchmark.py`; `pytest`; `ruff check survarena tests configs scripts`; `python scripts/check_environment.py --include-foundation`; `survarena foundation-check`; bounded `whas500` smoke runs for `mitra_survival_frozen`, `tabm_survival`, and `realtabpfn_survival`.
- files_changed: `survarena/methods/automl/mitra_survival.py`, `survarena/methods/foundation/catalog.py`, `survarena/evaluation/metrics.py`, `survarena/evaluation/_metric_stats.py`, `configs/methods/tabicl_survival.yaml`, `configs/methods/tabm_survival.yaml`, `configs/methods/realtabpfn_survival.yaml`, `README.md`, `docs/foundation_models.md`, `tests/test_methods.py`, `tests/test_evaluation.py`.
