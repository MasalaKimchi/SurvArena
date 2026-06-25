# Manuscript Publishability Audit

Generated from local configs and result artifacts on 2026-06-18.

## Verdict

**Not yet publication-ready as the full no-HPO-plus-HPO manuscript evidence bundle.** The current-default clinical
no-HPO matrix and report are complete; the remaining blockers are listed below.

## Completed Locally

- Canonical foundation adapters default to pooled discrete-time hazard.
- Maintained benchmark configs dry-run with canonical foundation IDs.
- Clinical no-HPO current-default evidence has complete 7-dataset x 27-method x 15-split coverage.
- Current-default clinical and genomics Elo/report bundles have been rebuilt.
- Protocol smoke validation passes locally.

## Blocking Gaps

- Complete or explicitly scope out clinical HPO evidence.
- Decide whether incomplete-success genomics coverage is a main benchmark, appendix robustness analysis, or excluded sensitivity analysis.
- Freeze dependency environment in a lockfile or archival environment export.
- Stage final manuscript tables/figures from the current-default compact artifact bundles.

## Retained Evidence Coverage

| track                               | status   | rows      | complete_pairs   | artifact                                                                         |
|:------------------------------------|:---------|:----------|:-----------------|:---------------------------------------------------------------------------------|
| Clinical no-HPO current default     | complete | 2835/2835 | 189/189          | results/manuscript_grade/clinical_no_hpo_current_default/dataset_model           |
| Clinical no-HPO                     | complete | 2835/2835 | 189/189          | results/manuscript_grade/clinical_no_hpo/elo/manuscript_fold_results_success.csv |
| Clinical HPO                        | missing  | 0/2835    | 0/189            | results/manuscript_grade/clinical_hpo/elo/manuscript_fold_results_success.csv    |
| Genomics no-HPO current default     | partial  | 1619/2025 | 105/135          | results/manuscript_grade/genomics_no_hpo_current_default/dataset_model           |
| Genomics no-HPO                     | partial  | 1260/2025 | 84/135           | results/manuscript_grade/genomics_no_hpo/elo/manuscript_fold_results_success.csv |
| Clinical discrete-hazard foundation | partial  | 246/420   | 16/28            | results/manuscript_grade/clinical_discrete_hazard_foundation                     |

## Legacy Foundation-Only Evidence Detail

This table audits the older foundation-only artifact root. It is retained for provenance and no longer blocks the
complete current-default clinical no-HPO matrix.

| canonical_method_id   | dataset_id   | source_method_id                    |   success_rows |   expected_splits | gap                                                                                  |
|:----------------------|:-------------|:------------------------------------|---------------:|------------------:|:-------------------------------------------------------------------------------------|
| realtabpfn_survival   | aids         |                                     |              0 |                15 | missing                                                                              |
| realtabpfn_survival   | flchain      |                                     |              0 |                15 | missing                                                                              |
| realtabpfn_survival   | gbsg2        |                                     |              0 |                15 | missing                                                                              |
| realtabpfn_survival   | metabric     |                                     |              0 |                15 | missing                                                                              |
| realtabpfn_survival   | nwtco        |                                     |              0 |                15 | missing                                                                              |
| realtabpfn_survival   | support      | realtabpfn_discrete_hazard_survival |              0 |                15 | all attempted rows failed                                                            |
| realtabpfn_survival   | whas500      |                                     |              0 |                15 | missing                                                                              |
| tabicl_survival       | aids         | tabicl_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabicl_survival       | flchain      |                                     |              0 |                15 | missing                                                                              |
| tabicl_survival       | gbsg2        | tabicl_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabicl_survival       | metabric     | tabicl_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabicl_survival       | nwtco        | tabicl_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabicl_survival       | support      | tabicl_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabicl_survival       | whas500      | tabicl_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabm_survival         | aids         | tabm_discrete_hazard_survival       |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabm_survival         | flchain      | tabm_discrete_hazard_survival       |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabm_survival         | gbsg2        | tabm_discrete_hazard_survival       |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabm_survival         | metabric     | tabm_discrete_hazard_survival       |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabm_survival         | nwtco        | tabm_discrete_hazard_survival       |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabm_survival         | support      | tabm_discrete_hazard_survival       |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabm_survival         | whas500      | tabm_discrete_hazard_survival       |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabpfn_survival       | aids         | tabpfn_discrete_hazard_survival     |              6 |                15 | incomplete: 6/15 successful splits                                                   |
| tabpfn_survival       | flchain      | tabpfn_discrete_hazard_survival     |              0 |                15 | all attempted rows failed                                                            |
| tabpfn_survival       | gbsg2        | tabpfn_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabpfn_survival       | metabric     | tabpfn_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |
| tabpfn_survival       | nwtco        |                                     |              0 |                15 | missing                                                                              |
| tabpfn_survival       | support      | tabpfn_discrete_hazard_survival     |              0 |                15 | all attempted rows failed                                                            |
| tabpfn_survival       | whas500      | tabpfn_discrete_hazard_survival     |             15 |                15 | complete alias-run evidence; rerun or provenance-map before citing canonical default |

## Genomics Status

- Genomics status audit rows with failed/ineligible labels: 25.
- The current-default five-cohort matrix has complete attempt coverage but partial universal-success coverage;
  eligibility-complete comparisons are available in `results/manuscript_grade/genomics_no_hpo_current_default/elo/`.

## Minimum Completion Criteria

- Either complete `configs/benchmark/manuscript_hpo_v1.yaml` or move HPO to a clearly labeled appendix/future-work scope.
- Label the five-dataset genomics matrix as a robustness analysis unless universal successful coverage is required.
- Export a frozen environment spec alongside the artifact bundle.
- Stage final manuscript tables/figures from the current-default artifacts and cite their paths/checksums.
