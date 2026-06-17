# Manuscript Publishability Audit

Generated from local configs and result artifacts on 2026-06-15.

## Verdict

**Not yet publication-ready as a final manuscript evidence bundle.** The code path is validated, but the retained
evidence is not fully refreshed under the canonical discrete-hazard foundation defaults.

## Completed Locally

- Canonical foundation adapters default to pooled discrete-time hazard.
- Maintained benchmark configs dry-run with canonical foundation IDs.
- Clinical no-HPO retained artifact has full row coverage for its historical matrix.
- Protocol smoke validation passes locally.

## Blocking Gaps

- Refresh clinical no-HPO Elo/report bundle after canonical discrete-hazard foundation defaults.
- Produce canonical foundation evidence or a locked provenance bridge from alias-run evidence to canonical IDs.
- Complete or explicitly scope out clinical HPO evidence.
- Decide whether genomics is a main benchmark, appendix partial, or excluded sensitivity analysis.
- Freeze dependency environment in a lockfile or archival environment export.
- Stage manuscript tables/figures from the refreshed compact artifact bundle.

## Retained Evidence Coverage

| track                               | status   | rows      | complete_pairs   | artifact                                                                         |
|:------------------------------------|:---------|:----------|:-----------------|:---------------------------------------------------------------------------------|
| Clinical no-HPO current default     | partial  | 240/2835  | 16/189           | results/manuscript_grade/clinical_no_hpo_current_default/dataset_model           |
| Clinical no-HPO                     | complete | 2835/2835 | 189/189          | results/manuscript_grade/clinical_no_hpo/elo/manuscript_fold_results_success.csv |
| Clinical HPO                        | missing  | 0/2835    | 0/189            | results/manuscript_grade/clinical_hpo/elo/manuscript_fold_results_success.csv    |
| Genomics no-HPO current default     | missing  | 0/2025    | 0/135            | results/manuscript_grade/genomics_no_hpo_current_default/dataset_model           |
| Genomics no-HPO                     | partial  | 1260/2025 | 84/135           | results/manuscript_grade/genomics_no_hpo/elo/manuscript_fold_results_success.csv |
| Clinical discrete-hazard foundation | partial  | 246/420   | 16/28            | results/manuscript_grade/clinical_discrete_hazard_foundation                     |

## Foundation Evidence Detail

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
- Current retained genomics Elo view is a partial 4-dataset eligible-method view, not the full configured
  `manuscript_genomics_v1.yaml` matrix.

## Minimum Completion Criteria

- Run or provenance-map the canonical foundation IDs across the clinical manuscript matrix.
- Rebuild `results/manuscript_grade/clinical_no_hpo/elo/` from the refreshed fold results.
- Either complete `configs/benchmark/manuscript_hpo_v1.yaml` or move HPO to a clearly labeled appendix/future-work scope.
- Either complete the five-dataset genomics matrix or label the retained genomics bundle as exploratory/appendix evidence.
- Export a frozen environment spec alongside the artifact bundle.
- Regenerate manuscript tables/figures from the exact retained artifacts and cite their paths/checksums.
