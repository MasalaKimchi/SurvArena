# Manuscript-Grade Run Status

Last reviewed against local configs and retained result artifacts: 2026-06-05.

This page summarizes which manuscript-grade benchmark tracks have completed
evidence and which model/dataset combinations are implemented or configured but
not yet retained as complete manuscript evidence.

## Legend

| Mark | Status | Meaning |
| --- | --- | --- |
| Complete | Retained manuscript evidence | Successful fold-level results are present in the compact manuscript artifact bundle. |
| Partial | Some retained evidence | At least one successful manuscript-grade subset exists, but the full configured matrix is not complete or not fully eligible. |
| Configured | Ready to run | Benchmark YAML exists, but no retained complete manuscript-grade result bundle was found. |
| Implemented | Adapter available | Method is registered/configured, but it is excluded from the maintained manuscript evidence track or has only failed/ineligible attempts. |
| Excluded | Intentionally out of scope | Excluded from the maintained manuscript track because of runtime, RAM, or protocol constraints. |

## Benchmark Track Overview

| Track | Config | Datasets | Methods | Mode | Retained status | Evidence notes |
| --- | --- | ---: | ---: | --- | --- | --- |
| Clinical no-HPO | `configs/benchmark/manuscript_v1.yaml` | 7 | 27 | `no_hpo` | Complete | 2,835 successful fold rows: 7 datasets x 27 methods x 15 splits. |
| Clinical HPO | `configs/benchmark/manuscript_hpo_v1.yaml` | 7 | 27 | `hpo` | Configured | Same clinical dataset/model matrix; no retained complete HPO bundle found under `results/manuscript_grade/`. |
| Clinical foundation subset | `configs/benchmark/manuscript_autogluon_foundation_v1.yaml` | 7 | 3 | `no_hpo` | Configured | Covers `tabicl_survival`, `tabm_survival`, and `realtabpfn_survival`; these are already represented in the complete clinical no-HPO bundle. |
| Genomics no-HPO | `configs/benchmark/manuscript_genomics_v1.yaml` | 5 | 27 | `no_hpo` | Partial | Retained success bundle has 1,260 fold rows: 4 TCGA datasets x 21 eligible methods x 15 splits. |
| Genomics foundation subset | `configs/benchmark/manuscript_genomics_autogluon_foundation_v1.yaml` | 5 | 3 | `no_hpo` | Partial | Status artifacts show successful `tabicl_survival` runs and failed/ineligible `tabm_survival` attempts; no complete retained subset bundle. |

## Dataset x Model-Family Status

| Dataset group | Dataset ids | Classical / linear | Trees / boosting | Deep survival | Foundation adapters | Mitra survival |
| --- | --- | --- | --- | --- | --- | --- |
| Clinical benchmark | `support`, `metabric`, `nwtco`, `aids`, `gbsg2`, `flchain`, `whas500` | Complete | Complete | Complete | Complete for `tabpfn_survival`, `tabicl_survival`, `tabm_survival`, `realtabpfn_survival` | Excluded from maintained no-HPO manuscript run |
| TCGA genomics retained evidence | `tcga_kirc_xena`, `tcga_luad_xena`, `tcga_ov_xena`, `tcga_skcm_xena` | Partial: 21 eligible methods per dataset; AFT/`coxnet` failures excluded from retained success bundle | Complete for eligible tree/boosting methods | Complete for eligible deep methods | Partial: `tabpfn_survival` and `tabicl_survival` retained; `tabm_survival` ineligible; `realtabpfn_survival` not in retained success bundle | Implemented, attempted, ineligible |
| TCGA BRCA genomics | `tcga_brca_xena` | Configured | Configured | Configured | Partial status artifacts only: `tabicl_survival` succeeded, `tabm_survival` failed | Implemented, not retained as successful evidence |

## Method-Family Detail

| Family | Method ids | Clinical no-HPO status | Genomics no-HPO status | Notes |
| --- | --- | --- | --- | --- |
| Cox / linear survival | `coxph`, `coxnet`, `fast_survival_svm` | Complete | Partial | `coxph` and `fast_survival_svm` are retained for the 4-dataset TCGA success bundle; `coxnet` is ineligible in the genomics status artifact because of numerical errors. |
| Parametric AFT | `weibull_aft`, `lognormal_aft`, `loglogistic_aft` | Complete | Partial | Clinical evidence is complete; TCGA status artifacts show convergence failures for the AFT methods on retained genomics datasets. |
| Additive / boosting survival | `aalen_additive`, `gradient_boosting_survival`, `componentwise_gradient_boosting` | Complete | Complete for retained TCGA success datasets | Present in the 4-dataset TCGA retained success bundle. |
| Ensemble trees | `rsf`, `extra_survival_trees` | Complete | Complete for retained TCGA success datasets | Present in the 4-dataset TCGA retained success bundle. |
| Gradient boosting backends | `xgboost_cox`, `xgboost_aft`, `catboost_cox`, `catboost_survival_aft` | Complete | Complete for retained TCGA success datasets | Present in the 4-dataset TCGA retained success bundle. |
| PyCox / neural survival | `deepsurv`, `deepsurv_moco`, `logistic_hazard`, `pmf`, `mtlr`, `deephit_single`, `pchazard`, `cox_time` | Complete | Complete for retained TCGA success datasets | Present in the 4-dataset TCGA retained success bundle. |
| Direct foundation horizon adapters | `tabpfn_survival`, `tabicl_survival` | Complete | Complete for retained TCGA success datasets | `tabicl_survival` also has a BRCA status-artifact success row. |
| AutoGluon foundation horizon adapters | `tabm_survival`, `realtabpfn_survival` | Complete | Partial / not retained | `tabm_survival` appears in genomics status artifacts as failed/ineligible. `realtabpfn_survival` is configured and complete clinically but is not present in the retained genomics success bundle. |
| Mitra event-risk adapter | `mitra_survival_frozen` | Implemented, excluded | Implemented, attempted, ineligible | Registered and documented, but excluded from maintained clinical no-HPO because local CPU/RAM use can exceed the conventional model wall-clock budget. Genomics status artifacts show failed/ineligible attempts. |

## Completion Snapshot

| Evidence bundle | Retained success rows | Dataset/method pairs | Split coverage | Primary artifact |
| --- | ---: | ---: | --- | --- |
| Clinical no-HPO | 2,835 | 189 | 15 / 15 for every dataset-method pair | `results/manuscript_grade/clinical_no_hpo/elo/manuscript_fold_results_success.csv` |
| Genomics no-HPO | 1,260 | 84 | 15 / 15 for retained successful pairs | `results/manuscript_grade/genomics_no_hpo/elo/manuscript_fold_results_success.csv` |
| Genomics status audit | n/a | 110 audited pairs | 85 eligible, 25 ineligible | `results/manuscript_grade/genomics_no_hpo/genomics_dataset_method_status.csv` |

## Source Files

- Maintained project state: `PROJECT_STATE.md`
- Clinical evidence summary: `docs/manuscript_evidence.md`
- Method catalog: `docs/methods.md`
- Foundation model notes: `docs/foundation_models.md`
- Clinical no-HPO config: `configs/benchmark/manuscript_v1.yaml`
- Clinical HPO config: `configs/benchmark/manuscript_hpo_v1.yaml`
- Genomics no-HPO config: `configs/benchmark/manuscript_genomics_v1.yaml`
