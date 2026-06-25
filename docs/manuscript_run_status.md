# Manuscript-Grade Run Status

Last reviewed against local configs and retained result artifacts: 2026-06-19.

This page summarizes which manuscript-grade benchmark tracks have completed
evidence and which model/dataset combinations are implemented or configured but
not yet retained as complete manuscript evidence.

For the stricter publication-readiness gate, run
`python scripts/audit_manuscript_publishability.py` and review
[`manuscript_publishability.md`](manuscript_publishability.md). That audit is
source-backed by local configs and retained result CSVs, and currently marks the
final manuscript evidence bundle as not publication-ready until the canonical
discrete-hazard foundation evidence is refreshed or provenance-mapped.

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
| Clinical no-HPO current default | `configs/benchmark/manuscript_v1.yaml` | 7 | 27 | `no_hpo` | Complete | 2,835 successful fold rows: 7 datasets x 27 methods x 15 splits, including canonical bounded discrete-hazard foundation adapters. |
| Clinical HPO | `configs/benchmark/manuscript_hpo_v1.yaml` | 7 | 27 | `hpo` | Configured | Same clinical dataset/model matrix; no retained complete HPO bundle found under `results/manuscript_grade/`. |
| Clinical foundation subset | `configs/benchmark/manuscript_autogluon_foundation_v1.yaml` | 7 | 3 | `no_hpo` | Configured | Covers discrete-hazard `tabicl_survival`, `tabm_survival`, and `realtabpfn_survival`; retained historical evidence predates the default switch. |
| Genomics no-HPO current default | `configs/benchmark/manuscript_genomics_v1.yaml` | 5 | 27 | `no_hpo` | Complete attempt coverage | All 135 cells were attempted: 105 complete, 7 partial, and 23 all-fold failures, yielding 1,619 successful fold rows. |
| Genomics foundation subset | `configs/benchmark/manuscript_genomics_autogluon_foundation_v1.yaml` | 5 | 3 | `no_hpo` | Partial | Historical status artifacts predate the discrete-hazard default switch; no complete retained subset bundle. |

## Dataset x Model-Family Status

| Dataset group | Dataset ids | Classical / linear | Trees / boosting | Deep survival | Foundation adapters | Mitra survival |
| --- | --- | --- | --- | --- | --- | --- |
| Clinical benchmark | `support`, `metabric`, `nwtco`, `aids`, `gbsg2`, `flchain`, `whas500` | Complete | Complete | Complete | Complete under bounded current defaults | Excluded from maintained no-HPO manuscript run |
| TCGA genomics current default | `tcga_brca_xena`, `tcga_kirc_xena`, `tcga_luad_xena`, `tcga_ov_xena`, `tcga_skcm_xena` | Attempted; Coxnet/AFT failures reduce eligibility | Complete for eligible tree/boosting methods | Complete for eligible deep methods | Attempted on every cohort; RealTabPFN failed all folds and TabM was mixed | Implemented, outside the maintained matrix |

## Method-Family Detail

| Family | Method ids | Clinical no-HPO status | Genomics no-HPO status | Notes |
| --- | --- | --- | --- | --- |
| Cox / linear survival | `coxph`, `coxnet`, `fast_survival_svm` | Complete | Partial | `coxph` and `fast_survival_svm` are retained for the 4-dataset TCGA success bundle; `coxnet` is ineligible in the genomics status artifact because of numerical errors. |
| Parametric AFT | `weibull_aft`, `lognormal_aft`, `loglogistic_aft` | Complete | Partial | Clinical evidence is complete; TCGA status artifacts show convergence failures for the AFT methods on retained genomics datasets. |
| Additive / boosting survival | `aalen_additive`, `gradient_boosting_survival`, `componentwise_gradient_boosting` | Complete | Complete for retained TCGA success datasets | Present in the 4-dataset TCGA retained success bundle. |
| Ensemble trees | `rsf`, `extra_survival_trees` | Complete | Complete for retained TCGA success datasets | Present in the 4-dataset TCGA retained success bundle. |
| Gradient boosting backends | `xgboost_cox`, `xgboost_aft`, `catboost_cox`, `catboost_survival_aft` | Complete | Complete for retained TCGA success datasets | Present in the 4-dataset TCGA retained success bundle. |
| PyCox / neural survival | `deepsurv`, `deepsurv_moco`, `logistic_hazard`, `pmf`, `mtlr`, `deephit_single`, `pchazard`, `cox_time` | Complete | Complete for retained TCGA success datasets | Present in the 4-dataset TCGA retained success bundle. |
| Foundation discrete-hazard adapters | `tabpfn_survival`, `tabicl_survival`, `tabm_survival`, `realtabpfn_survival` | Complete under bounded current defaults | Attempted on all cohorts; TabPFN/TabICL complete, TabM mixed, RealTabPFN failed | Canonical IDs resolve to pooled discrete-time hazard adapters. The current-default artifacts record the CPU and context limits used for manuscript execution. |
| Mitra event-risk adapter | `mitra_survival_frozen` | Implemented, excluded | Implemented, attempted, ineligible | Registered and documented, but excluded from maintained clinical no-HPO because local CPU/RAM use can exceed the conventional model wall-clock budget. Genomics status artifacts show failed/ineligible attempts. |

## Completion Snapshot

| Evidence bundle | Retained success rows | Dataset/method pairs | Split coverage | Primary artifact |
| --- | ---: | ---: | --- | --- |
| Clinical no-HPO current default | 2,835 | 189 complete | 15 / 15 for every dataset-method pair | `results/manuscript_grade/clinical_no_hpo_current_default/` |
| Genomics no-HPO current default | 1,619 | 105 complete, 7 partial, 23 failed | All 135 pairs attempted | `results/manuscript_grade/genomics_no_hpo_current_default/` |
| Genomics status audit | n/a | 110 audited pairs | 85 eligible, 25 ineligible | `results/manuscript_grade/genomics_no_hpo/genomics_dataset_method_status.csv` |
| Clinical discrete-hazard foundation audit | 246 | 16 / 28 complete alias-run pairs | 15 / 15 for complete retained pairs | `results/manuscript_grade/clinical_discrete_hazard_foundation/` |

For retained comparison evidence, prefer compact `elo/` bundles over raw
`dataset_model/` trees. Raw per-dataset/model outputs are useful rebuild inputs
and local provenance, but they are not required once
`manuscript_fold_results_success.csv` and the aggregate Elo tables are retained.
See [`manuscript_evidence.md`](manuscript_evidence.md#result-retention-policy)
for the local cleanup policy and pruning commands.

## Source Files

- Maintained project state: `PROJECT_STATE.md`
- Clinical evidence summary: `docs/manuscript_evidence.md`
- Method catalog: `docs/methods.md`
- Foundation model notes: `docs/foundation_models.md`
- Clinical no-HPO config: `configs/benchmark/manuscript_v1.yaml`
- Clinical HPO config: `configs/benchmark/manuscript_hpo_v1.yaml`
- Genomics no-HPO config: `configs/benchmark/manuscript_genomics_v1.yaml`
- Publishability audit: `docs/manuscript_publishability.md`
- Publishability audit script: `scripts/audit_manuscript_publishability.py`
