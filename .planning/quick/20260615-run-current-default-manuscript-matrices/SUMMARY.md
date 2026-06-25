---
status: complete
completed: 2026-06-19
---

# Current-Default Manuscript Matrix Summary

## Outcome

Completed attempt coverage for the full canonical no-HPO benchmark:

- Clinical: 189 / 189 dataset-method cells complete, 2,835 successful fold rows.
- Genomics: 135 / 135 cells attempted; 105 complete, 7 partial, and 23 all-fold failures, with 1,619 successful fold rows.
- Missing cells after execution: 0.

## Runtime Contract

Restored the bounded foundation settings recorded by the completed current-default manifests:

- Clinical TabPFN/TabICL: CPU, 5 intervals, 500 stacked rows, prediction batch size 256.
- Genomics TabPFN/TabICL: CPU, 3 intervals, 100 stacked rows, prediction batch size 256.
- TabM: 30-second fit budget.
- RealTabPFN: 100 stacked rows and a 1-second fit budget.

This avoided the Apple MPS allocator crash observed when direct TabPFN used `device: auto`.

## Report Artifacts

- `results/manuscript_grade/clinical_no_hpo_current_default/elo/`
- `results/manuscript_grade/genomics_no_hpo_current_default/elo/`
- `results/manuscript_grade/no_hpo_benchmark_report/report.html`
- `results/manuscript_grade/no_hpo_benchmark_report/coverage_status.csv`
- `results/manuscript_grade/no_hpo_benchmark_report/failure_summary.csv`

The Elo bundles use 1,000 bootstrap draws and eligibility-complete paired comparisons across the maintained metric suite.

## Verification

- `python -m pytest -q` — 211 passed, 6 skipped.
- `python -m ruff check survarena tests scripts configs`
- `python -m compileall -q survarena scripts`
- `./scripts/validate_benchmark_protocol.sh`
- `python scripts/audit_manuscript_publishability.py --strict` — exits 2 as expected because the broader manuscript program still lacks clinical HPO and genomics does not have universal successful coverage.
- HTML report checked at desktop and mobile viewport widths with no page-level horizontal overflow and all report images loaded.
