---
status: in_progress
created: 2026-06-13
---

# Run Discrete-Hazard Manuscript Evidence

## Objective

Run the remaining manuscript-grade discrete-hazard foundation survival benchmark cells needed to expand evidence under
`results/manuscript_grade/clinical_discrete_hazard_foundation/`, using existing benchmark runner resume/output controls and
without rerunning completed fold rows.

## Scope

- Identify missing clinical dataset/method cells for discrete-hazard foundation adapters.
- Run missing cells with `configs/benchmark/manuscript_foundation_adapters_v1.yaml`.
- Keep outputs in the existing clinical discrete-hazard evidence directory layout.
- Summarize completed and still-missing coverage.

## Verification

- Confirm fold-result row counts by method and dataset after runs.
- Record any failed or skipped cells with command context.
