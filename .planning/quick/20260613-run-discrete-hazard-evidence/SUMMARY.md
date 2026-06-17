---
status: incomplete
completed: 2026-06-13
---

# Run Discrete-Hazard Manuscript Evidence

Expanded clinical discrete-hazard foundation evidence under
`results/manuscript_grade/clinical_discrete_hazard_foundation/`.

Completed full 15-fold cells:

- `tabm_discrete_hazard_survival` / `aids`
- `tabm_discrete_hazard_survival` / `gbsg2`
- `tabm_discrete_hazard_survival` / `whas500`
- `tabm_discrete_hazard_survival` / `flchain`

Result: `tabm_discrete_hazard_survival` now has successful 15-fold clinical coverage across all 7 clinical datasets.

Attempted but not completed:

- `tabicl_discrete_hazard_survival` / `flchain`: interrupted after ~20 minutes with no export; stack showed TabICL prediction attention.
- `tabpfn_discrete_hazard_survival` / `nwtco`: interrupted after a long low-progress run; stack showed TabPFN prediction attention.
- `realtabpfn_discrete_hazard_survival` / `support`: original run wrote failure rows because the stacked hazard frame exceeded RealTabPFN-v2 `ag.max_rows=10000`; capped config avoided the row-cap failure but was interrupted after ~30 minutes in TabPFN prediction attention.

Created temporary capped config:

- `.planning/quick/20260613-run-discrete-hazard-evidence/realtabpfn_discrete_hazard_capped.yaml`

Final coverage snapshot:

- `tabm_discrete_hazard_survival`: 7/7 clinical datasets have 15 successful folds.
- `tabicl_discrete_hazard_survival`: 6/7 clinical datasets complete; `flchain` remains missing.
- `tabpfn_discrete_hazard_survival`: existing artifacts include successes and failures; `nwtco` remains missing.
- `realtabpfn_discrete_hazard_survival`: no successful clinical dataset completed in this pass; `support` has failure rows from the uncapped attempt.

Verification:

- Re-scanned per-method/per-dataset fold-result row counts and statuses after runs.
- Confirmed no active `python -m survarena.run_benchmark` process remains.
