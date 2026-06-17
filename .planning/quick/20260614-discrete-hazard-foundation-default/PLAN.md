---
status: complete
created: 2026-06-14
---

# Discrete-Hazard Foundation Default

## Objective

Make pooled discrete-time hazard survival the default implementation path for all foundation models, remove retired
foundation horizon-adapter defaults from maintained configs/docs, and keep the codebase concise around one canonical
foundation survival contract.

## Passes

### Pass: Implementation Surface
Current behavior: canonical foundation method IDs include retired horizon adapters and separate discrete-hazard variants.
Structural improvement: canonical foundation method IDs resolve to discrete-hazard implementations; retired horizon paths stop
appearing in maintained benchmark configs.
Validation check: targeted foundation/method tests and dry-run configs.
Migration split: historical result artifacts are left untouched.

### Pass: Config And Documentation
Current behavior: manuscript configs/docs list both horizon and discrete-hazard foundation adapters.
Structural improvement: maintained docs and configs describe discrete-hazard foundation survival as the default; old horizon
adapters are documented only as retired/historical if needed.
Validation check: grep for stale legacy-default claims and run ruff/tests touched by config parsing.
Migration split: benchmark result backfills remain separate from code cleanup.

### Pass: Review Graph And Verification
Current behavior: code-review graph may be stale after refactor.
Structural improvement: update graph and inspect review context for changed files.
Validation check: graph update plus focused pytest/ruff.
Migration split: broader benchmark reruns remain separate.
