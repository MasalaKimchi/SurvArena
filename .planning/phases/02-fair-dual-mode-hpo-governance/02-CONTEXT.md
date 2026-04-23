# Phase 2: Fair Dual-Mode HPO Governance - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase enforces fair no-HPO vs HPO comparison governance so every model run in a benchmark collection follows one explicit, auditable dual-mode contract and budget policy. Scope is limited to parity, budget policy enforcement, and reporting required to trust those comparisons.

</domain>

<decisions>
## Implementation Decisions

### Dual-Mode Pairing Contract
- **D-01:** Fairness unit is dataset/split/seed parity: no-HPO and HPO must be paired on the same dataset, split, and seed.
- **D-02:** Results remain in one canonical artifact set, with explicit per-row mode labeling (for example `hpo_mode`) rather than separate mode-specific artifacts.
- **D-03:** Comparative summaries enforce a hard parity gate: if one mode is missing for a pairing unit, that unit is comparison-ineligible rather than silently mixed.
- **D-04:** Execution order is deterministic and sequential per pairing unit: run no-HPO first, then HPO.

### Carried Forward Constraints
- **D-05:** Profile contracts remain strict and locked (from Phase 1), so dual-mode governance must preserve profile intent and reproducibility guarantees.
- **D-06:** Determinism policy remains strict (from Phase 1): seed propagation and manifest consistency are mandatory for both modes.

### Claude's Discretion
- Exact schema field names for parity eligibility markers and mode labels.
- Internal orchestration shape (loop decomposition, helper boundaries, and retry bookkeeping details).
- Presentation details for CLI/status messaging around parity failures.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` — Phase 2 goal, requirement mapping (`EXEC-02`, `EXEC-03`), and success criteria.
- `.planning/REQUIREMENTS.md` — v1 requirement definitions and traceability contract for dual-mode execution and budget governance.
- `.planning/PROJECT.md` — constraints on fairness, runtime practicality, and artifact compactness.
- `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md` — locked profile/determinism/resume decisions that Phase 2 must preserve.

### Protocol and benchmark contracts
- `docs/protocol.md` — shared-budget and shared-split benchmark contract, profile semantics, and HPO artifact expectations.
- `configs/benchmark/smoke_all_models_no_hpo.yaml` — no-HPO baseline profile behavior.
- `configs/benchmark/standard_v1.yaml` — research profile with current HPO configuration defaults.
- `configs/benchmark/cloud_comprehensive_all_models_hpo.yaml` — manuscript-scale HPO profile and comprehensive method coverage.

### Implementation anchors in code
- `survarena/benchmark/runner.py` — benchmark orchestration, split looping, resume/retry flow, and HPO metadata/trial export hooks.
- `survarena/benchmark/tuning.py` — HPO config parsing and Optuna-driven search behavior.
- `survarena/logging/export.py` — HPO trial and summary artifact writers.
- `tests/test_hpo_config.py` — existing HPO config expectation tests to extend for governance rules.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `survarena/benchmark/runner.py`: central run loop already emits run payload metadata and exports HPO trial rows.
- `survarena/benchmark/tuning.py`: `_parse_hpo_config()` and `select_hyperparameters()` already resolve HPO defaults, eligibility, and status codes.
- `survarena/logging/export.py`: established export surfaces for trial-level and summary-level HPO artifacts.

### Established Patterns
- Config-driven benchmark policy via `configs/benchmark/*.yaml` with `hpo` sections.
- Structured run payload contracts carrying status, trial counts, and backend metadata.
- Fail-fast validation and explicit statuses instead of silent fallback behavior.

### Integration Points
- Pairing and mode labeling should connect where run records are created in `runner.py`.
- Parity gating should connect in summary/export stages where comparative aggregates are computed.
- Uniform budget policy enforcement should connect where `hpo_cfg` is normalized and propagated from benchmark config into tuning execution.

</code_context>

<specifics>
## Specific Ideas

- Keep one canonical benchmark collection output while making no-HPO vs HPO lineage explicit per run.
- Prefer strict comparability over maximum row retention: missing counterpart mode should block that pairing unit from comparative claims.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-fair-dual-mode-hpo-governance*
*Context gathered: 2026-04-23*
