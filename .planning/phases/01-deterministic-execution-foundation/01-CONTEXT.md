# Phase 1: Deterministic Execution Foundation - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase defines deterministic execution behavior for benchmark profile tiers (`smoke`, `standard`, `manuscript`) and establishes reliable resume semantics for interrupted runs. It does not add new benchmark capabilities outside execution determinism and recovery policy.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Methodology Positioning
- **D-01:** Use a TabArena-inspired benchmarking philosophy, but SurvArena's own protocol and contracts are canonical for implementation and claims.

### Profile Contract
- **D-02:** Profile definitions are strict and locked; each profile must have fixed default split/seed/metric behavior to preserve comparability.
- **D-03:** Profile intent is canonical: `smoke` = health check, `standard` = iterative research, `manuscript` = publication-grade claims.

### Determinism Policy
- **D-04:** Split manifest mismatch is a hard failure; regeneration must be explicit rather than automatic.
- **D-05:** Seed handling is strict: all stochastic components must receive tracked seeds; missing seed propagation is an error.

### Resume Policy
- **D-06:** Resume treats only `status=success` rows with valid required outputs as complete.
- **D-07:** Failed rows are retried only within configured retry budget; beyond budget, failures remain recorded.

### Claude's Discretion
- Exact field names and schema structure for resume eligibility checks.
- UX text and verbosity details for CLI status/progress output.
- Internal helper decomposition/refactor strategy in orchestration modules.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope and requirements
- `.planning/ROADMAP.md` — Phase 1 goal, requirements mapping (`EXEC-01`, `EXEC-04`), and success criteria.
- `.planning/REQUIREMENTS.md` — v1 requirement definitions and traceability contract.
- `.planning/PROJECT.md` — product constraints, Python-only scope, and quality/runtimes principles.

### Existing benchmark protocol and contracts
- `docs/protocol.md` — benchmark profile framing, evaluation rules, reproducibility, and output contract.
- `configs/benchmark/smoke_all_models_no_hpo.yaml` — current `smoke` profile behavior baseline.
- `configs/benchmark/standard_v1.yaml` — current `standard`/`research` profile baseline.
- `configs/benchmark/manuscript_v1.yaml` — current manuscript profile baseline.

### Implementation anchors in code
- `survarena/benchmark/runner.py` — profile handling, status lifecycle, resume/retry behavior.
- `survarena/data/splitters.py` — split manifest creation/reuse and deterministic split contract.
- `survarena/run_benchmark.py` — CLI entrypoint for benchmark config selection and resume flags.
- `survarena/logging/export.py` — run ledger and summary export surfaces tied to resume/failure semantics.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `survarena/benchmark/runner.py`: existing run loop already tracks `status`, `hpo_status`, `retry_attempt`, and resume filtering.
- `survarena/data/splitters.py`: existing manifest payload + split-id reuse path can enforce deterministic contract checks.
- `survarena/logging/export.py`: existing run ledger/export functions provide persistence points for strict resume eligibility metadata.

### Established Patterns
- Config-driven behavior via YAML benchmark profiles in `configs/benchmark/`.
- Fail-fast validation for invalid configuration paths and explicit exceptions for unsupported split strategies.
- Structured status records (`success` / `failed`) emitted rather than silent skips.

### Integration Points
- Profile contract enforcement should connect at benchmark config ingestion and runner setup.
- Determinism checks should connect where split manifests are read/validated before execution.
- Resume policy enforcement should connect where existing fold rows are loaded and filtered for skip/retry decisions.

</code_context>

<specifics>
## Specific Ideas

- Keep benchmark framing aligned with TabArena-style comparability and ranking philosophy, while treating SurvArena protocol definitions as the source of truth.
- Make profile guarantees and deterministic behavior explicit enough that downstream statistical claims are auditable.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 01-deterministic-execution-foundation*
*Context gathered: 2026-04-23*
