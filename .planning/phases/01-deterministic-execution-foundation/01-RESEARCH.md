# Phase 1: Deterministic Execution Foundation - Research

**Researched:** 2026-04-23  
**Domain:** Deterministic benchmark execution, split-governance integrity, and resumable run ledgers for SurvArena benchmark orchestration  
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXEC-01 | User can run benchmark profile tiers (`smoke`, `standard`, `manuscript`) with deterministic split governance. | Locked profile contract and deterministic split-manifest governance strategy, plus strict seed propagation checkpoints. |
| EXEC-04 | User can resume interrupted benchmark runs with structured failure records instead of losing completed progress. | Resume eligibility policy (`success` + output integrity), retry-budget semantics, and run-ledger/failure record structure. |
</phase_requirements>

## Project Constraints (from .cursor/rules/)

No `.cursor/rules/` directory was detected in this workspace scan. [VERIFIED: Glob `.cursor/rules/**`]  
Planning constraints therefore come from `01-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `AGENTS.md`, and repository conventions in `pyproject.toml`/docs. [VERIFIED: `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`, `.planning/REQUIREMENTS.md`, `AGENTS.md`, `pyproject.toml`]

## Summary

Phase 1 should be planned as a hardening pass over existing execution primitives rather than a greenfield build. The repository already has profile YAMLs, deterministic split manifests, run status fields, resume flags, and structured run ledgers; the gap is enforcing strict contracts and eligibility checks required by D-02..D-07 so behavior is auditable and failure-safe under interruption. [VERIFIED: `configs/benchmark/smoke_all_models_no_hpo.yaml`, `configs/benchmark/standard_v1.yaml`, `configs/benchmark/manuscript_v1.yaml`, `survarena/data/splitters.py`, `survarena/benchmark/runner.py`, `survarena/logging/export.py`, `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]

Current `run_benchmark()` resume logic skips records by `(dataset_id, method_id, split_id, seed)` only when `status == "success"`, and retries failures via `max_retries`; however, there is no explicit "required-output-integrity" check before marking prior work complete, and split manifest mismatch currently regenerates splits implicitly instead of hard-failing as required by D-04. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/data/splitters.py`, `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]

The planning focus should therefore be: (1) strict profile governance for `smoke`/`standard`/`manuscript`, (2) explicit manifest mismatch failure path + operator-controlled regeneration, (3) resume eligibility schema and validation, and (4) test coverage for deterministic replay and interruption recovery. [VERIFIED: `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`, `docs/protocol.md`, `survarena/benchmark/runner.py`, `survarena/data/splitters.py`, `pyproject.toml`]

**Primary recommendation:** Implement a "strict deterministic contract" layer at benchmark-load + split-load + resume-filter boundaries, and gate it with dedicated phase tests (`EXEC-01`, `EXEC-04`) before any downstream phase work. [VERIFIED: `.planning/ROADMAP.md`, `survarena/run_benchmark.py`, `survarena/benchmark/runner.py`, `survarena/data/splitters.py`]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Profile tier governance (`smoke`/`standard`/`manuscript`) | API / Backend | Database / Storage | Profiles are loaded and validated in benchmark runtime logic, while persisted artifacts encode contract evidence. [VERIFIED: `survarena/run_benchmark.py`, `survarena/benchmark/runner.py`, `docs/protocol.md`, `survarena/logging/export.py`] |
| Deterministic split governance and manifest policy | Database / Storage | API / Backend | Split materialization and manifest persistence live in `data/splitters`, while runner enforces usage. [VERIFIED: `survarena/data/splitters.py`, `survarena/benchmark/runner.py`] |
| Seed propagation to stochastic components | API / Backend | — | Seed fan-out occurs in evaluate/tuning/runtime param resolution, not in storage tier. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`, `survarena/utils/seeds.py`] |
| Resume completed-work preservation | API / Backend | Database / Storage | Resume decisioning is in runner; persisted fold outputs and ledgers are source of truth. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/logging/export.py`] |
| Structured failure inspection | Database / Storage | API / Backend | Failure records are emitted as structured payload/CSV/JSON artifacts, then surfaced by CLI/output consumers. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/logging/export.py`, `docs/protocol.md`] |

## Standard Stack

### Core
| Library/Module | Version | Purpose | Why Standard |
|---------------|---------|---------|--------------|
| `survarena.benchmark.runner` | repo-local | Orchestrates profile execution, resume, retries, and run status lifecycle. | Existing execution surface already owns phase scope, minimizing migration risk. [VERIFIED: `survarena/benchmark/runner.py`] |
| `survarena.data.splitters` | repo-local (manifest version `1`) | Creates/reuses deterministic splits with manifest payload checks and integrity validation. | Central split-governance locus; should be hardened for D-04 explicit failure semantics. [VERIFIED: `survarena/data/splitters.py`] |
| `survarena.logging.export` | repo-local (ledger schema `2.0`, compact `1.0`) | Persists run ledgers and comparison artifacts including failure summaries. | Structured failure persistence already implemented and aligns with EXEC-04 intent. [VERIFIED: `survarena/logging/export.py`, `docs/protocol.md`] |
| `survarena.run_benchmark` CLI entrypoint | repo-local | Binds benchmark config selection, `--resume`, and `--max-retries`. | Required control plane for deterministic/resume UX. [VERIFIED: `survarena/run_benchmark.py`, `README.md`] |

### Supporting
| Library | Version | Purpose | When to Use |
|--------|---------|---------|-------------|
| `numpy` | `1.26.4` | Seeded numeric ops and index handling in splits/evaluation. | Any split and metric path requiring deterministic array operations. [VERIFIED: `pyproject.toml`, `survarena/data/splitters.py`, `survarena/benchmark/runner.py`] |
| `pandas` | `2.2.2` | Resume source reads and fold/summary dataframe exports. | Resume eligibility scanning and output contracts. [VERIFIED: `pyproject.toml`, `survarena/benchmark/runner.py`, `survarena/logging/export.py`] |
| `scikit-learn` | `1.6.1` | Stratified split generation (`StratifiedKFold`, `train_test_split`). | Deterministic split generation with seeded stratification. [VERIFIED: `pyproject.toml`, `survarena/data/splitters.py`] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Reusing `survarena.data.splitters` manifest pipeline | External workflow/orchestrator split cache | Introduces cross-system state complexity without phase value; phase scope is internal deterministic contract hardening. [VERIFIED: `.planning/ROADMAP.md`, `survarena/data/splitters.py`] |
| Existing JSONL run ledger format | External DB-backed run-state service | Overkill for Phase 1 and violates current disk-first artifact contract. [VERIFIED: `docs/protocol.md`, `survarena/logging/export.py`] |

**Installation:**
```bash
python -m pip install -e ".[dev]"
```

**Version verification notes:** Project pins versions in `pyproject.toml`; phase planning should honor these pins rather than introducing new dependencies. [VERIFIED: `pyproject.toml`]

## Architecture Patterns

### System Architecture Diagram

```text
CLI (`python -m survarena.run_benchmark`)
  -> load benchmark YAML + parse profile/resume flags
  -> validate deterministic profile contract (NEW strict gate)
  -> load/create splits via manifest policy
      -> if manifest matches: reuse splits
      -> if manifest mismatches: hard-fail with explicit regenerate path (D-04)
  -> iterate dataset/method/split(+track)
      -> evaluate split with seeded execution
      -> emit success/failure run payload
      -> retry failed payloads up to budget
  -> export fold/summary/ledger/failure artifacts
  -> on resume: skip only outputs passing eligibility checks (status + required fields/files)
```

Data enters from benchmark YAML and dataset loaders, is transformed through deterministic split/evaluation loops, and exits as structured artifacts under `results/summary/exp_*`. [VERIFIED: `survarena/run_benchmark.py`, `survarena/benchmark/runner.py`, `survarena/data/splitters.py`, `survarena/logging/export.py`, `docs/protocol.md`]

### Recommended Project Structure
```text
survarena/
├── run_benchmark.py           # CLI flags and config loading
├── benchmark/runner.py        # deterministic orchestration + resume eligibility
├── data/splitters.py          # manifest policy + split integrity
└── logging/export.py          # run ledger + failure artifact export
tests/
├── test_benchmark_runner.py   # profile+manifest determinism + resume/retry/EXEC-04
```

### Pattern 1: Split Manifest as Deterministic Contract
**What:** Treat manifest payload equality as a reproducibility guard for split reuse, with explicit validation of split integrity and stratification. [VERIFIED: `survarena/data/splitters.py`]  
**When to use:** Every benchmark run that loads or creates splits for repeated/fixed split strategies. [VERIFIED: `survarena/benchmark/runner.py`]

**Example:**
```python
# Source: survarena/data/splitters.py
manifest_payload = _expected_split_manifest_payload(...)
if manifest_path.exists():
    manifest = read_split_manifest(manifest_path)
    if manifest.get("manifest_payload") == manifest_payload:
        loaded_splits = [read_split(...) for split_id in split_ids]
        _validate_split_integrity(loaded_splits, n_samples)
        _validate_event_stratification(loaded_splits, event)
        return loaded_splits
```

### Pattern 2: Resume by Success-Key Preservation
**What:** Build a completed-key set from successful prior rows and skip only those keys during reruns. [VERIFIED: `survarena/benchmark/runner.py`]  
**When to use:** Resume mode with an existing output directory and partial prior results. [VERIFIED: `survarena/run_benchmark.py`, `survarena/benchmark/runner.py`]

**Example:**
```python
# Source: survarena/benchmark/runner.py
if resume and existing_fold_results.exists():
    existing = pd.read_csv(existing_fold_results)
    for row in existing.to_dict(orient="records"):
        if row.get("status") == "success":
            completed_keys.add((dataset_id, method_id, split_id, seed))
```

### Anti-Patterns to Avoid
- **Silent split regeneration on mismatch:** regenerating splits when manifest payload changes breaks deterministic comparability and violates D-04; fail explicitly and require operator action. [VERIFIED: `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`, `survarena/data/splitters.py`]
- **Status-only resume eligibility:** `status=success` without required-field/file checks can skip corrupted partial work. [VERIFIED: `survarena/benchmark/runner.py`, `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]
- **Untracked seed paths:** any model/tuner randomness path not explicitly seeded undermines deterministic replay and D-05. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/benchmark/tuning.py`, `survarena/utils/seeds.py`, `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Split persistence and deterministic reuse | Ad-hoc custom cache keyed by filename conventions | Existing split manifest payload + split-id set in `survarena.data.splitters` | Already validates integrity/stratification and records seed policy; extending this is lower risk than replacing it. [VERIFIED: `survarena/data/splitters.py`] |
| Resume checkpoint ledger | New custom checkpoint DB service | Existing fold results + run ledger artifacts (`*_fold_results.csv`, `*_run_records*.jsonl.gz`) | Current artifacts already preserve status/failure metadata needed for resumable semantics. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/logging/export.py`, `docs/protocol.md`] |
| Failure audit trail | Free-form logs only | Structured `failure` payload + failure summary exports | Structured records are queryable and align with EXEC-04 success criteria. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/logging/export.py`, `.planning/ROADMAP.md`] |

**Key insight:** This phase should refine contract enforcement around existing deterministic/resume primitives, not replace the primitives themselves. [VERIFIED: `.planning/ROADMAP.md`, `survarena/benchmark/runner.py`, `survarena/data/splitters.py`]

## Common Pitfalls

### Pitfall 1: Manifest Drift Accepted Implicitly
**What goes wrong:** Dataset/event/seed/fold changes can produce a different expected manifest, but runtime silently regenerates splits, making runs non-comparable across interruptions. [VERIFIED: `survarena/data/splitters.py`, `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]  
**Why it happens:** Current logic falls through to split regeneration whenever manifest payload mismatch occurs. [VERIFIED: `survarena/data/splitters.py`]  
**How to avoid:** Add strict mode default for benchmark profiles that raises a deterministic contract error on mismatch and prints explicit regeneration instructions. [VERIFIED: `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]  
**Warning signs:** Existing manifest file present, expected payload differs, and new split files get rewritten during resume-oriented workflows. [VERIFIED: `survarena/data/splitters.py`]

### Pitfall 2: Resume Marks Corrupt Rows as Complete
**What goes wrong:** A prior row with `status=success` but missing mandatory metrics/files can be skipped, losing chance to recompute valid output. [VERIFIED: `survarena/benchmark/runner.py`, `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]  
**Why it happens:** Completion key currently checks status and tuple key, not full output integrity. [VERIFIED: `survarena/benchmark/runner.py`]  
**How to avoid:** Introduce explicit resume-eligibility validator requiring `status=success` plus required fields (`benchmark_id`, `dataset_id`, `method_id`, `split_id`, `seed`, `primary_metric`) and metric validity checks before adding to completed set. [VERIFIED: `survarena/benchmark/runner.py`, `docs/protocol.md`]  
**Warning signs:** Resume skips records where downstream summaries/ledgers show missing metric fields or NaN-heavy rows unexpectedly labeled success. [VERIFIED: `survarena/logging/export.py`, `docs/protocol.md`]

### Pitfall 3: Retry Semantics Overwrite Failure Evidence
**What goes wrong:** Failure context can be obscured if retry loops do not preserve per-attempt payloads and counts. [VERIFIED: `survarena/benchmark/runner.py`]  
**Why it happens:** Retry loops append records per attempt; downstream planners might accidentally collapse attempts in refactors. [VERIFIED: `survarena/benchmark/runner.py`]  
**How to avoid:** Keep per-attempt `retry_attempt`, `status`, and `failure` fields in ledger and ensure no dedupe during export. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/logging/export.py`]  
**Warning signs:** `*_run_records.jsonl.gz` attempt count lower than observed retries in console output. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/logging/export.py`]

## Code Examples

Verified patterns from repository sources:

### Deterministic seed fan-out into split evaluation
```python
# Source: survarena/benchmark/runner.py + survarena/benchmark/tuning.py
set_global_seed(split.seed)
fold_cache = prepare_inner_cv_cache(..., seed=split.seed)
model = get_method_class(method_id)(**resolve_runtime_method_params(best_params, seed=split.seed))
```

### Structured failure payload preservation
```python
# Source: survarena/benchmark/runner.py
except Exception as exc:
    run_payload = {
        "manifest": manifest.to_dict(),
        "metrics": {"status": "failed", "failure_type": type(exc).__name__, ...},
        "failure": {"traceback": tb_str},
    }
```

### Retry loop with attempt tracking
```python
# Source: survarena/benchmark/runner.py
attempt = 0
while True:
    record = evaluate_split(...)
    run_payload["metrics"]["retry_attempt"] = int(attempt)
    if record["status"] == "success" or attempt >= max_retries:
        break
    attempt += 1
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Re-run everything after interruption | Resume mode with completed-key skipping and bounded retries | Already present in current branch state | Major runtime savings; now needs stricter completion integrity checks for Phase 1 contract. [VERIFIED: `survarena/run_benchmark.py`, `survarena/benchmark/runner.py`, `.planning/ROADMAP.md`] |
| Unstructured benchmark logs as primary debugging signal | Structured run ledgers + failure summaries + compact index metadata | Already present in current branch state | Enables auditable failure inspection and machine-readable postmortems. [VERIFIED: `survarena/logging/export.py`, `docs/protocol.md`] |
| Implicit tolerance for split regeneration | Locked decision requiring explicit hard failure on manifest mismatch | Defined in Phase 1 context | Required to preserve deterministic comparability across profile tiers. [VERIFIED: `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`] |

**Deprecated/outdated for this phase:**
- Blindly trusting `status=success` as sufficient resume proof is outdated against D-06; move to success+integrity eligibility. [VERIFIED: `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`, `survarena/benchmark/runner.py`]

## Assumptions Log

> Claims tagged `[ASSUMED]` require confirmation before locking implementation decisions.

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A dedicated `--regenerate-splits` flag should be the preferred explicit regeneration UX. | Open Questions | Medium - planner may allocate CLI work that user may prefer as operational/manual policy. |
| A2 | `pytest tests/test_benchmark_runner.py -x` is the best default quick validation command for this phase. | Validation Architecture | Low - only affects dev feedback loop efficiency, not core behavior correctness. |

## Open Questions (RESOLVED)

1. **What is the minimal required field set for "resume-complete" eligibility?**
   - What we know: D-06 requires `status=success` plus valid required outputs. [VERIFIED: `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]
   - Resolution: Treat a row as resume-complete only when `status=success` and required identifiers/outputs are present and valid: `benchmark_id`, `dataset_id`, `method_id`, `split_id`, `seed`, and non-empty primary metric field; rows missing any required value are ineligible and must be rerun.
   - Implementation lock: Use one centralized eligibility validator in `survarena/benchmark/runner.py` and enforce via `tests/test_benchmark_runner.py` to keep fold CSV and ledger semantics aligned. [RESOLVED]

2. **How should explicit split-regeneration be requested by users?**
   - What we know: D-04 forbids automatic regeneration on mismatch. [VERIFIED: `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`]
   - Resolution: Add an explicit CLI/runtime switch (`--regenerate-splits`) that is off by default; on manifest mismatch, default behavior is hard failure with actionable instruction to rerun with this switch when regeneration is intentional.
   - Implementation lock: Thread the explicit regeneration flag from CLI to runner to split loader so the policy is scriptable, auditable, and deterministic by default. [RESOLVED]

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `python3` | Benchmark runtime + tests | ✓ | 3.13.2 | Use supported 3.10/3.11/3.12 virtualenv if incompatibility appears. [VERIFIED: Shell `python3 --version`, `README.md`, `pyproject.toml`] |
| `pip3` | Editable/dev install | ✓ | 25.0 | Use `python -m pip` in project venv. [VERIFIED: Shell `pip3 --version`, `README.md`] |
| `pytest` | Validation architecture | ✓ (version drift) | 7.4.4 | Reinstall dev deps in `.venv` to satisfy `>=8.3,<9`. [VERIFIED: Shell `pytest --version`, `pyproject.toml`] |
| `ruff` | Lint quality gate | ✓ (version drift) | 0.14.10 | Reinstall dev deps in `.venv` to satisfy `>=0.15,<0.16`. [VERIFIED: Shell `ruff --version`, `pyproject.toml`] |
| `survarena` CLI entrypoint | CLI-level manual verification | ✗ (global shell) | — | Use module invocation: `python -m survarena.run_benchmark ...` from repo. [VERIFIED: Shell command check, `README.md`] |

**Missing dependencies with no fallback:**
- None identified for planning-level work.

**Missing dependencies with fallback:**
- `survarena` command not globally installed; use module entrypoint and/or local editable install in `.venv`. [VERIFIED: Shell command check, `README.md`]

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest` (configured in project, required `>=8.3,<9`) [VERIFIED: `pyproject.toml`] |
| Config file | `pyproject.toml` (`[tool.pytest.ini_options]`) [VERIFIED: `pyproject.toml`] |
| Quick run command | `pytest tests/test_benchmark_runner.py -x` [ASSUMED] |
| Full suite command | `pytest` [VERIFIED: `README.md`] |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXEC-01 | Profile tiers enforce deterministic split governance and seeded execution contract | unit/integration | `pytest tests/test_benchmark_runner.py -k "not exec04" -x` | ❌ Wave 0 |
| EXEC-04 | Resume skips only valid successful work, retries failures within budget, preserves failure records | unit/integration | `pytest tests/test_benchmark_runner.py -k "exec04" -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_benchmark_runner.py -x` [ASSUMED]  
- **Per wave merge:** `pytest` [VERIFIED: `README.md`]  
- **Phase gate:** Full suite green before `/gsd-verify-work`. [VERIFIED: `AGENTS.md` quality gate + roadmap requirements context]

### Wave 0 Gaps
- [ ] `tests/test_benchmark_runner.py` - EXEC-01 (profile + split manifest) and EXEC-04 (resume / retry / failure preservation) coverage; use `-k "not exec04"` vs `-k exec04` to focus.
- [ ] Add shared fixtures (temporary output dir, synthetic fold rows, split manifests) if duplication emerges.
- [ ] Dev environment alignment: `python -m pip install -e ".[dev]"` in local `.venv` to satisfy pinned test/lint tool versions. [VERIFIED: `pyproject.toml`, `README.md`]

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Not in scope for local benchmark CLI phase. [VERIFIED: `survarena/run_benchmark.py`, `.planning/ROADMAP.md`] |
| V3 Session Management | no | Not in scope for stateless CLI run orchestration. [VERIFIED: `survarena/run_benchmark.py`] |
| V4 Access Control | no | No role-based control plane in current phase boundary. [VERIFIED: `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`, `survarena/run_benchmark.py`] |
| V5 Input Validation | yes | Explicit argument/config validation and split integrity checks (`ValueError` + index/stratification checks). [VERIFIED: `survarena/run_benchmark.py`, `survarena/data/splitters.py`] |
| V6 Cryptography | no | No new crypto controls needed for deterministic execution/resume semantics in this phase. [VERIFIED: `.planning/ROADMAP.md`, `survarena/benchmark/runner.py`] |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Tampered or stale split cache leads to unfair comparisons | Tampering | Manifest payload validation + explicit mismatch hard-fail + integrity checks. [VERIFIED: `survarena/data/splitters.py`, `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`] |
| Corrupt resume source causes false "completed" skips | Tampering/Repudiation | Resume eligibility validator requiring status and required-output integrity before skip. [VERIFIED: `survarena/benchmark/runner.py`, `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md`] |
| Lost failure traceability after retries | Repudiation | Preserve structured `failure` payload and per-attempt `retry_attempt` in run ledger exports. [VERIFIED: `survarena/benchmark/runner.py`, `survarena/logging/export.py`] |

## Sources

### Primary (HIGH confidence)
- `.planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md` - locked decisions D-01..D-07 and phase scope.  
- `.planning/ROADMAP.md` - Phase 1 goal, requirements mapping, and success criteria.  
- `.planning/REQUIREMENTS.md` - EXEC-01 and EXEC-04 requirement definitions.  
- `docs/protocol.md` - profile intent, reproducibility contract, and output artifacts.  
- `survarena/benchmark/runner.py` - execution loop, resume/retry behavior, status/failure payloads.  
- `survarena/data/splitters.py` - split manifest, deterministic split generation, integrity checks.  
- `survarena/run_benchmark.py` - CLI options for benchmark config, resume, retries.  
- `survarena/logging/export.py` - structured run ledger and failure-summary artifact paths.  
- `pyproject.toml` - dependency/tooling constraints and pytest/ruff configuration.  
- `README.md` - runtime support matrix and benchmark execution commands.

### Secondary (MEDIUM confidence)
- Shell environment audit outputs (`python3`, `pip3`, `pytest`, `ruff`, `survarena` availability) captured during this research session.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - directly anchored to current repository modules and config pins.  
- Architecture: HIGH - derived from active benchmark execution and split/ledger code paths.  
- Pitfalls: HIGH - grounded in observed control flow against locked phase decisions.

**Research date:** 2026-04-23  
**Valid until:** 2026-05-23 (30 days, unless major benchmark runner/splitter refactor lands first)
