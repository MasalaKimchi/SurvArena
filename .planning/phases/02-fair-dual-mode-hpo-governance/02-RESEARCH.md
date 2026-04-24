# Phase 2: Fair Dual-Mode HPO Governance - Research

**Researched:** 2026-04-23
**Domain:** Benchmark orchestration fairness, HPO budget governance, and run-ledger accountability for SurvArena
**Confidence:** MEDIUM

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
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

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXEC-02 | User can run every selected model in both no-HPO and HPO modes under the same benchmark profile. | Dual-mode pairing contract, deterministic no-HPO->HPO ordering, and same split/seed execution unit design. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md] |
| EXEC-03 | User can enforce a uniform HPO budget policy and inspect realized budget usage per model run. | Central budget-policy normalization and run-level requested-vs-realized budget telemetry in run payloads and HPO exports. [VERIFIED: .planning/REQUIREMENTS.md] [VERIFIED: survarena/benchmark/runner.py] [VERIFIED: survarena/benchmark/tuning.py] [VERIFIED: survarena/logging/export.py] |
</phase_requirements>

## Summary

Phase 2 should be implemented as a governance layer on top of existing benchmark orchestration, not a new benchmarking engine. `run_benchmark()` already centralizes config ingestion and split/method loops, and `evaluate_split()` already emits per-run `hpo_status`, `hpo_trial_count`, `hpo_backend`, `hpo_metadata`, and `hpo_trials`, which is the right insertion point for explicit dual-mode labels and budget accounting. [VERIFIED: survarena/benchmark/runner.py]

The key planning move is to define a single run-unit contract `(benchmark_id, dataset_id, method_id, split_id, seed, hpo_mode)` and enforce hard parity in downstream summaries before comparative claims are produced. This aligns exactly with D-01 through D-04 and avoids silent fairness drift when one mode fails or is skipped. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md] [VERIFIED: docs/protocol.md]

Budget governance should stay config-driven and uniform at benchmark scope (single policy object for all methods) with explicit exceptions recorded as status, not hidden behavior. Optuna supports explicit `n_trials` and `timeout` limits; current SurvArena HPO config already normalizes these fields and stores realized trial counts, so Phase 2 is primarily policy enforcement plus reporting rigor. [VERIFIED: survarena/benchmark/tuning.py] [CITED: https://context7.com/optuna/optuna/llms.txt]

**Primary recommendation:** Implement dual-mode execution as two deterministic passes per pairing unit with a single centralized `hpo_policy` contract and parity-gated comparative aggregation. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md]

## User Constraints

- Locked decisions require paired no-HPO/HPO on identical dataset/split/seed with no-HPO first, canonical shared artifacts, and hard parity gating for comparisons. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md]
- Scope is constrained to EXEC-02 and EXEC-03 outcomes and must preserve strict profile/determinism guarantees inherited from Phase 1. [VERIFIED: .planning/ROADMAP.md] [VERIFIED: .planning/phases/01-deterministic-execution-foundation/01-CONTEXT.md]

## Project Constraints (from .cursor/rules/)

- No `.cursor/rules/` directives were found in this repository, so no extra project rule constraints apply beyond existing planning documents. [VERIFIED: Glob .cursor/rules/**/*.md]

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Dual-mode run generation (no-HPO + HPO) | API / Backend | Database / Storage | Runner owns orchestration loops and ledger payload creation; storage only persists emitted records. [VERIFIED: survarena/benchmark/runner.py] |
| Uniform HPO budget policy normalization | API / Backend | — | `_parse_hpo_config` is the canonical normalization path for `max_trials`, `timeout_seconds`, sampler, pruner. [VERIFIED: survarena/benchmark/tuning.py] |
| Requested vs realized budget telemetry | API / Backend | Database / Storage | Tuning computes realized `trial_count`; exports persist `hpo_trials.csv` and `hpo_summary.json`. [VERIFIED: survarena/benchmark/tuning.py] [VERIFIED: survarena/logging/export.py] |
| Parity eligibility gating for comparisons | API / Backend | Database / Storage | Comparative artifacts are generated after fold results; parity filters should be applied before these exports. [VERIFIED: survarena/benchmark/runner.py] [VERIFIED: survarena/logging/export.py] |
| Fairness labeling in canonical artifacts | API / Backend | Database / Storage | Shared artifact set already exists; add mode label fields instead of separate outputs. [VERIFIED: docs/protocol.md] [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md] |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SurvArena benchmark runner (`survarena/benchmark/runner.py`) | repo-local | Canonical orchestration for datasets/methods/splits and run-ledger generation. | This is already the single execution control plane in the codebase, so governance belongs here. [VERIFIED: survarena/benchmark/runner.py] |
| SurvArena tuning layer (`survarena/benchmark/tuning.py`) | repo-local | Canonical HPO policy parsing and Optuna execution metadata. | Existing parse/normalize + realized trial capture already solves most budget accounting requirements. [VERIFIED: survarena/benchmark/tuning.py] |
| Optuna | 4.8.0 (published 2026-03-16) | Native HPO backend for search-space methods. | Supports explicit trial/time budgets, seeded samplers, and pruners needed for uniform policy enforcement. [VERIFIED: PyPI JSON optuna] [CITED: https://context7.com/optuna/optuna/llms.txt] |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| AutoGluon Tabular | 1.5.0 (published 2025-12-19) | AutoML path with independent tuning controls (`time_limit`, `hyperparameter_tune_kwargs`). | Use when `method_id == autogluon_survival`; normalize into same benchmark budget policy surface for fairness reporting. [VERIFIED: PyPI JSON autogluon.tabular] [VERIFIED: survarena/benchmark/runner.py] [CITED: https://context7.com/autogluon/autogluon/llms.txt] |
| Pandas | 2.2.2 (pinned in project) | Aggregation/export surface for fold and summary artifacts. | Use for parity gating and budget report tables before export. [VERIFIED: pyproject.toml] [VERIFIED: survarena/logging/export.py] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Optuna-native search for non-AutoGluon methods | Custom random/grid loops | Reimplements sampling/pruning/metadata with high bug risk and weak reproducibility. [VERIFIED: survarena/benchmark/tuning.py] [ASSUMED] |
| Single shared artifact set with mode labels | Separate no-HPO and HPO output trees | Easier short-term implementation but violates locked D-02 and increases mismatch risk. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md] |

**Installation:**
```bash
python3 -m pip install -e ".[dev]"
python3 -m pip install optuna==4.8.0
```

**Version verification:** Latest version and publish date were verified against PyPI JSON endpoints during this research. [VERIFIED: PyPI JSON optuna] [VERIFIED: PyPI JSON autogluon.tabular]

## Architecture Patterns

### System Architecture Diagram

```text
Benchmark YAML (profile + hpo policy)
          |
          v
run_benchmark() config normalization
          |
          +----> Split loader (deterministic split/seed set)
          |
          v
for each (dataset, method, split, seed):
  pass A: evaluate_split(hpo_mode=no_hpo, policy forcing HPO disabled)
  pass B: evaluate_split(hpo_mode=hpo, policy forcing HPO enabled)
          |
          v
run_payload rows with:
  - requested policy (max_trials, timeout, sampler, pruner)
  - realized usage (trial_count, status, backend)
  - mode label + parity key
          |
          v
parity gate (drop comparison-ineligible units)
          |
          +----> comparative exports (leaderboard/significance/etc)
          |
          +----> governance exports (hpo_trials.csv + hpo_summary.json + parity report)
```

### Recommended Project Structure
```text
survarena/
├── benchmark/
│   ├── runner.py          # dual-mode orchestration + parity key emission
│   └── tuning.py          # policy normalization and realized HPO metadata
├── logging/
│   └── export.py          # parity-gated summaries and budget usage exports
└── api/
    └── compare.py         # optional user-facing mirror of dual-mode contract
tests/
├── test_hpo_config.py     # policy parsing and coercion behavior
├── test_compare_api.py    # compare entrypoint parity wiring
└── test_*.py              # new dual-mode parity and budget governance tests
```

### Pattern 1: Centralized Policy Normalization
**What:** Parse and normalize one benchmark-level HPO policy object once, then pass to all run executions. [VERIFIED: survarena/benchmark/runner.py] [VERIFIED: survarena/benchmark/tuning.py]
**When to use:** Every benchmark run, before entering dataset/method/split loops. [VERIFIED: survarena/benchmark/runner.py]
**Example:**
```python
# Source: https://context7.com/optuna/optuna/llms.txt
study.optimize(objective, n_trials=100, timeout=300)
```

### Pattern 2: Explicit Mode-Labeled Canonical Rows
**What:** Keep one canonical artifact set and label each row with execution mode and parity eligibility. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md]
**When to use:** Every fold/run record and all downstream summary frames. [VERIFIED: docs/protocol.md] [VERIFIED: survarena/logging/export.py]
**Example:**
```python
# Source: internal SurvArena pattern (run_payload shaping in runner.py)
run_payload["metrics"]["hpo_mode"] = "no_hpo"  # or "hpo"
run_payload["metrics"]["parity_key"] = f"{dataset_id}|{method_id}|{split_id}|{seed}"
```

### Anti-Patterns to Avoid
- **Implicit mixed-mode comparisons:** Computing pairwise/leaderboard results without checking that both modes exist per parity key. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md]
- **Method-specific hidden budget rules:** Allowing some models to use different effective budget semantics without explicit policy and telemetry. [VERIFIED: .planning/REQUIREMENTS.md] [ASSUMED]
- **Silent fallback from HPO to defaults:** Treating missing Optuna or empty search spaces as "success" without clear status surfaces in governance views. [VERIFIED: survarena/benchmark/tuning.py]

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Hyperparameter search engine | Custom trial scheduler/pruner | Optuna samplers + pruners | Optuna already provides reproducible samplers, pruning hooks, and trial history. [CITED: https://context7.com/optuna/optuna/llms.txt] |
| HPO trial ledger format | New bespoke file schema | Existing `hpo_trials.csv` + `hpo_summary.json` export path | Export functions and schema scaffolding already exist and integrate with experiment output contract. [VERIFIED: survarena/logging/export.py] |
| Cross-run artifact duplication logic | Separate per-mode artifact trees | Existing shared canonical output directory with added mode labels | Locked D-02 explicitly requires one artifact set and per-row mode labels. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md] |

**Key insight:** Phase 2 is mostly governance and schema discipline over existing orchestration primitives, not invention of new optimization or export infrastructure. [VERIFIED: survarena/benchmark/runner.py] [VERIFIED: survarena/benchmark/tuning.py] [VERIFIED: survarena/logging/export.py]

## Common Pitfalls

### Pitfall 1: "Enabled HPO" does not mean HPO actually ran
**What goes wrong:** Runs are tagged as HPO-enabled but execute default params because `search_space` is absent or Optuna is missing. [VERIFIED: survarena/benchmark/tuning.py]
**Why it happens:** `_parse_hpo_config` disables when no search space, and missing Optuna yields `optuna_missing` status. [VERIFIED: survarena/benchmark/tuning.py]
**How to avoid:** Require mode-specific status checks (`hpo_mode == "hpo"` plus `hpo_status in {"success","no_valid_trial"}` with explicit policy exception handling). [ASSUMED]
**Warning signs:** High count of `hpo_status=disabled` or `optuna_missing` in HPO-designated mode rows. [VERIFIED: survarena/benchmark/tuning.py]

### Pitfall 2: Non-comparable pairs pollute aggregate claims
**What goes wrong:** One mode fails for a split and aggregate statistics still include the surviving mode. [ASSUMED]
**Why it happens:** Existing export pipeline aggregates all rows unless filtered first. [VERIFIED: survarena/logging/export.py]
**How to avoid:** Build and enforce a parity eligibility mask before every comparative export call. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md]
**Warning signs:** Unequal per-method row counts between modes for same dataset/split/seed keys. [ASSUMED]

### Pitfall 3: Budget policy drifts across backends
**What goes wrong:** AutoGluon and native methods expose different requested budget fields, making fairness auditing ambiguous. [VERIFIED: survarena/benchmark/runner.py] [VERIFIED: survarena/api/compare.py]
**Why it happens:** Separate config surfaces (`autogluon.*` and `hpo.*`) currently coexist. [VERIFIED: configs/benchmark/cloud_comprehensive_all_models_hpo.yaml] [VERIFIED: configs/benchmark/standard_v1.yaml]
**How to avoid:** Add a normalized "requested budget contract" record in every run payload and map backend-specific fields into it. [ASSUMED]
**Warning signs:** Missing `requested_trials` or `requested_timeout_seconds` in some method rows. [ASSUMED]

## Code Examples

Verified patterns from official sources and current codebase:

### Bounded HPO execution
```python
# Source: https://context7.com/optuna/optuna/llms.txt
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, timeout=300)
```

### Existing SurvArena realized-budget capture point
```python
# Source: survarena/benchmark/tuning.py
"hpo_metadata": {
    "trial_count": int(len(study.trials)),
    "max_trials": int(resolved_hpo["max_trials"]),
    "timeout_seconds": resolved_hpo["timeout_seconds"],
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-mode benchmark profiles (either HPO enabled or disabled per profile) | Profiles can specify HPO config, but dual-mode parity governance is not yet explicit | Current as of this phase kickoff | Need explicit two-pass mode contract to satisfy EXEC-02 fairness goals. [VERIFIED: configs/benchmark/smoke_all_models_no_hpo.yaml] [VERIFIED: configs/benchmark/standard_v1.yaml] [VERIFIED: .planning/REQUIREMENTS.md] |
| Implicit budget interpretation from backend-specific fields | Structured `hpo_metadata`/`hpo_trials` already emitted for native runs | Already in runner+tuning | Strong base for requested-vs-realized reporting once policy schema is unified. [VERIFIED: survarena/benchmark/runner.py] [VERIFIED: survarena/benchmark/tuning.py] |

**Deprecated/outdated:**
- Treating no-HPO and HPO as separate benchmark artifacts for comparison should be considered outdated for this project because D-02 locks a single canonical artifact set. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md]

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | A normalized per-run requested budget schema can cover both native and AutoGluon flows without major restructuring. | Architecture Patterns / Pitfalls | Medium: could require larger refactor in runner+compare payload shaping. |
| A2 | Parity-gating should execute before all comparative exports, including significance and ELO. | Common Pitfalls | Medium: if some exports intentionally use non-parity rows, gating strategy must be scoped. |
| A3 | Missing mode counterpart should remain in raw ledgers but excluded from comparison claims. | Summary / Pitfalls | Low: aligns with D-03 intent but needs explicit acceptance criteria language. |

## Open Questions (RESOLVED)

1. **Policy semantics when both trial and timeout limits are present**
   - **Decision:** Enforce combined-stop semantics: both `max_trials` and `timeout_seconds` remain active in one run, and the run stops at whichever limit is reached first (native Optuna behavior). There is no precedence mode switch.
   - **Implementation contract:** Always emit both requested fields (`requested_max_trials`, `requested_timeout_seconds`) plus a realized field (`realized_trial_count`) in run/export artifacts so the terminating condition is auditable. [VERIFIED: survarena/benchmark/tuning.py] [CITED: https://context7.com/optuna/optuna/llms.txt]

2. **AutoGluon-to-unified budget mapping**
   - **Decision:** Keep one canonical governance schema across backends. Backend-specific knobs are translated internally, but artifacts must expose the same canonical fields:
     - `requested_max_trials`
     - `requested_timeout_seconds`
     - `requested_sampler`
     - `requested_pruner`
     - `realized_trial_count`
   - **Mapping path:** For legacy/native rows where realized count currently appears as `hpo_metadata.trial_count`, map it to `realized_trial_count` at run-payload shaping and export stages; keep `trial_count` only as a backward-compatible alias during transition.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | benchmark + tests | ✓ | 3.13.2 | — |
| pytest | validation architecture | ✓ (below project target) | 7.4.4 | run via project venv after `pip install -e ".[dev]"` |
| ruff | lint gate (QUAL-01) | ✓ (below project target) | 0.14.10 | run via project venv after `pip install -e ".[dev]"` |
| Optuna | native HPO mode for EXEC-02/03 | ✗ | — | None for native HPO fairness goals (blocking) |
| AutoGluon Tabular | `autogluon_survival` path | ✗ | — | Exclude AutoGluon methods or run native-only subset |

**Missing dependencies with no fallback:**
- `optuna` for native HPO governance in this phase. [VERIFIED: local environment audit]

**Missing dependencies with fallback:**
- `autogluon.tabular` can be treated as optional if benchmark method set excludes `autogluon_survival`. [VERIFIED: local environment audit] [VERIFIED: configs/benchmark/cloud_comprehensive_all_models_hpo.yaml]

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest` (configured in `pyproject.toml`) [VERIFIED: pyproject.toml] |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` [VERIFIED: pyproject.toml] |
| Quick run command | `pytest tests/test_hpo_config.py tests/test_compare_api.py -x` |
| Full suite command | `pytest` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXEC-02 | Every selected model runs in both no-HPO and HPO modes under same profile | unit + integration | `pytest tests/test_dual_mode_hpo_governance.py::test_profile_runs_both_modes_for_each_method -x` | ❌ Wave 0 |
| EXEC-03 | Uniform budget policy enforcement and requested-vs-realized usage visibility | unit + integration | `pytest tests/test_dual_mode_hpo_governance.py::test_requested_vs_realized_budget_usage_is_emitted -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_hpo_config.py tests/test_compare_api.py -x`
- **Per wave merge:** `pytest -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_dual_mode_hpo_governance.py` - parity contract, mode labeling, and hard parity gate coverage for EXEC-02.
- [ ] `tests/test_dual_mode_hpo_governance.py` - uniform policy and requested-vs-realized budget accounting for EXEC-03.
- [ ] `tests/test_hpo_config.py` - extend with precedence/normalization tests for combined trial+timeout policy semantics.

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | CLI/local benchmark flow; no auth surface in phase scope. [VERIFIED: survarena/run_benchmark.py] |
| V3 Session Management | no | No session protocol in benchmark execution path. [VERIFIED: survarena/run_benchmark.py] |
| V4 Access Control | no | No role/permission checks in local orchestration path. [VERIFIED: survarena/benchmark/runner.py] |
| V5 Input Validation | yes | Continue fail-fast config validation and strict key coercion in runner/tuning layers. [VERIFIED: survarena/benchmark/runner.py] [VERIFIED: survarena/benchmark/tuning.py] |
| V6 Cryptography | no | Phase scope does not introduce crypto operations. [VERIFIED: phase scope docs] |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Malformed benchmark config causing silent fairness drift | Tampering | Strict schema/default coercion + explicit status fields (`disabled`, `optuna_missing`, etc.). [VERIFIED: survarena/benchmark/tuning.py] |
| Partial mode execution used for claims | Integrity | Hard parity gate before comparative exports. [VERIFIED: .planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md] |
| Unbounded tuning causing resource exhaustion | Denial of Service | Enforce max trials and timeout policy in canonical config path. [VERIFIED: survarena/benchmark/tuning.py] |

## Sources

### Primary (HIGH confidence)
- `survarena/benchmark/runner.py` - orchestration loop, run payload fields, HPO trial export hooks.
- `survarena/benchmark/tuning.py` - HPO policy normalization and Optuna execution metadata.
- `survarena/logging/export.py` - artifact export paths, run-ledger and HPO summary generation.
- `.planning/phases/02-fair-dual-mode-hpo-governance/02-CONTEXT.md` - locked phase decisions and fairness contract.
- `.planning/REQUIREMENTS.md` - EXEC-02 and EXEC-03 requirement definitions.
- [Optuna Context7 docs](https://context7.com/optuna/optuna/llms.txt) - bounded `study.optimize`, sampler/pruner examples.
- [AutoGluon Context7 docs](https://context7.com/autogluon/autogluon/llms.txt) - `TabularPredictor.fit` budget/tuning arguments.
- [PyPI optuna JSON](https://pypi.org/pypi/optuna/json) - latest version and publish timestamp.
- [PyPI autogluon.tabular JSON](https://pypi.org/pypi/autogluon.tabular/json) - latest version and publish timestamp.

### Secondary (MEDIUM confidence)
- `docs/protocol.md` - benchmark output contract and fairness framing in project docs.
- Benchmark profile YAMLs in `configs/benchmark/` - current no-HPO/HPO profile behavior.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: MEDIUM - codebase and package metadata are verified, but environment mismatches mean some execution assumptions remain pending environment setup.
- Architecture: HIGH - insertion points and data flow are directly observed in runner/tuning/export code.
- Pitfalls: MEDIUM - several pitfalls are directly observed; a few governance outcomes depend on not-yet-implemented parity schema decisions.

**Research date:** 2026-04-23
**Valid until:** 2026-05-23
