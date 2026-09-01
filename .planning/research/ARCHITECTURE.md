# Architecture Research

**Domain:** Reproducible tabular survival-analysis benchmarking
**Researched:** 2026-08-31
**Confidence:** HIGH

## Standard Architecture

### System Overview

```text
Frozen release recipe
  ProtocolSpec + DatasetSpec + MethodSpec + EnvironmentSpec
                              |
                              v
Orchestrator
  dataset context + split indices -> immutable RunUnit records
                              |
                              v
Killable run worker
  preprocess -> tune -> fit -> validate predictions -> score
                              |
                              v
Canonical result collection
  runs + metrics + failures + timings + predictions/digests + provenance
                              |
                              v
Pure derived reports
  common-support performance + reliability + cost/Pareto views
```

### Component Responsibilities

| Component | Responsibility | Implementation Direction |
|-----------|----------------|--------------------------|
| Protocol compiler | Strict validation and stable recipe digest | Promote `core.protocol.ProtocolSpec`; fail closed on ambiguous values. |
| Dataset registry | Frozen acquisition, metadata, groups, checksums | Add version/license/domain/event/follow-up and separate groups/row IDs. |
| Model contract | Capabilities, fit inputs, output semantics | Promote `ModelCapabilities`; adapters declare and verify outputs/budgets. |
| Split service | Outer/inner/early-stop partitions and manifests | One group-aware service with algorithm version and event/censoring minima. |
| Run worker | Preprocess, tune, fit, validate, score under hard limits | Disposable subprocess with explicit threads, memory, and timeout policy. |
| Metric kernel | Censoring-aware metrics and estimability | Reference-tested functions with frozen horizons/support. |
| Comparison kernel | Cell aggregation, support policy, inference | Dataset-level summaries and complete-block post-hoc tests. |
| Result store | Immutable canonical evidence | SQLite transactions and provenance-conflict rejection. |
| Report layer | Pure views over canonical evidence | No direct file discovery or recomputation from ad hoc CSV paths. |

## Recommended Project Structure

```text
survarena/
├── core/
│   ├── protocol/       # strict release recipe
│   ├── datasets/       # frozen dataset identity and groups
│   ├── models/         # capabilities and prediction contracts
│   ├── execution/      # run units, budgets, subprocess isolation
│   └── results/        # immutable schema and canonical store
├── evaluation/          # verified metrics and dataset-level statistics
├── methods/             # adapters only; no benchmark policy
├── benchmark/           # protocol compiler/orchestration facade
└── reporting/           # store-derived exports and figures
```

The migration remains strangler-style: add authoritative interfaces, port one vertical sentinel path, then remove legacy dictionary/CSV policy only after parity tests pass.

## Architectural Patterns

### Frozen Recipe / Content-Addressed Identity

Normalize every protocol, dataset, method, split, environment, and code input; hash the complete recipe; make it part of run identity. A duplicate recipe is idempotent. The same natural cell under different provenance is a different run or an explicit conflict, never a silent overwrite.

### Disposable Worker Boundary

The parent process owns scheduling and durable writes. Each worker receives immutable indices/config, caps native threads, performs one bounded unit, and exits. Timeout/memory termination happens outside the worker.

### Two-Stage Aggregation

Fold/repeat rows estimate performance within a dataset. Statistical comparison across algorithms operates on one paired method effect per dataset. This prevents pseudo-replication while preserving fold diagnostics.

### Pure Reporting

Reports query one collection and never infer canonical evidence from directory names. Regeneration is deterministic and records the query, schema, and source digest.

## Data Flow

1. **Compile:** YAML/reference IDs -> validated specs -> normalized recipe digest.
2. **Plan:** dataset snapshot + groups -> frozen split hierarchy -> run units.
3. **Execute:** run unit -> disposable worker -> validated predictions -> metrics/failure.
4. **Persist:** transactionally append provenance-complete result; reject conflicts.
5. **Compare:** common-support dataset cells -> ranks/effects/tests/reliability.
6. **Publish:** regenerate reports and release manifest from store digest.

## Scaling Considerations

| Scale | Architecture Adjustment |
|-------|-------------------------|
| Sentinel / PR | Single scheduler, tiny fixtures, ephemeral store. |
| Medium core suite | Local process scheduler, dataset context initialized once per worker, explicit native thread limits. |
| CI-sharded reference | Each shard writes a validated partial collection; merge rejects provenance conflicts. |
| Community tier | Add submission validation only after store/protocol schemas are frozen. |

The first bottlenecks are DataFrame serialization and nested model parallelism, not database throughput.

## Anti-Patterns

- **Policy inside adapters:** adapters implement declared capabilities; they do not choose benchmark horizons, missingness rules, or comparison eligibility.
- **Filename-as-database:** directory layout is an export concern, not result identity.
- **Silent compatibility:** cache/config migrations explicitly report what changed.
- **One monolithic runner rewrite:** port a complete CoxPH sentinel path first, then broaden coverage.

## Integration Points

| Boundary | Contract |
|----------|----------|
| Config -> protocol | Strict parse, cross-field validation, normalized digest. |
| Dataset -> splits | Features, targets, groups, row IDs, version, checksum, algorithm version. |
| Protocol -> adapter | Capabilities, validated parameters, declared budgets and output type. |
| Adapter -> metrics | Validated risk/survival bundle with explicit time grid and orientation. |
| Worker -> store | Immutable `RunResult` plus complete provenance/failure metadata. |
| Store -> report | Query-only derived views with common-support declaration. |

## Sources

- Repository codebase architecture and 2026-08-31 audit.
- https://arxiv.org/abs/2506.16791 — living benchmark architecture and maintenance model.
- https://www.jmlr.org/papers/v7/demsar06a.html — multiple-dataset comparison structure.
- Official scikit-survival metric documentation.

---
*Architecture research for: SurvArena v2 verified benchmark kernel*
*Researched: 2026-08-31*
