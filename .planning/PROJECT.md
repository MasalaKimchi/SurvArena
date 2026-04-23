# SurvArena Benchmark Modernization

## What This Is

SurvArena is a Python benchmark toolkit for comparing survival analysis methods across datasets with reproducible, config-driven runs. This project evolves the existing codebase into a TabArena-style, manuscript-grade benchmarking system focused on practitioners choosing robust models under realistic constraints. It emphasizes comparable no-HPO vs HPO evaluations, strong statistical reporting, and compact, non-redundant result storage.

## Core Value

A practitioner can trust one benchmark run to produce fair, statistically robust, and compactly stored model comparisons across diverse survival datasets.

## Requirements

### Validated

- ✓ Users can run benchmark experiments from YAML configs via CLI and Python entrypoints — existing
- ✓ Multiple survival method adapters can be compared through a unified registry/runner pipeline — existing
- ✓ Benchmark outputs and summaries are exported to structured artifacts under `results/summary/` — existing
- ✓ Split generation/reuse and reproducibility-oriented manifests already exist in the current stack — existing

### Active

- [ ] Deliver a comprehensive Python-only survival benchmark spanning multiple packages and methods across a medium-but-diverse dataset suite.
- [ ] Run every selected model in both modes: without HPO and with HPO, using controlled and reproducible settings.
- [ ] Produce manuscript-level statistical robustness outputs (pairwise testing, uncertainty-aware comparisons, and publication-grade summaries).
- [ ] Generate TabArena-like full pairwise matchup analysis and a global ELO leaderboard.
- [ ] Persist each experiment collection into a single comprehensive results file that is compact and avoids redundancy.
- [ ] Enforce code quality gates for touched code (lint/type/test green) and remove dead code/functions encountered in benchmark pathways.

### Out of Scope

- R and other non-Python package execution — project scope is exclusively Python package ecosystems.
- Web dashboard/UI productization — CLI/API and artifact outputs are the immediate priority.
- Enterprise orchestration/platform features (multi-tenant scheduling, hosted service) — not required for benchmark core value.

## Context

The repository is already a brownfield survival benchmarking toolkit with layered modules for API, benchmark orchestration, data handling, method adapters, evaluation, and export. Recent work indicates movement toward stronger protocol/statistical reporting and robustness tracks, which aligns with the stated objective. The desired direction is to harden this into a practitioner-grade benchmark standard similar in spirit to TabArena while keeping execution practical and time-aware.

## Constraints

- **Ecosystem**: Python-only package coverage (exclusive) — non-Python model ecosystems are out of scope.
- **Runtime Budget**: Wall-clock time is the primary operational constraint — phase plans must prioritize efficient benchmark design.
- **Benchmark Scope**: Medium balanced dataset suite — broad enough for external validity without unbounded runtime.
- **Quality Gate**: All touched-code lint/type/test checks must pass — maintain maintainability while evolving benchmark logic.
- **Storage Contract**: Results must be compact and non-redundant, with one comprehensive artifact per experiment collection.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Require both no-HPO and HPO runs for each selected model in v1 | Fair model comparison requires controlling for tuning effects | — Pending |
| Implement full pairwise + ELO ranking (TabArena-style) in v1 | Ranking layer is a core deliverable, not a nice-to-have | — Pending |
| Target practitioners as primary user | Output format and reporting should optimize model selection decisions, not only academic exploration | — Pending |
| Keep benchmark scope exclusively Python packages with medium dataset breadth | Aligns with explicit user scope and controls ecosystem complexity | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-23 after initialization*
