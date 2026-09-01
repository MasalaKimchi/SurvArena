---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Verified Benchmark Kernel
status: planning
last_updated: "2026-09-01T04:16:52.825Z"
last_activity: 2026-09-01
progress:
  total_phases: 6
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 17
---

# Project State

## Current Position

Phase: 2 (Scientific Comparison Kernel) — READY TO PLAN
Plan: Not started
Status: Ready to plan
Last activity: 2026-09-01

## Project Reference

See: `.planning/PROJECT.md` (updated 2026-08-31)

**Core value:** A practitioner can trust one benchmark run to produce fair, statistically robust, provenance-complete, and compactly stored model comparisons across a representative survival dataset suite.

**Current focus:** Phase 2 — Scientific Comparison Kernel

## Accumulated Context

### Decisions

- Statistical correctness and truthful verification precede dataset/model expansion.
- `SurvArena v2.0` is the software milestone; `survbench-1.0-rc1` is the first citable protocol target.
- Public leaderboard and submission infrastructure are deferred until the kernel is independently reproducible.

### Blockers

- Existing comparison/ranking logic is not yet safe for citable claims: matched-cell pairing, dataset-level inference, common support, failure eligibility, and independent metric references remain unverified.

### Todos

- Plan Phase 2 from constructed pairing/ranking counterexamples and independent survival-metric reference fixtures.
- Keep retained pre-fix benchmark matrices invalidated until Phases 2–6 complete.
