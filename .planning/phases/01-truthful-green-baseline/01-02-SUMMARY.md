---
phase: 01-truthful-green-baseline
plan: "02"
subsystem: audit-and-split-compatibility
tags: [release-audit, markdown, split-cache, provenance, fail-closed]
requires: [01-01]
provides:
  - Dependency-free strict publication audit with a fail-closed evidence verdict
  - Deterministic field-level split-manifest mismatch diagnostics and explicit recovery
affects: [01-03, phase-6-release-reproduction]
tech-stack:
  added: []
  patterns: [dependency-light-markdown, bounded-compatibility-diff, explicit-cache-regeneration]
key-files:
  created: []
  modified:
    - scripts/audit_manuscript_publishability.py
    - survarena/data/splitters.py
    - tests/test_manuscript_report.py
    - tests/test_benchmark.py
key-decisions:
  - The audit uses an internal scalar Markdown renderer rather than pandas' optional tabulate dependency.
  - Publication readiness is fail-closed while retained benchmark matrices predate behavior-changing fixes.
  - Split mismatch diagnostics identify missing, unexpected, and changed fields but never mutate cache state without --regenerate-splits.
requirements-completed: [BASE-02, BASE-03]
duration: 6 min
completed: 2026-09-01
---

# Phase 1 Plan 02: Release Audit and Split Compatibility Summary

The strict release audit now reaches a truthful verdict without optional presentation dependencies, and incompatible split caches fail closed with a bounded field-level diff and exact recovery path.

## Performance

- **Duration:** 6 min
- **Started:** 2026-09-01T03:39:10Z
- **Completed:** 2026-09-01T03:45:23Z
- **Tasks:** 2 plus one correctness deviation
- **Files modified:** 4

## Accomplishments

- Replaced `DataFrame.to_markdown` with deterministic internal pipe-table rendering covering pipes, backslashes, newlines, missing values, and empty frames.
- Proved `--strict` writes a report, emits `publishable=false`, exits 2 for blockers, and produces no traceback.
- Fixed a fail-open verdict where complete coverage could have returned publishable despite explicit environment/evidence blockers.
- Added stable split-cache diagnostics containing the manifest path, exact changed/missing fields, and literal `--regenerate-splits` guidance.
- Proved rejection leaves legacy cache bytes untouched and explicit regeneration creates a reusable exact-match manifest.

## Task Commits

| Task | Commit | Description |
|---|---|---|
| 1 | `84f11d5` | Make the release audit independent of optional tabulate rendering |
| 2 | `7a838f9` | Add actionable, fail-closed split-manifest compatibility diagnostics |
| Deviation | `309a5c4` | Prevent invalidated retained evidence from ever passing the strict verdict |

## Verification

- Focused audit/cache suite — **7 passed**.
- Targeted Ruff scope — **All checks passed**.
- Audit/splitter mypy scope — **Success: no issues found in 2 source files**.
- Strict audit — report written, `publishable=false`, expected exit code **2**, no rendering/import traceback.

## Deviations from Plan

### [Rule 1 - Bug] Strict audit could pass while reporting blockers

- **Found during:** Plan-level verification
- **Issue:** `is_publishable` checked three coverage summaries but ignored the audit's own lockfile, regenerated-evidence, and final-artifact blockers.
- **Fix:** Added an explicit fail-closed current-evidence validity gate, invalidation blocker, and structural/historical wording for retained matrices.
- **Files modified:** `scripts/audit_manuscript_publishability.py`, `tests/test_manuscript_report.py`
- **Verification:** Strict CLI now deterministically returns 2 and report tests assert invalidation/reproduction guidance.
- **Commit:** `309a5c4`

**Total deviations:** 1 auto-fixed correctness bug. **Impact:** The audit can no longer produce a false publication-ready result.

## Issues Encountered

None after the verdict correction. An exit code of 2 is the expected scientific blocker result, not an execution failure.

## Next Plan Readiness

Ready for Plan 01-03: CI and maintained documentation can now point to executable checks and a truthful strict-audit outcome.

## Self-Check: PASSED

- Renderer has no `to_markdown` or `tabulate` dependency.
- Default split mismatch paths are read-only and explicit regeneration is verified.
- BASE-02 and BASE-03 behaviors are automated and fail closed.
