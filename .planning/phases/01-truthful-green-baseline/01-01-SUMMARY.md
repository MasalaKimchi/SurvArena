---
phase: 01-truthful-green-baseline
plan: "01"
subsystem: verification-baseline
tags: [pytest, mypy, artifacts, resume, multiprocessing, parquet]
requires: []
provides:
  - Deterministic artifact, resume-arm, process-scheduler, and Parquet verification contracts
  - Incremental static type gate for the typed core and Phase 1 kernel modules
affects: [01-02, 01-03, phase-2-scientific-kernel]
tech-stack:
  added: [mypy-2.3.1]
  patterns: [structured-artifact-failure, arm-qualified-resume, process-boundary-test, forced-capability-branch]
key-files:
  created: []
  modified:
    - pyproject.toml
    - survarena/benchmark/resume.py
    - survarena/core/results/schema.py
    - survarena/core/results/store.py
    - tests/test_benchmark.py
    - tests/test_core_results.py
key-decisions:
  - Model serialization failure remains non-fatal to an otherwise successful scientific run; predictions and an error-bearing manifest are retained.
  - Legacy resume rows without hpo_mode satisfy no_hpo only; explicit arm rows suppress only their exact arm.
  - ProcessPoolExecutor remains the production parallel boundary because run units mutate process-global RNG state.
  - Static typing starts with an explicit 11-file/module kernel scope and expands in later phases.
requirements-completed: [BASE-01]
duration: 8 min
completed: 2026-09-01
---

# Phase 1 Plan 01: Verification Contracts and Type Gate Summary

Four environment- and implementation-drift failures were converted into deterministic behavioral checks, with a pinned incremental mypy gate over the typed core and Phase 1 kernel modules.

## Performance

- **Duration:** 8 min
- **Started:** 2026-09-01T03:30:00Z
- **Completed:** 2026-09-01T03:37:59Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Artifact tests now prove arm-qualified storage, retained prediction evidence, structured serialization failure, and run-level scientific success.
- Resume tests prove both legacy and explicit arm behavior; HPO work cannot be skipped by an arm-less historical row.
- Scheduler tests observe the real process executor boundary without relying on shared child/parent memory.
- Parquet tests execute both an installed round-trip and a forced unavailable dependency branch.
- Mypy 2.3.1 passes over `survarena/core`, resume, split compatibility, and the release audit.

## Task Commits

| Task | Commit | Description |
|---|---|---|
| 1 | `fcfbf26` | Align artifact and resume regression contracts |
| 2 | `1acc76f` | Verify the process scheduler at the production boundary |
| 3 | `68cabed` | Pin/configure mypy and complete legacy resume identity behavior |

## Verification

- `pytest -q tests/test_benchmark.py tests/test_core_results.py` — **59 passed**.
- Declared mypy scope — **Success: no issues found in 11 source files**.
- Targeted Ruff scope — **All checks passed**.
- Standalone `tests/test_core_results.py` harness — **15/15 passed**.

## Deviations from Plan

### [Rule 3 - Blocking] Preserve pre-existing untracked typed-core work

- **Found during:** Task 3
- **Issue:** `survarena/core/` and `tests/test_core_results.py` existed as untracked user work before Phase 1, so they have no repository baseline from which this plan's small typing/Parquet edits can be staged independently.
- **Resolution:** Preserved those files in place, verified them in the working tree, and did not absorb the user's broader untracked implementation into an unrelated atomic task commit. The tracked mypy configuration and resume fix were committed normally.
- **Impact:** The current workspace is green, but a future integration commit must deliberately adopt the typed-core files before a clean clone can run that scope. Phase 4 remains the authoritative integration boundary.

**Total deviations:** 1 scope-preservation deviation. **Impact:** No runtime behavior or verification was weakened; clean-checkout adoption remains explicit.

## Issues Encountered

- The fixture's small Uno-C example returns `NaN`; equality verification was corrected to use NumPy's NaN-aware assertion without changing the metric logic.
- PyArrow emits sandboxed macOS CPU-cache discovery warnings in the standalone harness; export and read-back both succeed.

## Next Plan Readiness

Ready for Plan 01-02: the release audit and split-cache compatibility diagnostics can now build on a green focused suite and declared type gate.

## Self-Check: PASSED

- All task acceptance criteria pass in the current workspace.
- All three task commits exist.
- No broad skips, thread fallback, silent cache changes, or blanket mypy suppressions were introduced.
