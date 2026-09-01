---
phase: 01-truthful-green-baseline
plan: "03"
subsystem: ci-and-verification-evidence
tags: [github-actions, clean-snapshot, semantic-smoke, documentation, mypy]
requires: [01-01, 01-02]
provides:
  - Clean-committed-snapshot development baseline with semantic WHAS500/CoxPH smoke
  - CI contracts for Ruff, bounded mypy, pytest, compileall, and semantic artifact assertions
  - Dated documentation separating local evidence, configured automation, and invalidated results
affects: [phase-2-scientific-kernel, phase-6-release-reproduction]
tech-stack:
  added: []
  patterns: [clean-archive-verification, semantic-smoke-assertions, bounded-incremental-typing]
key-files:
  created:
    - .github/workflows/ci.yml
    - .github/workflows/benchmark-smoke.yml
    - docs/test_status.md
    - docs/ci_and_reproducibility.md
  modified:
    - README.md
    - PROJECT_STATE.md
    - pyproject.toml
    - survarena/__init__.py
    - survarena/benchmark/runner.py
    - survarena/data/splitters.py
key-decisions:
  - A green dirty workspace is insufficient evidence; the committed archive must pass independently.
  - Incremental mypy uses follow_imports=skip so its named targets do not silently become a whole-package type claim.
  - Semantic smoke requires successful finite metric rows and compact artifacts, not merely a zero process exit.
  - Clean-snapshot and dirty-workspace test counts are reported separately.
requirements-completed: [BASE-01, BASE-04]
duration: 24 min
completed: 2026-09-01
---

# Phase 1 Plan 03: CI and Integrated Baseline Summary

Phase 1 now has a truthful, clean-snapshot development baseline and a semantic benchmark smoke, while publication readiness remains deliberately blocked pending the scientific, execution, protocol, pilot, and reproduction phases.

## Performance

- **Duration:** 24 min
- **Tasks:** 3 plus clean-snapshot coherence corrections
- **Committed snapshot:** 250 passed, 6 skipped
- **Current workspace:** 258 passed, 6 skipped

## Accomplishments

- Added least-privilege CI jobs for Ruff, bounded incremental mypy, Python 3.10–3.12 lazy-import smoke, full pytest, and compileall.
- Added a manual WHAS500/CoxPH workflow that checks identities, successful rows, finite Uno C, compact artifacts, and absence of a redundant leaderboard JSON.
- Verified a clean committed archive rather than relying only on the dirty workspace.
- Ran the clean semantic smoke: five successful outer folds, finite Uno C from `0.670736789703` to `0.887688636780`, and the exact compact artifact set.
- Published dated status/reproducibility documentation that distinguishes executed evidence from defined CI and invalidated historical matrices.
- Closed the Plan 01-01 adoption gap by committing the typed core and its tests.

## Task Commits

| Task | Commit | Description |
|---|---|---|
| 1 | `6099d2e` | Add bounded type/compile CI and semantic benchmark assertions |
| 2 | `8d4bcb0` | Publish truthful status and reproducibility boundaries |
| 3 | `3e3739d` | Record final clean-snapshot and workspace evidence |
| Coherence | `a712af0` | Adopt the verified typed core required by clean CI |
| Coherence | `99303a5` | Commit runner, arm, scheduler, and capability behavior exercised by tests |
| Coherence | `3f609aa` | Make top-level package import lazy and dependency-light |
| Coherence | `2416fa2` | Bound incremental type analysis to the declared targets |
| Coherence | `48637f4` | Complete split identity/group-aware implementation used by the runner |

## Verification

- Workspace Ruff — **passed**.
- Workspace declared mypy scope — **no issues in 11 source files**.
- Workspace pytest — **258 passed, 6 skipped, 22 warnings**.
- Workspace compileall — **passed**.
- Clean committed archive pytest — **250 passed, 6 skipped, 22 warnings**.
- Clean committed archive Ruff/mypy/compileall — **passed**.
- Clean committed archive semantic smoke — **5/5 rows successful; all Uno C finite; compact artifacts present**.
- Strict evidence audit — **`publishable=false`, expected exit 2, no traceback**.
- Workflow YAML, documentation contracts, and key links — **passed**.

## Deviations from Plan

### [Rule 3 - Blocking] Dirty-tree success did not imply committed CI success

- **Found during:** Task 3 clean-archive verification.
- **Issue:** Committed tests and CI referenced uncommitted core, runner, lazy-import, and split-identity behavior. The first archive check failed 10 focused tests, and the initial type check recursively reached legacy modules.
- **Resolution:** Deliberately adopted the verified core, committed the minimum coherent runtime dependencies, bounded mypy import traversal, and reran all clean-snapshot gates.
- **Impact:** The branch now supports the CI contract it declares. Remaining unrelated user changes stay uncommitted and preserved.

### [Rule 1 - Bug] Plan referenced a nonexistent benchmark config

- **Found during:** Task 3 read-first verification.
- **Issue:** The plan named `configs/benchmark/standard_v1.yaml`, which does not exist.
- **Resolution:** Corrected the plan to the live `configs/benchmark/manuscript_v1.yaml` and used that protocol for the smoke.
- **Impact:** Verification now follows the actual configured protocol.

**Total deviations:** 2 correctness/coherence deviations, both resolved and reverified.

## Remaining Boundary

- Hosted GitHub Actions has not yet been observed remotely.
- Historical benchmark matrices remain invalidated and non-citable.
- The typed protocol/result store is additive, not yet authoritative in the live runner.
- Statistical comparison correctness and killable execution isolation remain Phase 2 and Phase 3 work.

## Self-Check: PASSED

- All Plan 01-03 acceptance criteria are backed by executed commands.
- No smoke artifact was copied into repository results.
- Clean-snapshot, workspace, CI-definition, and publication-evidence claims remain explicitly separated.
