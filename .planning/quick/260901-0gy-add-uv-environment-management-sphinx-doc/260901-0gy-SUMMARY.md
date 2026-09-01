---
phase: quick-260901-0gy
plan: "01"
subsystem: developer-infrastructure
tags: [uv, sphinx, myst-parser, github-actions, ruff, mypy, pytest]

# Dependency graph
requires: []
provides:
  - Locked uv 0.12.8 developer and CI environment across Python 3.10-3.12
  - Strict MyST/Sphinx site covering every tracked Markdown guide
  - Immutable, least-privilege GitHub Actions quality and benchmark-smoke workflows
  - Executable regression tests for dependency and workflow policy
affects: [contributor-workflow, ci, documentation, release-reproducibility]

# Tech tracking
tech-stack:
  added: [uv 0.12.8, Sphinx 8.2.3, myst-parser 5.1]
  patterns: [locked sync followed by no-sync execution, dependency-profile CI caches, strict narrative documentation builds]

key-files:
  created: [.python-version, uv.lock, docs/conf.py, tests/test_developer_infrastructure.py]
  modified: [pyproject.toml, requirements.txt, README.md, docs/environment.md, docs/index.md, docs/ci_and_reproducibility.md, .github/workflows/ci.yml, .github/workflows/benchmark-smoke.yml, .gitignore, survarena/core/protocol/spec.py]

key-decisions:
  - "Treat uv.lock as the cross-platform developer/CI lock, not the separately governed canonical Linux/amd64 manuscript-release environment."
  - "Keep the Python 3.10 import and infrastructure lanes, but run the full suite on 3.11/3.12 until the overlapping user-owned timezone.utc compatibility edit in tuning.py can be committed separately."
  - "Build narrative Markdown with MyST and a plain-text Mermaid lexer, without autodoc or a Mermaid renderer."

patterns-established:
  - "Environment commands explicitly sync with --locked before all uv run --no-sync checks."
  - "GitHub Actions use immutable SHAs, disabled checkout credential persistence, read-only permissions, and stable dependency-profile cache suffixes."
  - "Sphinx builds are nitpicky and fail on every warning while importing no heavy SurvArena modules."

requirements-completed: []

# Metrics
duration: 26min
completed: 2026-09-01
---

# Quick Task 260901-0gy: uv, Sphinx, and Locked CI Summary

**A uv 0.12.8 lock-backed contributor environment, strict 18-page Sphinx guide site, and immutable GitHub Actions quality gates now share one executable local/CI contract.**

## Performance

- **Duration:** 26 min
- **Started:** 2026-09-01T05:07:56Z
- **Completed:** 2026-09-01T05:33:31Z
- **Tasks:** 3
- **Files modified:** 14

## Accomplishments

- Added a generated cross-platform `uv.lock`, preferred Python 3.11 pin, uv version requirement, PEP 735 quality/test/import/docs groups, and strict pytest configuration without changing runtime or optional-backend pins.
- Turned all 18 tracked Markdown pages into a strict MyST/Sphinx site with complete toctree coverage and a warning-safe plain-text Mermaid fallback.
- Rebuilt PR and manual benchmark workflows around locked uv syncs, immutable official action SHAs, read-only permissions, credential-safe checkout, stable cache profiles, and focused infrastructure regression tests.
- Verified the committed result from a detached zero-overlay worktree while preserving all 30 pre-existing dirty files byte-for-byte.

## Task Commits

Each implementation task was committed atomically:

1. **Task 1: Define the uv and pytest project contract** - `10be59f` (chore)
2. **Task 2: Build the existing Markdown guides as strict Sphinx documentation** - `c999e5d` (docs)
3. **Task 3: Enforce the locked local contract in GitHub Actions** - `c2be225` (chore)

Plan metadata and this summary remain uncommitted for the root GSD orchestrator, as required by the quick-task handoff.

## Files Created/Modified

- `.python-version` - Selects Python 3.11 for the preferred local environment.
- `uv.lock` - Generated uv 0.12.8 developer/CI resolution for project extras and dependency groups.
- `pyproject.toml` - Declares Python support, uv policy, dependency groups, and strict pytest defaults.
- `requirements.txt` - Retains the pip compatibility path while directing contributors to uv first.
- `docs/conf.py` - Configures strict MyST/Sphinx with the plain-text Mermaid lexer.
- `docs/index.md` - Routes every tracked guide through four explicit toctrees.
- `README.md` - Documents uv-first setup, locked checks, and compatibility paths.
- `docs/environment.md` - Documents exact sync/run, optional-extra, lock-update, and Sphinx commands.
- `docs/ci_and_reproducibility.md` - Defines the local/CI equivalence and release-environment boundary.
- `.github/workflows/ci.yml` - Enforces locked quality, type, import, full-test, compile, and docs gates.
- `.github/workflows/benchmark-smoke.yml` - Uses the same pinned uv/action and cache-security contract for manual smoke runs.
- `tests/test_developer_infrastructure.py` - Semantically locks dependency metadata and workflow security/version policy.
- `.gitignore` - Excludes generated Sphinx output.
- `survarena/core/protocol/spec.py` - Uses factories for immutable empty mapping defaults on Python 3.11+.

## Decisions Made

- The committed lock is deliberately described as the reproducible developer/CI resolution. Phase 6 still owns a canonical Linux/amd64 release environment for citable benchmark evidence.
- Cheap jobs install isolated dependency groups; only full tests and manual benchmark smoke install the runtime graph.
- Sphinx renders maintained narrative sources only, avoiding autodoc imports of the heavy scientific stack.
- Because the clean committed tree uses `datetime.UTC` in `survarena/benchmark/tuning.py`, full Python 3.10 collection cannot start. The working tree already contains an overlapping user-owned `timezone.utc` fix plus unrelated HPO changes, so those bytes were preserved and not staged. The non-red full-test matrix is 3.11/3.12 pending a separate owner-controlled commit of that compatibility hunk.

## Verification

The final gate ran from a detached validation worktree at `c2be225` with no dirty-file overlays:

- uv 0.12.8 version and `uv lock --check`: passed.
- `ruff check --output-format=github survarena tests scripts`: passed.
- Scoped mypy over `survarena/core`, resume/split modules, and the manuscript audit script: passed with 11 source files checked.
- Lazy `import survarena`: passed on Python 3.10, 3.11, and 3.12.
- Infrastructure regression suite: 4 passed on Python 3.10, 3.11, and 3.12 across candidate/final validation.
- Full pytest on Python 3.11: 290 passed, 6 skipped.
- Full pytest on Python 3.12: 290 passed, 6 skipped.
- `compileall` on both full-test lanes: passed.
- Strict `sphinx-build -n -W --keep-going`: passed for all 18 pages.
- Pre-existing dirty-file SHA-256/status manifest: exact byte-for-byte match after validation cleanup.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Made immutable protocol defaults valid dataclass defaults**

- **Found during:** Task 1 (Define the uv and pytest project contract)
- **Issue:** Python 3.11 rejected seven `MappingProxyType({})` dataclass values as mutable defaults, so clean test collection failed before any infrastructure test could run.
- **Fix:** Added one `_empty_map` factory and changed only those fields to `field(default_factory=_empty_map)`, preserving immutable-empty-map behavior.
- **Files modified:** `survarena/core/protocol/spec.py`
- **Verification:** Focused core-protocol tests passed (8 tests), clean collection completed, and both final full-test lanes passed.
- **Committed in:** `10be59f` (part of Task 1)

**2. [Rule 3 - Blocking] Removed links embedded directly in Sphinx headings**

- **Found during:** Task 2 (Build the existing Markdown guides as strict Sphinx documentation)
- **Issue:** Sphinx 8.2 raised `KeyError: anchorname` for existing Markdown headings containing inline links.
- **Fix:** Kept the headings as plain text and moved their destination links into adjacent paragraphs; Task 3 then completed the planned CI/reproducibility rewrite of that guide.
- **Files modified:** `docs/ci_and_reproducibility.md`
- **Verification:** The strict build rendered all 18 pages with no warnings in both candidate and detached final validation.
- **Committed in:** `c999e5d` (part of Task 2)

### Constrained Plan Adjustment

**3. [Rule 3 - Blocking] Substituted Python 3.11 for the full Python 3.10 test lane**

- **Found during:** Task 3 (Enforce the locked local contract in GitHub Actions)
- **Issue:** Clean Python 3.10 collection fails because committed `survarena/benchmark/tuning.py` imports `datetime.UTC`, introduced in Python 3.11. The pre-existing dirty version already changes the exact lines to `timezone.utc`, alongside unrelated user HPO work.
- **Resolution:** Per the dirty-worktree guardrail, neither the overlapping line nor the rest of `tuning.py` was modified or staged. CI and its regression/documentation contract use full tests on 3.11/3.12; the minimum version remains covered by locked import and infrastructure-policy lanes.
- **Files modified:** `.github/workflows/ci.yml`, `tests/test_developer_infrastructure.py`, `docs/ci_and_reproducibility.md`
- **Verification:** Clean final lanes passed as listed above, and the original `tuning.py` hash remained identical to the baseline manifest.
- **Committed in:** `c2be225` (part of Task 3)

---

**Total deviations:** 2 auto-fixed blockers and 1 constrained matrix adjustment.
**Impact on plan:** The uv, documentation, security, and supported-version import contracts are complete. The original minimum-version full-suite criterion remains pending the owner-controlled Python 3.10 compatibility commit described below; no unrelated user edits were absorbed.

## Issues Encountered

- The sandbox initially denied registry/cache network access, including one Python 3.12 NWTCO dataset fetch. The same exact commands were rerun with approved network access and passed; no dependency names or versions were substituted.
- uv correctly warned when matrix environments differed from the repository's preferred `.python-version`; each command used an explicit requested interpreter and isolated `UV_PROJECT_ENVIRONMENT`, so this did not change resolution or results.

## Known Stubs

None. A scan found only the pre-existing `.gitignore` comment referring to placeholder data directories; no empty/mock implementation or unfinished UI path was introduced.

## User Setup Required

None - no secrets, deployment, or external service configuration was added.

## Remaining Issue

- Full-suite Python 3.10 CI should replace the temporary 3.11 lane after the owner safely commits the already-present `timezone.utc` compatibility hunk in `survarena/benchmark/tuning.py`. Until then, Python 3.10 is proven only by locked import smoke and infrastructure tests, not the full 290-test suite.

## Next Phase Readiness

- Contributors and PR CI can now reproduce the same locked Ruff, mypy, pytest, compile, and Sphinx commands with uv 0.12.8.
- Phase 6 can build its separately governed canonical Linux/amd64 release environment without conflating it with the cross-platform developer lock.
- Root review is needed only for the documented Python 3.10 full-suite variance; all other planned gates passed cleanly.

## Self-Check: PASSED

- All key created artifacts and the summary exist.
- Task commits `10be59f`, `c999e5d`, and `c2be225` are present in repository history.
- The implementation commit range exactly matches the 13 planned files plus the authorized `survarena/core/protocol/spec.py` blocking fix.

---
*Phase: quick-260901-0gy*
*Completed: 2026-09-01*
