---
phase: 01-truthful-green-baseline
verified: 2026-09-01T04:12:42Z
status: passed
score: 12/12 must-haves verified
overrides_applied: 0
deferred:
  - truth: Killable process-tree budgets, group-safe inner validation, exact split/group fingerprints, and adapter conformance are not complete.
    addressed_in: Phase 3
    evidence: Phase 3 success criteria explicitly require process-tree termination, group-disjoint outer/inner/early-stopping splits, material split fingerprints, and adapter capability conformance.
  - truth: Strict nested protocol scalars and conflict-rejecting canonical result storage are not authoritative.
    addressed_in: Phase 4
    evidence: Phase 4 success criteria require ambiguous configs to fail before work and provenance conflicts to be rejected without overwrite.
  - truth: CI actions and the release environment are not digest/lock pinned or independently reproduced.
    addressed_in: Phase 6
    evidence: Phase 6 success criteria require hashed locks, digest-pinned Linux/amd64 execution, clean wheel/container gates, and independent reproduction.
---

# Phase 1: Truthful Green Baseline Verification Report

**Phase Goal:** Establish a trustworthy baseline where every documented verification command runs and accurately reports current evidence status.

**Verified:** 2026-09-01T04:12:42Z

**Status:** passed

**Re-verification:** No — initial goal-backward verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---:|---|---|---|
| 1 | Declared lint, bounded static-type, full test, compile, and core smoke commands run without failures. | ✓ VERIFIED | Clean `HEAD` archive: Ruff passed, mypy passed 11 targets, pytest 250 passed/6 skipped, compileall passed; current workspace: 258 passed/6 skipped. |
| 2 | The known artifact/resume/process/Parquet failures are deterministic assertions rather than skips or weakened checks. | ✓ VERIFIED | Focused behavior tests pass; full suites retain only six explicit optional foundation/integration skips. |
| 3 | Model serialization failure preserves prediction evidence and scientific success with structured artifact failure. | ✓ VERIFIED | `_save_model_artifacts` writes predictions before serialization, catches serialization failure, writes `model_artifact_status=failed`; regression tests pass. |
| 4 | Legacy resume evidence without an arm completes `no_hpo` only. | ✓ VERIFIED | `completed_resume_keys` maps arm-less legacy rows only to `no_hpo`; explicit arm tests cover both directions. |
| 5 | Production parallelism remains process-isolated; Parquet available/unavailable behavior and the intentionally bounded mypy scope are verified. | ✓ VERIFIED | `_execute_run_units` uses `ProcessPoolExecutor`; installed and forced-missing PyArrow branches pass; `follow_imports=skip` prevents a false whole-package type claim. |
| 6 | The strict release audit reaches an actionable scientific verdict using declared dependencies. | ✓ VERIFIED | `--strict` writes the report, prints `publishable=false`, exits 2 for named blockers, and produces no traceback. |
| 7 | Audit Markdown rendering is deterministic without optional `tabulate`. | ✓ VERIFIED | Exact fixtures cover pipes, backslashes, newlines, missing values, and empty frames; no `to_markdown` dependency remains. |
| 8 | Incompatible split caches fail closed with precise compatibility details. | ✓ VERIFIED | Error includes manifest path plus bounded missing/unexpected/changed fields and literal `--regenerate-splits` guidance. |
| 9 | Split regeneration is deliberate and reusable; legacy cache bytes are not silently changed. | ✓ VERIFIED | Regression tests verify rejection is non-mutating, explicit regeneration writes a new manifest, and the next exact load reuses it. |
| 10 | Local docs and CI declare the same lint/type/test/compile boundaries without claiming hosted execution. | ✓ VERIFIED | README, `docs/test_status.md`, `docs/ci_and_reproducibility.md`, and workflow commands/scopes match; hosted CI remains labeled unobserved. |
| 11 | A clean committed WHAS500/CoxPH one-seed smoke produces real successful finite results and compact outputs. | ✓ VERIFIED | Verifier rerun produced 5/5 successful outer-fold rows, finite Uno C, required identity columns, six compact artifacts, and no redundant JSON leaderboard. |
| 12 | Maintained status pages distinguish verified code behavior, configured automation, invalid historical evidence, and remaining release blockers. | ✓ VERIFIED | `PROJECT_STATE.md` and `docs/test_status.md` explicitly invalidate retained matrices and point to Phases 2–6; the audit is fail-closed. |

**Score:** 12/12 truths verified

### Deferred Items

These are genuine repository gaps found by the standard-depth review, but they are outside Phase 1's baseline goal and explicitly contracted by later milestone phases.

| # | Item | Addressed In | Evidence |
|---:|---|---|---|
| 1 | Killable timeout/process trees, group-safe inner validation, exact split/group fingerprints, adapter conformance | Phase 3 | Execution and Data Safety success criteria 1–5 |
| 2 | Strict nested protocol types and conflict-rejecting canonical result storage | Phase 4 | Authoritative Protocol and Results success criteria 1–4 |
| 3 | Digest-pinned CI/actions/environment and independent reproduction | Phase 6 | Release Reproduction success criteria 1–5 |

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `pyproject.toml` | Pinned mypy and bounded incremental config | ✓ VERIFIED | Mypy 2.3.1 declared; explicit Python target/config; target command passes. |
| `tests/test_benchmark.py` | Artifact, resume, scheduler, cache regressions | ✓ VERIFIED | Substantive tests execute through runner behavior; focused and full suites pass. |
| `tests/test_core_results.py` | Deterministic Parquet branches | ✓ VERIFIED | Installed round-trip and forced import failure both covered. |
| `scripts/audit_manuscript_publishability.py` | Self-contained strict audit | ✓ VERIFIED | Internal renderer and fail-closed validity gate are live through CLI. |
| `survarena/data/splitters.py` | Compatibility diff and explicit regeneration | ✓ VERIFIED | Live runner passes content/group inputs; default mismatch path is non-mutating. |
| `tests/test_manuscript_report.py` | Exact renderer/CLI coverage | ✓ VERIFIED | Imports real script and exercises strict subprocess. |
| `.github/workflows/ci.yml` | Lint/type/import/test/compile automation | ✓ VERIFIED | YAML parses and commands match documentation. |
| `.github/workflows/benchmark-smoke.yml` | Semantic smoke automation | ✓ VERIFIED | Asserts successful finite rows, identities, compact artifacts, and nonredundancy. |
| `docs/test_status.md` | Dated observed evidence | ✓ VERIFIED | Separates clean snapshot, current workspace, configured CI, and citable evidence. |
| `PROJECT_STATE.md` | Current milestone/evidence boundary | ✓ VERIFIED | Links v2 roadmap, invalidates historical matrices, and names remaining phases. |

### Key Link Verification

The generic path-string checker could not infer five Python import/call links, so they were verified manually.

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `tests/test_benchmark.py` | `survarena/benchmark/runner.py` | Artifact and executor contracts | ✓ WIRED | Imports `runner`; invokes `_save_model_artifacts`, `evaluate_split`, and real scheduler boundary. |
| `tests/test_benchmark.py` | `survarena/benchmark/resume.py` | Five-field completion identity | ✓ WIRED | Runner imports `completed_resume_keys`; tests drive legacy and explicit modes through `run_benchmark`. |
| `tests/test_core_results.py` | `survarena/core/results/store.py` | Lazy PyArrow boundary | ✓ WIRED | Imports `ResultStore`; invokes `export_parquet` with installed and forced-missing modules. |
| `scripts/audit_manuscript_publishability.py` | `tests/test_manuscript_report.py` | Exact rendering fixtures | ✓ WIRED | Test loads the real script path and invokes `_markdown_table` plus strict CLI. |
| `survarena/data/splitters.py` | `survarena/run_benchmark.py` | Explicit regeneration flag | ✓ WIRED | Error names `--regenerate-splits`; CLI parses and forwards `regenerate_splits`. |
| `README.md` | `.github/workflows/ci.yml` | Identical commands | ✓ WIRED | Direct link and exact lint/type/test/compile commands present. |
| `docs/ci_and_reproducibility.md` | `.github/workflows/benchmark-smoke.yml` | Semantic assertions | ✓ WIRED | Direct link and matching WHAS500/CoxPH assertion contract. |
| `PROJECT_STATE.md` | `.planning/ROADMAP.md` | v2 target/blockers | ✓ WIRED | Direct roadmap link and matching phase sequence. |

### Data-Flow Trace (Level 4)

Not applicable: Phase 1 artifacts are CLI/config/test/documentation contracts, not dynamic UI views. The benchmark smoke itself traces real dataset → split → fit → predictions → metrics → compact artifact output.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| Current workspace development gates | Ruff + declared mypy + full pytest + compileall | 258 passed, 6 skipped; all non-pytest gates passed | ✓ PASS |
| Clean committed snapshot development gates | `git archive HEAD`, then identical commands | 250 passed, 6 skipped; all non-pytest gates passed | ✓ PASS |
| Arm-qualified identity correction | `pytest tests/test_phase1_identity.py` | 2 passed | ✓ PASS |
| Strict audit behavior | `python scripts/audit_manuscript_publishability.py --strict` | `publishable=false`, expected exit 2, no traceback | ✓ PASS |
| Clean semantic smoke | WHAS500/CoxPH, seed 11, five folds | 5 successful finite rows; compact artifact set exact | ✓ PASS |

### Probe Execution

No phase-declared or conventional `probe-*.sh` files exist. The explicit semantic benchmark smoke is the phase's runnable end-to-end probe and passed independently above.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|---|---|---|---|---|
| BASE-01 | 01-01, 01-03 | Supported development checks have zero failures | ✓ SATISFIED | Workspace and clean snapshot gates plus semantic smoke pass. |
| BASE-02 | 01-02 | Strict audit returns actionable pass/fail using declared dependencies | ✓ SATISFIED | Expected blocker exit 2 without import/rendering failure. |
| BASE-03 | 01-02 | Legacy split cache has explicit safe compatibility/migration path | ✓ SATISFIED | Field-level mismatch, manifest path, recovery flag, non-mutation, and reusable regeneration verified. |
| BASE-04 | 01-03 | Docs distinguish verified, invalidated, and remaining evidence | ✓ SATISFIED | Dated state/test/CI docs and fail-closed audit agree. |

No Phase 1 requirement is orphaned; every mapped requirement appears in plan frontmatter.

### Anti-Patterns Found

No unreferenced `TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, placeholder, or stub marker was found in the 28-file Phase 1 review scope. The seven standard-depth review findings are recorded in `01-REVIEW.md`; the six open warnings are explicitly deferred above and do not contradict a Phase 1 must-have.

### Human Verification Required

None. Phase 1 behaviors are machine-verifiable, and no plan contains a deferred `<human-check>`.

### Gaps Summary

No Phase 1 goal gaps remain. The overall repository is intentionally not manuscript-ready: scientific comparison correctness, execution/data safety, authoritative protocol/storage, representative pilot design, and frozen reproduction remain Phases 2–6.

---

_Verified: 2026-09-01T04:12:42Z_
_Verifier: Codex (inline gsd-verifier fallback)_
