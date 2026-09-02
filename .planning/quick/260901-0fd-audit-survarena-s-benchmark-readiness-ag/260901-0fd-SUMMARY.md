---
quick_id: 260901-0fd
phase: quick-260901-0fd
plan: "01"
subsystem: benchmark-documentation
tags: [benchmark-readiness, scientific-validity, reproducibility, provenance, governance]
dependency_graph:
  requires: []
  provides: [durable-readiness-verdict, prioritized-improvement-register, research-coverage-crosswalk]
  affects: [benchmark-protocol, release-evidence, manuscript-readiness]
tech_stack:
  added: [standard-library-validator, pytest-contract-tests]
  patterns: [stable-audit-ids, observed-closure-gates, conflict-rejecting-provenance, secret-first-validation, baseline-bound-git-scope, digest-only-path-output]
key_files:
  created: [docs/benchmark_readiness.md, scripts/validate_benchmark_readiness.py, tests/test_benchmark_readiness_validator.py]
  modified: [docs/index.md]
decisions:
  - "Treat SurvArena as an internal benchmark framework and pre-release kernel, not a research/citable release."
  - "Prioritize Phase 3 execution and data safety before Phase 4 release provenance."
  - "Treat BR-009 as the highest subsequent release-provenance target before citable full-matrix compute, after its execution, dataset, and protocol prerequisites."
  - "Keep public-platform and broadened-task work deferred from the current single-event Python package gates."
metrics:
  duration: 64m
  completed: 2026-09-01
---

# Quick 260901-0fd: SurvArena Benchmark Readiness Audit Summary

SurvArena now has a durable, evidence-ranked benchmark-readiness register with a candid publication verdict, stable improvement IDs, falsifiable closure gates, and one documentation-index route.

## Tasks Completed

| Task | Description | Commit | Files |
|---|---|---|---|
| 1 | Establish maturity verdict, wrong-today ledger, and full domain audit | `f7ea000` | `docs/benchmark_readiness.md` |
| 2 | Add owned prioritized register, exact research crosswalk, maintenance semantics, and index route | `43457a0` | `docs/benchmark_readiness.md`, `docs/index.md` |
| Repair | Correct priority/status semantics and retain the secret-first, baseline-bound validator with adversarial tests | `72dedf3` | `docs/benchmark_readiness.md`, `scripts/validate_benchmark_readiness.py`, `tests/test_benchmark_readiness_validator.py` |
| Final validator repair | Redact all path metadata and state the Git-ignored boundary exactly | `fd5d890` | `scripts/validate_benchmark_readiness.py`, `tests/test_benchmark_readiness_validator.py` |
| Symlink-loop retry | Fail closed on path-resolution loops without exposing credential-shaped path or exception text | `1759561` | `scripts/validate_benchmark_readiness.py`, `tests/test_benchmark_readiness_validator.py` |
| Exception/CLI repair | Clear retained raw exception context and redact every malformed CLI diagnostic | `25c0e97` | `scripts/validate_benchmark_readiness.py`, `tests/test_benchmark_readiness_validator.py` |

## Outcome

- Defined the exact three maturity tiers: Internal benchmark framework, Research/citable benchmark, and Public/maintained benchmark.
- Recorded exactly nine `WRONG-*` findings and 13 `DOM-*` audit domains with live repository evidence and primary or official criteria.
- Distinguished current defects, missing release evidence, and deliberate future scope. Live re-inspection showed that exact matching, dataset-scale ranks, dataset-level inference, and dataset-shared horizon behavior are fixed; retained pre-fix matrices remain invalid and must be regenerated.
- Added 19 owned `BR-*` rows, 22 exact coverage-crosswalk rows, explicit dependencies, target gates, exit evidence, and maintenance rules. Three rows are truthfully `in_progress` because their live kernels have observed fixes but their release-evidence clauses remain open; one is `blocked`, four are `deferred_scope`, 11 are `open`, and none is yet `verified`.
- Added one and only one `docs/index.md` Reference Docs route.
- Retained a standard-library validator and focused tests covering exact-buffer secret-first ordering, safe URL/redirect and path diagnostics, bounded references, requirement reconciliation, dependency/status semantics, citation/link integrity, and baseline-bound Git scope.

## Publication-Readiness Verdict

The evidence is **not comprehensive or robust enough for manuscript-grade performance claims**. SurvArena is a credible internal benchmark framework and increasingly strong pre-release kernel, but it is not yet a citable benchmark release. Retained matrices predate behavior-changing fixes; HPO semantics, group safety, killable isolation, immutable dataset identity, canonical provenance, independent reproduction, and regenerated release evidence remain incomplete.

**Immediate highest priority:** execute **Phase 3: Execution and Data Safety**, especially `BR-007` killable/resource-isolated execution and `BR-004` group-safe validation, together with the Phase 3 budget/capability portions of `BR-006`. A canonical artifact cannot make leaking, non-killable, or resource-contaminated execution scientifically valid.

**BR-009's role:** `BR-009` is the highest subsequent **release-provenance** target before citable full-matrix compute. It belongs to Phase 4 and follows the relevant Phase 3 prerequisites plus `BR-008` immutable dataset identity and `BR-010` strict protocol identity.

## Verification Results

| Check | Result |
|---|---|
| Wrong-today and domain schemas | PASS — exact `WRONG-01..09` and `DOM-01..13` sets, required evidence classes, and nonempty cells |
| Register and crosswalk schemas | PASS — 19 unique canonical rows, 22 exact research items, all rows reachable, and one index route |
| Dependency and reference integrity | PASS — bounded whole-cell WRONG/DOM/BR/crosswalk/dependency grammar, no malformed or unknown references, no self-dependencies, and acyclic dependency and replacement graphs |
| Status semantics | PASS — `BR-001`, `BR-002`, and `BR-005` remain `in_progress` until release closure; `verified`, `blocked`, `superseded`, and deferred-scope invariants are enforced; deferred P2/broadened rows cannot be current blockers |
| Provenance and local links | PASS — 93 exact tracked in-root `VERIFIED` tags, 20 cited files, zero dirty cited files, and 25 resolving local links or fragments |
| Focused scientific tests | PASS — 35 tests in `test_scientific_comparison.py` and `test_metric_contracts.py`; 14 pre-existing dependency deprecation warnings |
| External criteria | PASS — 13 unique sources checked with redirect-following GET and manual semantic identity review; ACM returned `403` and is recorded `manual_required` with confirmed official identity |
| Credential scan | PASS — the retained validator scans all four input documents before Markdown/URL processing; named patterns cover AWS access-key ID/assignment, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, GitHub, OpenAI, Hugging Face, private keys, bearer tokens, and generic assignments |
| URL and path diagnostic safety | PASS — adversarial tests prove URL userinfo/query/fragment and credential-shaped filenames are absent from diagnostics, exceptions, captured output, and baseline JSON; failures use stable digests and sanitized labels |
| Durable validator tests | PASS — 91 focused tests include nested percent-encoded credentials, secret-before-parse/network ordering, safe redirect destinations, every configured credential family plus encrypted-PEM/GitLab/Slack-shaped dirty, untracked, and committed filenames, read/stat/decode and self-referential-symlink failures, malformed CLI subprocesses, malformed references, requirement disagreement, graph/status rules, scope bypasses, and the explicit ignored-file exclusion |
| Whitespace and diff | PASS — terminal newlines, no trailing whitespace, and `git diff --check` |
| Scoped commits | PASS — Task 1 changed one readiness document; Task 2 staged and committed only the two declared docs; neither commit deleted tracked files |

## Worktree Guard Behavioral Proof

The original task baseline was captured before its document edits with zero dirty paths, but its temporary checker did not inspect intervening clean commits. The repair's retained guard closes that bypass. Because the repair baseline was captured after the three owned repair outputs and two owned metadata files were already in progress, the final command explicitly acknowledged only those exact baseline-dirty paths with `--accept-dirty-allow-path`; no sibling path was allowed or acknowledged. Disposable Git repositories exercised each adversarial case:

| Case | Expected behavior | Result |
|---|---|---|
| Changed pre-existing dirty content | Reject | PASS |
| Index-only change | Reject | PASS |
| New ordinary Git-untracked path | Reject | PASS |
| Removed ordinary Git-untracked path | Reject | PASS |
| Rename with both paths preserved | Reject | PASS |
| Unauthorized tracked path | Reject | PASS |
| Exact allowlisted change | Accept | PASS |
| Changed pre-existing dirty allowlisted path without explicit acceptance | Reject | PASS |
| Explicitly accepted owned dirty allowlisted path | Accept | PASS |
| Untracked file-mode-only change | Reject | PASS |
| Unauthorized clean commit after baseline | Reject | PASS |
| Unauthorized commit followed by byte-reverting commit | Reject | PASS |
| Credential-shaped dirty, untracked, or committed path | Reject without emitting the raw path | PASS |
| Git-ignored cache path | Exclude by stated contract and report `git_ignored_paths_checked=0` | PASS |

Diagnostics contain only sanitized path labels, short path digests, and before/after state labels. Baseline JSON uses full deterministic path digests as keys and never serializes raw credential-shaped filenames. The retained guard binds the baseline commit to its tree, verifies ancestry/repository identity, walks every intervening commit, and separately compares the baseline and current staged, unstaged, and ordinary Git-untracked identities.

The guard is intentionally a **Git repository-change attribution guard**, not a whole-filesystem monitor. Git-ignored cache, virtual-environment, and generated-environment files are outside its scope unless a future caller opts into a separate ignored-path policy. This limitation is emitted as `git_ignored_paths_checked=0`, recorded in each baseline contract, and covered by an intentional-exclusion test; no claim is made that ignored filesystem contents are monitored.

## Repair Commit and Durable Gate

**Scoped repair commits:** `72dedf3` contains exactly `docs/benchmark_readiness.md`, `scripts/validate_benchmark_readiness.py`, and `tests/test_benchmark_readiness_validator.py`. Hardening commits `fd5d890`, `1759561`, and `25c0e97` each contain exactly the validator and its focused test file. None contains a deletion, `0gy` path, or planning artifact.

The retained commands in `260901-0fd-VALIDATION.md` supersede the flawed inline PLAN checks without rewriting PLAN history. Observed repair results:

- Authenticated current-format baseline-bound offline check before and immediately after `25c0e97`: PASS with 9 WRONG rows, 13 DOM rows, 19 BR rows, 22 crosswalk rows, 93 tracked provenance tags, 25 local links, `git_scope_checked=1`, and the explicit `git_ignored_paths_checked=0` boundary.
- Disposable Git scope self-test: 14/14 adversarial cases PASS, including unauthorized clean commits, commit-then-revert detection, credential-shaped path redaction, and intentional ignored-path exclusion.
- Focused validator suite: 91/91 passed. Validator plus scientific/metric suite: 126 passed; 14 pre-existing dependency deprecation warnings.
- Self-referential symlink proof: 2/2 HF- and GitLab-token-shaped loop cases fail closed with a digest-only `ValidationError`; the raw filename and path are absent from the exception and captured stdout/stderr.
- Exception-object proof: credential-shaped symlink, read, stat, and decode failures have safe `str`, safe `repr`, safe formatted tracebacks, `__context__ is None`, and `__cause__ is None`; no original exception is retained by the sanitized error.
- Malformed CLI proof: six subprocess cases—extra positional argument, invalid subcommand, invalid `--external` choice, unknown option, self-looping baseline/path, and URL userinfo/query/fragment—return nonzero without emitting any raw argument fragment or traceback.
- Ruff, Python compilation, staged diff check, and post-commit deletion check: PASS.
- External criteria: 13/13 sources accepted as automated success or recorded manual exception after the credential gate; the ACM 403 remains the sole `manual_required` row and its official identity was manually confirmed.
- Credential fixtures: AWS access-key IDs and assignments, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, GitHub, OpenAI, Hugging Face, bearer, private-key, generic token/signature/credential assignments, and nested percent encoding are detected without retaining values. Digest-only path output additionally redacts encrypted-PEM, GitLab, and Slack token-shaped filenames even though those are not scanner patterns.
- The prior read-only review did not exercise a self-referential symlink. Subsequent verification reproduced that missing `RuntimeError` boundary; `1759561` added the safe wrapper and omitted fail-closed fixtures. A further verifier probe showed that `raise ... from None` still retained raw `__context__` and that argparse echoed malformed argv. `25c0e97` creates sanitized failures only after leaving raw exception handlers and replaces default argparse errors with a constant, redacted failure path.

The retained residual limitation is deliberate and visible: Git-ignored files are not monitored. This validator attributes repository changes visible through tracked history, the index, the unstaged worktree, and ordinary Git-untracked paths; it is not a whole-filesystem or environment-integrity monitor.

## Deviations from Plan

### Evidence Refresh

The research inventory predated live Phase 2 work. Required re-inspection showed `WRONG-01`, `WRONG-02`, `WRONG-03`, and the narrow runtime behavior in `WRONG-06` are fixed in the live kernel. The document retains the mandated historical evidence classes by describing the observed invalidity of retained pre-fix artifacts, while stating the current implementation truth plainly.

### Authorized Shared-Worktree Concurrency Handling and Historical Limitation

During Task 2, the executor reported that it secondarily compared the exact sibling `260901-0gy-SUMMARY.md` and `260901-0gy-VERIFICATION.md` status, byte hashes or missing sentinels, and index entries before and after the scoped commit. The two `0fd` commits themselves provide durable proof that no `0gy` path was staged: `f7ea000` contains only the readiness document, and `43457a0` contains only the readiness document and docs-index route.

No retained secondary before/after snapshot was found under `/tmp` during the repair. The historical claim that both sibling artifacts were byte-for-byte identical is therefore **agent attestation, not cryptographically retained evidence**; the untracked sibling verification file's historical bytes are not auditable after the fact. This summary does not recreate or overstate that evidence. The new durable guard prevents the same gap prospectively by binding to the baseline commit/tree, retaining dirty/index identities, and rejecting unauthorized paths in every clean commit, including commit-then-revert sequences.

The `0gy` artifacts were never edited, staged, deleted, reset, or allowlisted by this task or its repair.

### External Publisher Block

The ACM policy page returned `HTTP 403`, which was correctly classified as `manual_required`, not success. Its destination and official ACM identity were manually confirmed and recorded in the readiness document.

## Decisions Made

- Phase 3 execution/data safety is the immediate priority; `BR-009` is the subsequent Phase 4 release-provenance priority before citable full-matrix compute.
- A row becomes `verified` only from observed closure evidence satisfying its complete exit criterion. A fixed live kernel with pending release regeneration remains `in_progress`.
- P2 public governance, external submissions, service security, and broadened survival tasks do not block the current accurately scoped package.
- Volatile manuscript, test, CI, requirements, and roadmap status remains owned by the existing documents; the new register links rather than duplicates it.

## Known Stubs

None.

## Threat Flags

| Flag | File | Description |
|---|---|---|
| threat_flag: local-validation-file-access | `scripts/validate_benchmark_readiness.py` | Reads declared documentation and Git object/index/worktree identities; paths are repository-contained or exact CLI inputs, and diagnostics omit contents. |
| threat_flag: optional-outbound-link-check | `scripts/validate_benchmark_readiness.py` | Optional HTTPS checks run only after the credential gate and static URL validation; diagnostics expose only source labels, stable URL IDs, sanitized hosts, and response disposition. |

## Self-Check: PASSED

The original deliverables and the durable validator/test files exist. Commits `f7ea000`, `43457a0`, `72dedf3`, `fd5d890`, `1759561`, and `25c0e97` are present in repository history, and the authenticated current-format baseline-bound check passed after the final repair commit.
