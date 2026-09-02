---
phase: quick-260901-0fd
verified: 2026-09-02T00:27:29Z
status: gaps_found
score: 5/6 plan truths verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/6
  gaps_closed:
    - "BR-001, BR-002, and BR-005 now truthfully remain in_progress until release evidence is regenerated."
    - "Phase 3 execution/data safety is now the immediate priority, while BR-009 is correctly qualified as the subsequent Phase 4 release-provenance target."
    - "A retained validator scans exact input buffers before parsing or network access and redacts URL userinfo, query, fragment, and credential values."
    - "The retained Git guard binds an authenticated baseline commit/tree and detects unauthorized clean commits and commit-then-revert attempts."
    - "BR, DOM, crosswalk, dependency, requirement, and status validation now uses bounded grammar and rejects malformed, unknown, self, and cyclic references."
  gaps_remaining:
    - "Scope diagnostics and baseline metadata expose credential-shaped repository path names verbatim."
    - "New ignored files are outside the guard's status snapshot despite the validation document's broad every-new/untracked-path claim."
  regressions: []
gaps:
  - truth: "Validation never discloses credential-like values."
    status: failed
    reason: "Document and URL values are safely handled, but scope diagnostics render repository-relative paths verbatim and the baseline serializes them. An adversarial untracked filename shaped like a Hugging Face token was reproduced unchanged in the diagnostic output."
    artifacts:
      - path: "scripts/validate_benchmark_readiness.py"
        issue: "format_scope_issues emits issue.path verbatim; baseline dirty-entry keys also retain raw paths."
    missing:
      - "Secret-scan or redact path metadata before serialization and diagnostics while retaining a stable non-sensitive path identity for investigation."
      - "Add a credential-shaped filename fixture proving no path-derived secret reaches stdout, stderr, exceptions, or baseline records."
  - truth: "The global worktree guard detects every new or changed path outside the exact allowlist."
    status: partial
    reason: "Commit history, staged, unstaged, and ordinary untracked changes are protected, but Git status is invoked without ignored paths. A new ignored file created after the baseline produced zero scope issues."
    artifacts:
      - path: "scripts/validate_benchmark_readiness.py"
        issue: "_status_paths uses --untracked-files=all without --ignored, while VALIDATION.md states that a new path blocks completion."
      - path: ".planning/quick/260901-0fd-audit-survarena-s-benchmark-readiness-ag/260901-0fd-VALIDATION.md"
        issue: "The safety contract does not disclose that ignored files are outside its coverage."
    missing:
      - "Either include ignored paths in the baseline/final identity comparison or narrow the written contract explicitly and add a separate check for prohibited generated result/state paths."
      - "Add an adversarial ignored-file fixture."
---

# Quick 260901-0fd: Benchmark Readiness Audit Verification

**Task goal:** Audit SurvArena against accepted benchmark standards and create a durable, prioritized improvement document that identifies the highest priority and whether current evidence supports manuscript-grade claims.

**Verified:** 2026-09-02T00:27:29Z
**Status:** gaps_found
**Re-verification:** Yes — after repair commit `72dedf3`

## Publication-Readiness Verdict

The audit's scientific conclusion remains supported: **SurvArena is a credible internal benchmark framework and pre-release kernel, but its evidence is not comprehensive or robust enough for manuscript-grade performance claims or a citable benchmark release.** Retained matrices predate behavior-changing fixes; group safety, killable execution, truthful HPO semantics, immutable dataset and protocol identity, canonical provenance, release regeneration, and independent reproduction remain incomplete.

The priority statement is now correct and evidence-based. **Phase 3 execution and data safety** is the immediate critical path, led by `BR-007` killable isolation, `BR-004` group-safe validation, and the Phase 3 resource portions of `BR-006`. `BR-009` is correctly framed as the highest subsequent **Phase 4 release-provenance** target before citable full-matrix computation, after execution, dataset-identity, and protocol prerequisites.

## Goal Achievement

### Observable Truths

| # | Plan truth | Status | Evidence |
|---|---|---|---|
| 1 | Readers can distinguish internal, research/citable, and deferred public/maintained maturity tiers. | VERIFIED | The verdict and three named tiers provide separate gates, evidence, and naming limits; “official” remains explicitly project-relative. |
| 2 | Findings distinguish observed defects, missing evidence, and future scope. | VERIFIED | `WRONG-01..08` retain observed live or invalidated-artifact evidence, `WRONG-09` is missing evidence, and `BR-016..019` remain deferred future scope rather than current bugs. |
| 3 | The register identifies where conclusions can be wrong with live source evidence and adjacent external criteria. | VERIFIED | P0/P1 observed claims remain supported by the cited live implementation; the focused scientific and metric tests pass. Exact engineering remedies are presented as SurvArena design choices rather than universal mandates from the cited standards. |
| 4 | Every improvement row has stable ownership, truthful status, remediation, falsifiable exit criteria, dependencies, and target placement. | VERIFIED | Exact `BR-001..019`; 3 `in_progress`, 11 `open`, 1 `blocked`, 4 `deferred_scope`, and 0 `verified`. `BR-001/002/005` no longer claim release closure, and BR-009 maps to Phase 4 with explicit prerequisites. |
| 5 | The register covers the full audit rubric, remains distinct from volatile status owners, and is discoverable. | VERIFIED | Exact 13 domains, 19 register rows, and 22 crosswalk items are complete and reachable; ownership links resolve and the docs index contains one readiness route. |
| 6 | Citations are canonical, blocked sources are routed manually, dirty evidence is accounted for, and validation never discloses credential-like values. | FAILED | Current citations, links, manual-source handling, document secret scan, and URL diagnostics pass. Scope diagnostics and baseline metadata still expose credential-shaped path names verbatim. |

**Score:** 5/6 plan truths verified

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `docs/benchmark_readiness.md` | Durable maturity verdict, evidence ledger, prioritized register, crosswalk, and maintenance contract | VERIFIED | Substantive and source-backed; corrected status and sequencing semantics; exact required row sets. |
| `docs/index.md` | One Reference Docs route to the readiness register | VERIFIED | Exactly one Markdown route resolves to the readiness document. |
| `scripts/validate_benchmark_readiness.py` | Durable replacement for the flawed inline validation | PARTIAL | Core structural, secret-first URL, dependency/status, and authenticated Git-history checks work; path redaction and ignored-file coverage remain incomplete. |
| `tests/test_benchmark_readiness_validator.py` | Adversarial contract tests | PARTIAL | Forty-two tests pass and cover the repaired core, but omit credential-shaped path and ignored-file cases. |

## Re-Verification of Prior Gaps

| Prior concern | Re-verification evidence | Status |
|---|---|---|
| Truthful statuses and priority sequencing | `BR-001`, `BR-002`, and `BR-005` are `in_progress`; the document and register place execution/data safety in Phase 3 and BR-009 in Phase 4 after `BR-004`, `BR-007`, `BR-008`, and `BR-010`. This matches ROADMAP Phase 3/4 and PROJECT critical-path text. | CLOSED |
| Secret-first ordering and safe URL diagnostics | `validate_files` reads the four exact buffers once and runs `_secret_gate` before Markdown parsing or network work. Fixtures cover AWS access/secret/session, HF, bearer, GitHub, OpenAI, private key, generic token/signature assignments, and nested percent encoding. URL errors use stable identities and omit userinfo/query/fragment. | CORE CLOSED; PATH-METADATA GAP |
| Baseline-bound worktree guard | Authenticated baseline digest and commit/tree check pass. Independent tampered-tree, unauthorized clean-commit, and commit-then-revert probes reject. Twelve built-in adversarial cases pass. Ignored files remain invisible. | PARTIAL |
| Bounded references, dependencies, and statuses | Current document passes exact ordered row sets, whole-cell known references, requirement reconciliation, no self-reference, acyclic dependency/replacement graphs, status rules, date checks, and BR-009 prerequisite checks. Malformed adversarial fixtures reject. | CLOSED |

## Structural, Citation, and Link Checks

| Check | Result | Status |
|---|---|---|
| Exact structures | 9 WRONG rows, 13 DOM rows, 19 BR rows, 22 crosswalk rows | PASS |
| Status distribution | 3 in-progress, 11 open, 1 blocked, 4 deferred-scope, 0 verified | PASS |
| References and graph | Bounded and known; sorted where required; no self-dependency or cycle; every BR row reachable | PASS |
| Provenance | 93 canonical tracked, in-root VERIFIED tags across 20 files; line ranges valid; no cited file currently dirty | PASS |
| Local links | 25 local Markdown links/fragments resolve | PASS |
| External-source handoff | 13 authoritative sources; ACM 403 remains the sole exact `manual_required` record | PASS |
| Document ownership | Protocol, test status, CI/reproducibility, manuscript, requirements, and roadmap owners are linked without duplicated volatile status | PASS |

## Behavioral Spot-Checks

| Behavior | Command/check | Result | Status |
|---|---|---|---|
| Full focused suite | `pytest` over validator, scientific comparison, and metric contracts | 77 passed; 14 dependency deprecation warnings | PASS |
| Validator-only suite | `pytest --collect-only` plus execution | 42 collected and passed | PASS |
| Static validator | Offline `check` | 9/13/19/22, 93 tags, 25 links; `git_scope_checked=0` disclosed | PASS |
| Authenticated scope gate | Repair baseline plus retained SHA-256 and exact allowlist | PASS with `git_scope_checked=1` | PASS |
| Git adversarial suite | `self-test` | 12/12 passed | PASS |
| Independent commit/tree probe | Tampered tree, unauthorized clean commit, and commit-then-revert | All rejected; both offending commits remained visible | PASS |
| Credential-shaped path probe | New non-allowlisted token-shaped filename, then format scope issues | Filename emitted verbatim | FAIL |
| Ignored-file probe | New ignored file after baseline | Zero scope issues | FAIL |
| Lint/compile/diff | Ruff, compileall, and repair diff whitespace checks | PASS | PASS |

## Commit Scope and Unrelated Work

Repair commit `72dedf3` contains exactly:

- `docs/benchmark_readiness.md`
- `scripts/validate_benchmark_readiness.py`
- `tests/test_benchmark_readiness_validator.py`

It contains no deletion, planning artifact, runtime source, result file, state file, or `0gy` path. Every tracked sibling `0gy` blob is identical between parent `f116292` and `72dedf3`. The authenticated repair baseline also excludes every `0gy` path from its allowlist. The original untracked-file byte attestation was not retained and remains historically unauditable, but the repair itself did not alter sibling work.

## Requirements and Deferred Scope

This quick task declares no milestone requirement IDs. The register now reconciles its related phase/requirement cells against the authoritative requirement checklist and roadmap. Gate C/public governance and broadened survival-task rows remain explicitly deferred and do not block the current accurately scoped Python toolkit.

## Anti-Patterns and Calibration Notes

No stub or unresolved debt marker occurs in the deliverable or repair files. One validation sign-off is broader than its implementation: `VALIDATION.md` says citation validation records dirty cited-file hashes, while the retained provenance function checks syntax, containment, tracking, and line bounds but does not itself populate or compare the dirty-hash table. Current cited files are clean, so this does not falsify the present document; it should be narrowed or implemented before relying on the gate after a dirty-evidence audit.

The external criteria support the broad benchmark principles. SurvArena's exact SQLite, process-tree, and support-policy mechanisms remain project synthesis and should not be represented as uniquely required by W3C PROV, AutoML Benchmark guidance, or Demšar.

## Gaps Summary

The repaired readiness document itself now achieves the intended scientific audit: it is candid, structurally complete, prioritized correctly, and explicit that SurvArena is not manuscript-ready. The retained validator also closes the material URL, clean-commit, and reference/dependency gaps. Completion is still blocked by two narrow but reproducible assurance failures: credential-like path values can reach diagnostics/baselines, and ignored new files bypass the stated global worktree guard.

---

_Verified: 2026-09-02T00:27:29Z_
_Verifier: gsd-verifier_
