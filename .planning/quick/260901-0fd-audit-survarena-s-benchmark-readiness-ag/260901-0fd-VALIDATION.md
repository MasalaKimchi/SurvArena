---
quick_id: 260901-0fd
phase: quick-260901-0fd
status: approved
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-01
---

# Quick Task 260901-0fd — Validation Strategy

## Validation Infrastructure

| Property | Value |
|---|---|
| Artifact type | Documentation-only benchmark-readiness register and docs-index route |
| Automated framework | Repository-local Python 3.10+ assertions, Git plumbing, `curl`, and direct text checks |
| Source of commands | Task-level `<verify><automated>` blocks in `260901-0fd-PLAN.md` |
| Manual mode | End-of-plan human checks, per `workflow.human_verify_mode: end-of-phase` |
| Scope protection | Global pre-edit dirty-worktree snapshot plus staged/unstaged/untracked content/index comparison |

No production-code test scaffold is needed: both deliverables are Markdown, and the plan supplies executable structural, provenance, link, redaction, whitespace, and scope assertions.

## Sampling Rate

- Before Task 1 writes either target: capture the global dirty-worktree baseline outside the repository.
- After Task 1: validate exact maturity/ledger table contracts and run the global scope comparator.
- After Task 2: run the complete structural, citation/hash, local-link/anchor, external-source, secret-redaction, whitespace, and global scope suite.
- Before summary/commit: complete semantic evidence and source-identity review, then rerun Task 2's full automated command.

## Per-Task Verification Map

| Task | Required evidence | Automated proof | Manual proof | Blocking failure |
|---|---|---|---|---|
| `260901-0fd-01` | Exact three-tier verdict; exact `WRONG-01..09` and `DOM-01..13` table schemas/sets; `WRONG-01..08 = OBSERVED DEFECT`, `WRONG-09 = MISSING EVIDENCE`; nonempty required cells; descriptive source link or exact `DOM-01..13` criterion per WRONG row; domain backlog/rationale field | Structural Markdown parser in Task 1; global worktree guard verifies every non-allowlisted pre-existing path is unchanged in status, worktree hash, and index identity | Confirm the first eight defect claims against live behavior, the ninth as absence of citable evidence, and every adjacent criterion's actual meaning | Any schema/set/cell/classification/criterion failure, unknown DOM reference, raw/malformed criterion URL, or out-of-allowlist state change |
| `260901-0fd-02` | Exact owned register schema; allowed owners/statuses/classes; named steward; complete research crosswalk; resolved DOM-to-BR links; canonical citations and dirty-file hashes; valid local links/anchors; exactly one docs-index route; external-link disposition; redacted secret scan | Structural cross-table parser; whole-document citation parser with containment/tracked/range checks; dirty-citation SHA-256 comparison; two-document local-link/anchor validator; bounded external GET classifier; file/line/pattern-only credential scanner; direct whitespace/EOF checks; final global worktree guard | Confirm every cited range's semantics, every external final destination/source identity, every `manual_required` record, and every P0/P1 exit criterion's falsifiability | Any orphan/missing mapping, malformed/escaping citation, hash mismatch, broken link/anchor, invalid external URL, unrecorded automation block, secret-like hit, whitespace fault, duplicate index route, or out-of-allowlist change |

## Automated-to-Manual Handoff

External checks have three dispositions:

- `ok`: automated `2xx`/`3xx`; still requires manual final-destination and primary/official-source identity confirmation.
- `manual_required`: `403`, `406`, `418`, `429`, `451`, network/curl failure, or publisher `5xx`; requires a row in `Manual External-Source Checks` with final destination, confirmed identity, and review date.
- `invalid`: other non-success responses; the URL must be replaced before completion.

Automation-blocked links are never silently counted as valid. Manual review confirms source identity, not merely that a page opens.

## Dirty-Worktree Safety Contract

The pre-task snapshot records every initially staged, unstaged, or untracked path with Git status, worktree kind/content SHA-256, and index entry. The final comparison operates over the repository-global status rather than a path-filtered diff. It ignores only these exact authorized outputs:

- `docs/benchmark_readiness.md`
- `docs/index.md`
- `.planning/quick/260901-0fd-audit-survarena-s-benchmark-readiness-ag/260901-0fd-SUMMARY.md`

An unchanged pre-existing user modification is accepted. A new path, removed path, status transition, staged-index change, or content change outside the allowlist blocks completion.

## Secret-Handling Contract

The scanner covers AWS, GitHub, OpenAI, Hugging Face, private-key, bearer-token, and generic credential-assignment forms. A match record contains only the file, line number, and pattern name; the matched value is redacted by omission and must never be printed, serialized, or included in an assertion.

## Sign-Off

- [x] Every task has an automated proof.
- [x] Every task has mapped semantic/manual evidence where automation cannot establish meaning.
- [x] Exact row sets, schemas, allowed values, and cross-table reachability are machine checked.
- [x] Citation validation covers every occurrence and records hashes for dirty cited files.
- [x] Automation-blocked external sources require explicit manual evidence.
- [x] The dirty-worktree guard distinguishes pre-existing user work from task-created changes.
- [x] No production source, test, config, result, or status document is part of this task.

**Approval:** approved 2026-09-01
