---
quick_id: 260901-0fd
phase: quick-260901-0fd
status: repaired
nyquist_compliant: true
wave_0_complete: true
created: 2026-09-01
---

# Quick Task 260901-0fd — Validation Strategy

## Validation Infrastructure

| Property | Value |
|---|---|
| Artifact type | Benchmark-readiness register, docs-index route, and focused durable validator |
| Automated framework | `scripts/validate_benchmark_readiness.py`, pytest, Git plumbing, and Ruff |
| Source of commands | Durable repair gate below; it supersedes the non-retained inline commands in `260901-0fd-PLAN.md` without rewriting plan history |
| Manual mode | End-of-plan human checks, per `workflow.human_verify_mode: end-of-phase` |
| Scope protection | Baseline Git commit/tree plus staged, unstaged, untracked, worktree-hash, and index identities; every intervening commit is inspected |

The validator is standard-library Python and has a focused pytest suite. It performs the credential gate before parsing Markdown or URLs, validates exact table/reference/status semantics, checks provenance and links, and optionally performs redacted external-source checks.

## Sampling Rate

- Before an audit or repair writes any target: capture the global Git baseline outside the repository.
- After Task 1: validate exact maturity/ledger table contracts and run the global scope comparator.
- After Task 2 or a repair: run the complete secret-first structural, citation, local-link/anchor, reference/dependency/status, and global scope suite.
- Before external checks: run the offline command. The `--external check` path repeats the secret gate before any URL processing or request.
- Before summary/commit: run the focused tests, scope self-test, Ruff, semantic evidence review, and the full checker against the captured baseline.

## Per-Task Verification Map

| Task | Required evidence | Automated proof | Manual proof | Blocking failure |
|---|---|---|---|---|
| `260901-0fd-01` | Exact three-tier verdict; exact `WRONG-01..09` and `DOM-01..13` table schemas/sets; `WRONG-01..08 = OBSERVED DEFECT`, `WRONG-09 = MISSING EVIDENCE`; nonempty required cells; descriptive source link or exact `DOM-01..13` criterion per WRONG row; domain backlog/rationale field | Structural Markdown parser in Task 1; global worktree guard verifies every non-allowlisted pre-existing path is unchanged in status, worktree hash, and index identity | Confirm the first eight defect claims against live behavior, the ninth as absence of citable evidence, and every adjacent criterion's actual meaning | Any schema/set/cell/classification/criterion failure, unknown DOM reference, raw/malformed criterion URL, or out-of-allowlist state change |
| `260901-0fd-02` | Exact owned register schema; allowed owners/statuses/classes; named steward; complete research crosswalk; resolved DOM-to-BR links; canonical citations; valid local links/anchors; exactly one docs-index route; external-link disposition; redacted secret scan | Durable cross-table parser; bounded whole-cell WRONG/DOM/BR/crosswalk/dependency grammar; self/cycle/status checks; whole-document provenance parser; local-link validator; secret-first external classifier; baseline commit/tree and dirty/index guard | Confirm every cited range's semantics, every external final destination/source identity, every `manual_required` record, and every P0/P1 exit criterion's falsifiability | Any orphan/missing mapping, malformed/unknown/self/cyclic reference, premature status, malformed/escaping citation, broken link/anchor, invalid external URL, unrecorded automation block, secret-like hit, duplicate index route, or out-of-allowlist change or commit |

## Durable Repair Gate

The following commands are the reproducible gate for this artifact. The baseline file remains outside the repository and contains only paths, status labels, Git identities, and content hashes—never file contents.

```bash
.venv/bin/python scripts/validate_benchmark_readiness.py snapshot-git \
  --output /tmp/survarena-260901-0fd-baseline.json

.venv/bin/python scripts/validate_benchmark_readiness.py check \
  --baseline /tmp/survarena-260901-0fd-baseline.json \
  --baseline-sha256 SHA256_PRINTED_BY_SNAPSHOT \
  --allow-path docs/benchmark_readiness.md \
  --allow-path docs/index.md \
  --allow-path scripts/validate_benchmark_readiness.py \
  --allow-path tests/test_benchmark_readiness_validator.py \
  --allow-path .planning/quick/260901-0fd-audit-survarena-s-benchmark-readiness-ag/260901-0fd-SUMMARY.md \
  --allow-path .planning/quick/260901-0fd-audit-survarena-s-benchmark-readiness-ag/260901-0fd-VALIDATION.md \
  --external offline

.venv/bin/python scripts/validate_benchmark_readiness.py self-test
.venv/bin/pytest -q tests/test_benchmark_readiness_validator.py
.venv/bin/ruff check scripts/validate_benchmark_readiness.py tests/test_benchmark_readiness_validator.py

# Run only after the offline gate. This command repeats the credential gate first.
.venv/bin/python scripts/validate_benchmark_readiness.py check --external check
```

Copy the digest printed by `snapshot-git` into `--baseline-sha256`; the check refuses an unauthenticated baseline. If an authorized output was already dirty when the snapshot was taken, the validator rejects overwriting it unless that exact path is additionally and consciously passed with `--accept-dirty-allow-path`. This repair used that exception only for its already-owned in-progress outputs, never for sibling work.

`check` rejects a baseline from another repository, a tampered or digest-mismatched baseline commit/tree pair, a non-ancestor baseline, unapproved merges, every non-allowlisted path in every intervening commit (including later revert commits), and changed non-allowlisted dirty/index identities. File identities include type, mode, and SHA-256. Allowlisting is exact-path only, and output records whether Git scope was checked.

The observed repair boundary used `/tmp/survarena-260901-0fd-repair-baseline.json` with externally retained digest `3dcab08519d131710e8246a71c8099b009789a55ac2979ea4a851d3bb2ac2c56`. Exact dirty-path acceptance was limited to the already-owned readiness document, validator, validator test, SUMMARY, and VALIDATION files. The docs index was allowlisted but was clean, and no `0gy` path was allowlisted or accepted.

## Automated-to-Manual Handoff

External checks have three dispositions:

- `ok`: automated `2xx`/`3xx`; still requires manual final-destination and primary/official-source identity confirmation.
- `manual_required`: `403`, `406`, `418`, `429`, `451`, network/curl failure, or publisher `5xx`; requires a row in `Manual External-Source Checks` with final destination, confirmed identity, and review date.
- `invalid`: other non-success responses; the URL must be replaced before completion.

Automation-blocked links are never silently counted as valid. Manual review confirms source identity, not merely that a page opens.

## Dirty-Worktree Safety Contract

The baseline records `HEAD`, its tree, a repository fingerprint, and every initially staged, unstaged, or untracked path with Git status, worktree kind/content SHA-256, and index entry. The final comparison operates over the repository-global status and every commit after the baseline. It ignores only exact `--allow-path` values; no prefix matching is used.

The repair invocation authorized only these exact outputs:

- `docs/benchmark_readiness.md`
- `docs/index.md`
- `scripts/validate_benchmark_readiness.py`
- `tests/test_benchmark_readiness_validator.py`
- `.planning/quick/260901-0fd-audit-survarena-s-benchmark-readiness-ag/260901-0fd-SUMMARY.md`
- `.planning/quick/260901-0fd-audit-survarena-s-benchmark-readiness-ag/260901-0fd-VALIDATION.md`

An unchanged pre-existing user modification is accepted. An unauthorized clean commit, commit-then-revert sequence, new path, removed path, status transition, staged-index change, rename, or content change outside the allowlist blocks completion.

## Secret-Handling Contract

The scanner covers AWS access-key IDs and assignments, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, GitHub, OpenAI, Hugging Face, private-key, bearer-token, and generic credential-assignment forms. It scans raw and safely percent-decoded lines. A match record contains only the file, line number, and named pattern; the matched value is omitted and must never be printed, serialized, or included in an assertion. URL validation diagnostics use only a stable URL identity and sanitized hostname; userinfo, query, and fragment are never emitted.

## Sign-Off

- [x] Every task has an automated proof.
- [x] Every task has mapped semantic/manual evidence where automation cannot establish meaning.
- [x] Exact row sets, schemas, allowed values, and cross-table reachability are machine checked.
- [x] Citation validation covers every occurrence and records hashes for dirty cited files.
- [x] Automation-blocked external sources require explicit manual evidence.
- [x] The Git guard distinguishes pre-existing user work from task-created dirty changes and post-baseline clean commits.
- [x] Disposable-repository tests cover dirty-content changes, index-only changes, new/removed untracked paths, renames, unauthorized paths, exact allowlisting, unauthorized clean commits, and commit-then-revert bypass attempts.
- [x] No benchmark runtime source, config, result, or state document is part of this task; the only executable addition is the focused validator and its tests.

**Approval:** repaired and revalidated 2026-09-01
