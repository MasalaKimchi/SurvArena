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
| Scope protection | Baseline Git commit/tree plus staged, unstaged, and ordinary Git-untracked identities; every intervening commit is inspected; Git-ignored cache/environment contents are explicitly excluded |

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
| `260901-0fd-01` | Exact three-tier verdict; exact `WRONG-01..09` and `DOM-01..13` table schemas/sets; `WRONG-01..08 = OBSERVED DEFECT`, `WRONG-09 = MISSING EVIDENCE`; nonempty required cells; descriptive source link or exact `DOM-01..13` criterion per WRONG row; domain backlog/rationale field | Structural Markdown parser in Task 1; Git scope guard verifies every non-allowlisted tracked, staged, unstaged, or ordinary Git-untracked identity | Confirm the first eight defect claims against live behavior, the ninth as absence of citable evidence, and every adjacent criterion's actual meaning | Any schema/set/cell/classification/criterion failure, unknown DOM reference, raw/malformed criterion URL, or out-of-allowlist Git-visible state change |
| `260901-0fd-02` | Exact owned register schema; allowed owners/statuses/classes; named steward; complete research crosswalk; resolved DOM-to-BR links; canonical citations; valid local links/anchors; exactly one docs-index route; external-link disposition; redacted secret scan | Durable cross-table parser; bounded whole-cell WRONG/DOM/BR/crosswalk/dependency grammar; self/cycle/status checks; whole-document provenance parser; local-link validator; secret-first external classifier; baseline commit/tree and dirty/index guard | Confirm every cited range's semantics, every external final destination/source identity, every `manual_required` record, and every P0/P1 exit criterion's falsifiability | Any orphan/missing mapping, malformed/unknown/self/cyclic reference, premature status, malformed/escaping citation, broken link/anchor, invalid external URL, unrecorded automation block, secret-like hit, duplicate index route, or out-of-allowlist change or commit |

## Durable Repair Gate

The following commands are the reproducible gate for this artifact. The baseline file remains outside the repository and contains only sanitized path labels, deterministic path digests, status labels, Git identities, and content hashes—never raw credential-shaped paths or file contents.

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

`check` rejects a baseline from another repository, a tampered or digest-mismatched baseline commit/tree pair, a non-ancestor baseline, unapproved merges, every non-allowlisted tracked path in every intervening commit (including later revert commits), and changed non-allowlisted staged, unstaged, or ordinary Git-untracked identities. File identities include type, mode, and SHA-256. Allowlisting is exact-path only. Output records both `git_scope_checked` and `git_ignored_paths_checked=0` so the boundary is not implicit.

The observed repair boundary used `/tmp/survarena-260901-0fd-repair-baseline.json` with externally retained digest `3dcab08519d131710e8246a71c8099b009789a55ac2979ea4a851d3bb2ac2c56`. Exact dirty-path acceptance was limited to the already-owned readiness document, validator, validator test, SUMMARY, and VALIDATION files. The docs index was allowlisted but was clean, and no `0gy` path was allowlisted or accepted.

The intermediate `/tmp/survarena-260901-0fd-path-repair-baseline.json` digest `eed1b2f4c2d00bde9f20a35f2fb12d0cd1e2be6f85fa46cd2e21ea7611275aae` is **non-authoritative for path-redaction evidence**. It was created before universal digest-only labels and retained benign raw path labels, so it must not be cited as proof of the final serialization contract. Commit `fd5d890` itself remains exactly scoped by Git history, but this document does not retrofit stronger historical baseline evidence.

The authoritative retry boundary is `/tmp/survarena-260901-0fd-symlink-retry-baseline.json` with externally retained SHA-256 `17b75ea0e527b620a56e25faed67e984733de745735149d187c8cb46c224ff90`. It was generated by the digest-only `fd5d890` format immediately before retry edits; all three pre-existing dirty planning entries serialize as full digest keys plus `<redacted-path>` labels. SUMMARY and VALIDATION received explicit dirty acceptance for their already-owned metadata updates. The separately modified verifier-owned VERIFICATION artifact was neither touched nor allowlisted and had to remain byte/index/status-identical. The authenticated check passed before and after `1759561`, which contains only the validator and test.

After the verifier-owned report changed, the older boundary correctly failed rather than silently accepting the concurrent edit. The final exception/CLI boundary therefore uses `/tmp/survarena-260901-0fd-exception-cli-baseline.json` with externally retained SHA-256 `8387098360a59cfd91087f42996b7650f91886cdb6ff6513fc099c216e9f2e2e`. All five dirty entries serialize only as digest keys and redacted labels. Exact dirty acceptance covers only the validator, test, SUMMARY, and VALIDATION; the verifier-owned VERIFICATION artifact remains unallowlisted and had to stay unchanged. The authenticated check passed before and after `25c0e97`, which contains only the validator and test.

## Automated-to-Manual Handoff

External checks have three dispositions:

- `ok`: automated `2xx`/`3xx`; still requires manual final-destination and primary/official-source identity confirmation.
- `manual_required`: `403`, `406`, `418`, `429`, `451`, network/curl failure, or publisher `5xx`; requires a row in `Manual External-Source Checks` with final destination, confirmed identity, and review date.
- `invalid`: other non-success responses; the URL must be replaced before completion.

Automation-blocked links are never silently counted as valid. Manual review confirms source identity, not merely that a page opens.

## Dirty-Worktree Safety Contract

The baseline records `HEAD`, its tree, a repository fingerprint, and every initially staged, unstaged, or ordinary Git-untracked path with Git status, worktree kind/content SHA-256, and index entry. Raw paths remain in memory only; serialized keys are full SHA-256 path digests and every label passes through the centralized digest-only path sanitizer. The final comparison operates over Git-visible repository status and every commit after the baseline. It ignores only exact `--allow-path` values; no prefix matching is used.

This is a **Git repository-change attribution guard**, not a whole-filesystem monitor. Files excluded by Git ignore rules—such as caches, virtual environments, and generated environment contents—are outside this contract unless a future caller opts into a separate ignored-path policy. Creating or changing an ignored file therefore does not produce a scope issue, and the validator reports that exclusion explicitly. Tracked files, the index, unstaged changes, ordinary Git-untracked paths, renames, modes, and post-baseline commits remain covered.

The repair invocation authorized only these exact outputs:

- `docs/benchmark_readiness.md`
- `docs/index.md`
- `scripts/validate_benchmark_readiness.py`
- `tests/test_benchmark_readiness_validator.py`
- `.planning/quick/260901-0fd-audit-survarena-s-benchmark-readiness-ag/260901-0fd-SUMMARY.md`
- `.planning/quick/260901-0fd-audit-survarena-s-benchmark-readiness-ag/260901-0fd-VALIDATION.md`

An unchanged pre-existing Git-visible user modification is accepted. An unauthorized clean commit, commit-then-revert sequence, new ordinary Git-untracked path, removed Git-visible path, status transition, staged-index change, rename, or content/mode change outside the allowlist blocks completion. Git-ignored files remain intentionally excluded as stated above.

## Secret-Handling Contract

The scanner covers AWS access-key IDs and assignments, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, GitHub, OpenAI, Hugging Face, private-key, bearer-token, and generic credential-assignment forms. It scans raw and safely percent-decoded lines. A single path-output boundary is stricter than pattern recognition: **every** path label becomes `<redacted-path>` plus a short digest before any diagnostic, exception, JSON, or baseline serialization, while full deterministic path digests preserve comparison identity. Path resolution, read, stat, and decode failures—including self-referential symlink loops—are converted by one safe wrapper without retaining the raw path, original exception text, `__context__`, or `__cause__`. Sanitized exceptions are created only after control exits the raw exception handler. This also protects token families not recognized by the configured content scanner. A document match record contains only the sanitized file identity, line number, and named pattern. URL validation diagnostics use only a stable URL identity; userinfo, query, fragment, and credential-shaped destinations are never emitted. A custom argument parser discards argparse's user-controlled error text and returns one constant redacted CLI failure; malformed options, choices, positional arguments, paths, and URL-shaped tokens are never echoed.

## Sign-Off

- [x] Every task has an automated proof.
- [x] Every task has mapped semantic/manual evidence where automation cannot establish meaning.
- [x] Exact row sets, schemas, allowed values, and cross-table reachability are machine checked.
- [x] Citation validation covers every occurrence, containment/tracking, and line bounds; this audit had no dirty cited file, and any future dirty citation requires a separately recorded snapshot hash before acceptance.
- [x] Automation-blocked external sources require explicit manual evidence.
- [x] The Git guard distinguishes pre-existing user work from task-created dirty changes and post-baseline clean commits.
- [x] Disposable-repository tests cover dirty-content changes, index-only changes, new/removed ordinary Git-untracked paths, renames, unauthorized paths, exact allowlisting, unauthorized clean commits, commit-then-revert bypass attempts, credential-shaped path redaction, and the explicit Git-ignored exclusion.
- [x] Exception-surface tests cover safe string/repr/traceback rendering, absent context/cause, self-looping symlinks, and read/stat/decode failures; subprocess tests cover six malformed CLI classes without argv disclosure.
- [x] No benchmark runtime source, config, result, or state document is part of this task; the only executable addition is the focused validator and its tests.

**Approval:** repaired and revalidated 2026-09-01
