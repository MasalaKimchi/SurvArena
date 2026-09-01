# Phase 1: Truthful Green Baseline - Research

**Researched:** 2026-08-31
**Domain:** Python benchmark verification, test contracts, and release-gate truthfulness
**Confidence:** HIGH

<user_constraints>
## User Constraints

### Locked Decisions
- Resolve the current four failures without broad skips or weakened behavior.
- Keep split-cache migration explicit and fail-closed.
- Make the strict audit self-contained without adding an optional rendering dependency.
- Establish a real but incremental static-type gate.
- Do not change scientific protocol behavior or generate new benchmark evidence in this phase.

### the agent's Discretion
- Internal helper placement, executor-test seam, initial mypy module list, and fixture organization.

### Deferred Ideas
- Scientific comparison, worker isolation, typed runner/store integration, representative protocol, and release reproduction are later phases.
</user_constraints>

<architectural_responsibility_map>
## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|--------------|----------------|-----------|
| Test contract alignment | Tests | Benchmark/core modules | Tests must assert intended live behavior and expose a narrow seam only where process isolation requires it. |
| Static verification | Tooling/CI | Typed core modules | Configuration and commands must have one declared source of truth. |
| Publishability audit | Scripts/reporting | Dependencies/docs | Audit rendering should be deterministic and dependency-light. |
| Split compatibility | Data layer | CLI/docs | Data layer computes field differences; entry points present recovery guidance. |
| Evidence status | Documentation | CI outputs | Claims must correspond to commands actually executed. |
</architectural_responsibility_map>

<research_summary>
## Summary

The four failures are contract drift rather than one shared production defect: an artifact test omits the new `hpo_mode` argument; a resume fixture lacks arm identity and therefore correctly triggers the missing HPO unit; a thread-executor monkeypatch no longer observes process workers; and a Parquet guard test assumes `pyarrow` is unavailable even though it is a required dependency. Each needs a deterministic test of both intended branches.

The strict audit failure is caused by `DataFrame.to_markdown`, which imports optional `tabulate` at call time. A small internal Markdown renderer for the audit's simple scalar tables avoids expanding runtime dependencies. Split-cache invalidation is correctly fail-closed but reports only a generic payload mismatch; a stable key-level diff and explicit recovery command make the new fingerprint behavior usable without weakening identity.

**Primary recommendation:** implement three independently verifiable plans: green code/test/tool contracts, self-contained audit plus cache compatibility UX, then truthful docs/CI/smoke consolidation.
</research_summary>

<architecture_patterns>
## Architecture Patterns

### Deterministic Optional-Dependency Testing

Patch the module import boundary or dependency capability helper so installed and absent branches are tested independently of the machine. Do not infer an optional-dependency branch from the current environment.

### Injectable Scheduling Boundary

Keep production process isolation. Test resolved `n_jobs` through a small executor factory or `_execute_run_units` unit seam; use picklable top-level fixtures for a real multi-process smoke. Do not rely on child processes mutating a parent dictionary.

### Structured Compatibility Diff

Compare expected and observed split-manifest payload keys in deterministic sorted order. Report missing, added, and changed fields plus the exact manifest path. Preserve the current opt-in regeneration behavior.

### Dependency-Light Release Script

The audit only needs pipe-separated Markdown tables. Escape `|`, normalize newlines, render headers/separators/rows, and test exact output. Pulling a presentation package into the release gate is unnecessary.

### Incremental Type Gate

Pin mypy 2.3.1 in the dev extra; configure Python 3.10 and missing third-party stubs deliberately; check `survarena/core` plus Phase 1 touched modules. Record exclusions instead of implying whole-package type safety.
</architecture_patterns>

<common_pitfalls>
## Common Pitfalls

### Updating Tests to Match Accidental Behavior
Verify intended contracts from the milestone decisions first. In particular, artifact failures and legacy resume identity need explicit behavior rather than simply adding missing arguments/columns until tests pass.

### Mocking Across Process Boundaries
Parent-memory monkeypatch counters are not shared with fork/spawn workers. Test the scheduler seam separately and reserve real subprocess tests for picklable behavior and result counts.

### Hiding Red Tests with Conditional Skips
The Parquet test must execute both capability branches through controlled imports; a skip when `pyarrow` exists would leave the actual export path unverified.

### Overclaiming Type Coverage
An incremental allow-list is acceptable only if commands and docs state exactly what is checked and expansion is planned.

### Regenerating Splits in Tests Without Proving the Gate
Tests need separate assertions for default rejection, diagnostic content, explicit regeneration, and new-manifest reuse.
</common_pitfalls>

## Validation Architecture

### Fast feedback layers

1. Targeted contract tests for each task (`pytest` node IDs, under 20 seconds).
2. Phase tests: `tests/test_benchmark.py`, `tests/test_core_results.py`, audit tests, and data/split tests.
3. Full gate: Ruff, declared mypy command, compileall, full pytest.
4. Semantic E2E: clean-root CoxPH/WHAS500 one-seed run plus strict audit execution.

### Required new/updated automated coverage

- Artifact identity includes `hpo_mode`; serialization failure behavior is asserted.
- Legacy resume without arm identity completes only `no_hpo`; explicit HPO identity suppresses only matching HPO work.
- Configured process worker count is observed at the scheduler boundary; real result count is validated independently.
- Parquet export succeeds when available and raises the documented error when import is forced unavailable.
- Markdown rendering has exact escaping/output fixtures and strict audit reaches a verdict.
- Split mismatch message reports changed keys/path/recovery command; explicit regeneration produces reusable manifest.
- Mypy command and CI job use identical declared scope.
- Documentation status claims are covered by the final executed command transcript and date.

### Nyquist sampling

Every plan task has an automated targeted command. Every wave ends with the phase test set. The final wave runs the complete baseline and E2E release checks. No three consecutive implementation tasks may pass without an automated behavior check.

<sources>
## Sources

### Primary
- Current SurvArena source/tests and executed four-failure targeted run, 2026-08-31.
- https://pypi.org/project/mypy/ — current mypy 2.3.1 release.
- https://mypy.readthedocs.io/en/stable/config_file.html — `pyproject.toml` configuration model.
- pytest monkeypatch and Python `concurrent.futures` behavior as exercised by the current suite.

</sources>

<metadata>
## Metadata

**Research scope:** current failures, release-audit dependency, split compatibility, type gate, documentation truthfulness.

**Confidence breakdown:**
- Test contracts: HIGH — reproduced directly.
- Audit/cache changes: HIGH — source paths and failure modes are explicit.
- Type gate: HIGH for configuration; initial error volume must be measured during execution.

**Research date:** 2026-08-31
**Valid until:** 2026-09-30
</metadata>

---
*Phase: 01-truthful-green-baseline*
*Research completed: 2026-08-31*
*Ready for planning: yes*
