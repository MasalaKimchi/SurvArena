# Quick Task 20260617: Dead Code Cleanup

**Status:** In progress
**Task:** Audit redundant, unused, legacy, and dead code after the refactor pass; safely remove or simplify behavior-preserving targets.

## Tasks

1. Use a subagent to inspect the codebase for safe cleanup candidates.
   - Files: read-only broad scan, then bounded recommendations or patches.
   - Verify: findings include evidence and avoid public API/config/artifact removals unless proven safe.

2. Apply high-confidence cleanup only.
   - Files: source/tests/configs as indicated by evidence.
   - Verify: `python -m ruff check survarena tests scripts` and `pytest`.

## Constraints

- Preserve public APIs, method IDs, CLI behavior, artifact fields, config IDs, and compatibility aliases.
- Do not remove legacy compatibility shims without tests or clear internal-only status.
- Keep changes small and reviewable.
