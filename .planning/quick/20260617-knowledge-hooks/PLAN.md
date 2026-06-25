# Quick Task 20260617: Knowledge Hooks

**Status:** In progress
**Task:** Refresh markdown wiki knowledge and code-review-graph data automatically on local commit and push events.

## Tasks

1. Add a small hook-safe script for updating code-review-graph and generated wiki pages.
   - Files: `scripts/update_code_knowledge.sh`
   - Verify: Run the script manually and confirm graph/wiki output is generated.

2. Add tracked Git hook wrappers and configure this checkout to use them.
   - Files: `scripts/git-hooks/post-commit`, `scripts/git-hooks/pre-push`
   - Verify: `git config core.hooksPath` points to `scripts/git-hooks`.

## Constraints

- Do not block commits or pushes by default if local knowledge tooling is unavailable.
- Keep generated code-review-graph artifacts local under `.code-review-graph/`.
- Make strict failure behavior opt-in for users who want push protection.
