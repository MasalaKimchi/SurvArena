#!/usr/bin/env bash
set -u

event="${1:-manual}"
strict="${SURVARENA_KNOWLEDGE_HOOK_STRICT:-0}"

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root" || exit 0

log_dir="$repo_root/.code-review-graph/logs"
mkdir -p "$log_dir" 2>/dev/null || true
log_file="$log_dir/knowledge-hook.log"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$log_file" >&2
}

soft_fail() {
  log "knowledge refresh failed: $*"
  if [ "$strict" = "1" ]; then
    exit 1
  fi
  exit 0
}

if ! command -v code-review-graph >/dev/null 2>&1; then
  soft_fail "code-review-graph CLI is not installed or not on PATH"
fi

mkdir -p "$repo_root/.code-review-graph" 2>/dev/null || true
lock_dir="$repo_root/.code-review-graph/.knowledge-hook.lock"
if ! mkdir "$lock_dir" 2>/dev/null; then
  log "knowledge refresh already running; skipping $event"
  exit 0
fi
trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT

base_ref="HEAD~1"
if [ "$event" = "pre-push" ]; then
  while read -r local_ref local_sha remote_ref remote_sha; do
    case "${remote_sha:-}" in
      ""|0000000000000000000000000000000000000000)
        ;;
      *)
        base_ref="$remote_sha"
        break
        ;;
    esac
  done
elif [ "$event" = "post-commit" ]; then
  if ! git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
    base_ref="HEAD"
  fi
fi

log "starting $event knowledge refresh with base $base_ref"

if [ ! -f "$repo_root/.code-review-graph/graph.db" ]; then
  code-review-graph build --repo "$repo_root" >>"$log_file" 2>&1 \
    || soft_fail "full graph build failed; see $log_file"
else
  code-review-graph update --repo "$repo_root" --base "$base_ref" >>"$log_file" 2>&1 \
    || soft_fail "incremental graph update failed; see $log_file"
fi

code-review-graph wiki --repo "$repo_root" >>"$log_file" 2>&1 \
  || soft_fail "wiki generation failed; see $log_file"

log "completed $event knowledge refresh"
