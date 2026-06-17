#!/usr/bin/env bash
set -euo pipefail

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
    APPLY=1
elif [[ "${1:-}" != "" ]]; then
    echo "Usage: $0 [--apply]" >&2
    exit 2
fi

PATHS=(
    ".pytest_cache"
    ".ruff_cache"
    "htmlcov"
    "catboost_info"
    "AutogluonModels"
    "survarena.egg-info"
    "results/summary"
)

GLOBS=(
    "data/splits/*"
    "weight_checkpoint_*.pt"
)

echo "Local artifact cleanup preview"
if [[ "${APPLY}" -eq 0 ]]; then
    echo "Use --apply to remove listed paths."
fi

remove_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        return
    fi
    if [[ "${APPLY}" -eq 1 ]]; then
        rm -rf "${path}"
        echo "removed ${path}"
    else
        echo "would remove ${path}"
    fi
}

for path in "${PATHS[@]}"; do
    remove_path "${path}"
done

for pattern in "${GLOBS[@]}"; do
    for path in ${pattern}; do
        remove_path "${path}"
    done
done

while IFS= read -r path; do
    remove_path "${path}"
done < <(find . -type d -name "__pycache__" -not -path "./.venv/*" -not -path "./venv/*" -not -path "./env/*")
