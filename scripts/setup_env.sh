#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev}"

echo "Creating virtual environment in ${VENV_DIR} using ${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PY'
import sys

major, minor = sys.version_info[:2]
if major != 3 or minor not in {10, 11, 12}:
    raise SystemExit(
        f"Unsupported Python {major}.{minor}. "
        "Use Python 3.10, 3.11, or 3.12 for SurvArena environment setup."
    )
PY
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

INSTALL_TARGET="."
if [[ -n "${INSTALL_EXTRAS}" ]]; then
    INSTALL_TARGET=".[${INSTALL_EXTRAS}]"
fi

echo "Installing SurvArena package from ${INSTALL_TARGET}"
python -m pip install -e "${INSTALL_TARGET}"

echo "Running environment check"
CHECK_ARGS=()
if [[ ",${INSTALL_EXTRAS}," == *",foundation,"* ]]; then
    CHECK_ARGS+=(--include-foundation)
fi
python scripts/check_environment.py "${CHECK_ARGS[@]}"

echo "Environment setup complete"
