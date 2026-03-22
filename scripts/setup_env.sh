#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv}"

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
python -m pip install -r requirements.txt

echo "Running environment check"
python scripts/check_environment.py

echo "Environment setup complete"
