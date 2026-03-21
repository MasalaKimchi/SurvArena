#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "Creating virtual environment in ${VENV_DIR} using ${PYTHON_BIN}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Running environment check"
python scripts/check_environment.py

echo "Environment setup complete"
