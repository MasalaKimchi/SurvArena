#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
VENV_DIR="${VENV_DIR:-.venv}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev,foundation-tabpfn,foundation-tabarena}"

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
case ",${INSTALL_EXTRAS}," in
    *",foundation,"*)
        CHECK_ARGS+=(--include-foundation)
        ;;
    *)
        FOUNDATION_METHODS=()
        if [[ ",${INSTALL_EXTRAS}," == *",foundation-tabpfn,"* ]]; then
            FOUNDATION_METHODS+=(tabpfn_survival)
        fi
        if [[ ",${INSTALL_EXTRAS}," == *",foundation-tabarena,"* ]]; then
            FOUNDATION_METHODS+=(tabicl_survival tabm_survival realtabpfn_survival)
        fi
        if [[ ",${INSTALL_EXTRAS}," == *",foundation-mitra,"* ]]; then
            FOUNDATION_METHODS+=(mitra_survival_frozen)
        fi
        if [[ "${#FOUNDATION_METHODS[@]}" -gt 0 ]]; then
            FOUNDATION_METHODS_CSV="$(IFS=,; echo "${FOUNDATION_METHODS[*]}")"
            CHECK_ARGS+=(--include-foundation --foundation-methods "${FOUNDATION_METHODS_CSV}")
        fi
        ;;
esac
python scripts/check_environment.py "${CHECK_ARGS[@]}"

echo "Environment setup complete"
