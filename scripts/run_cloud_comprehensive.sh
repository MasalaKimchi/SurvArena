#!/usr/bin/env bash
set -euo pipefail

# Launches the comprehensive all-models benchmark in a cloud or remote worker.
# Usage:
#   scripts/run_cloud_comprehensive.sh                 # full run
#   scripts/run_cloud_comprehensive.sh --dry-run       # validate only
#   DATASET=support METHOD=rsf scripts/run_cloud_comprehensive.sh

CONFIG_PATH="configs/benchmark/cloud_comprehensive_all_models_hpo.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATASET_ARG=()
METHOD_ARG=()
EXTRA_ARGS=()

while (($#)); do
  case "$1" in
    --dataset)
      DATASET_ARG=(--dataset "$2")
      shift 2
      ;;
    --method)
      METHOD_ARG=(--method "$2")
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${DATASET:-}" ]]; then
  DATASET_ARG=(--dataset "${DATASET}")
fi
if [[ -n "${METHOD:-}" ]]; then
  METHOD_ARG=(--method "${METHOD}")
fi

echo "[cloud-comprehensive] Launching benchmark with config: ${CONFIG_PATH}"
if [[ ${#DATASET_ARG[@]} -gt 0 ]]; then
  echo "[cloud-comprehensive] Dataset override: ${DATASET_ARG[1]}"
fi
if [[ ${#METHOD_ARG[@]} -gt 0 ]]; then
  echo "[cloud-comprehensive] Method override: ${METHOD_ARG[1]}"
fi

exec env PYTHONUNBUFFERED="$PYTHONUNBUFFERED" "$PYTHON_BIN" -m survarena.run_benchmark \
  --benchmark-config "$CONFIG_PATH" \
  "${DATASET_ARG[@]}" \
  "${METHOD_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
