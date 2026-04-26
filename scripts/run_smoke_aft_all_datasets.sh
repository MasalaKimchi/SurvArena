#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/benchmark/smoke_aft.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/smoke_aft}"
LIMIT_SEEDS="${LIMIT_SEEDS:-1}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/survarena-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/survarena-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

DATASETS=(
  support
  metabric
  aids
  gbsg2
  flchain
  whas500
)

timestamp="$(date +"%Y%m%d_%H%M%S")"
output_dir="${OUTPUT_ROOT}/${timestamp}"

echo "benchmark_config=${BENCHMARK_CONFIG}"
echo "output_dir=${output_dir}"
echo "limit_seeds=${LIMIT_SEEDS}"
echo "datasets=${#DATASETS[@]}"

for dataset in "${DATASETS[@]}"; do
  echo "---"
  echo "dataset=${dataset}"
  python -m survarena.run_benchmark \
    --benchmark-config "${BENCHMARK_CONFIG}" \
    --dataset "${dataset}" \
    --limit-seeds "${LIMIT_SEEDS}" \
    --output-dir "${output_dir}/${dataset}"
done

echo "---"
echo "done=1"
echo "output_dir=${output_dir}"
