#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/benchmark/smoke.yaml}"
WORK_DIR="${WORK_DIR:-results/summary/protocol_validation}"

echo "[protocol] Running dry-run config validation"
"$PYTHON_BIN" -m survarena.run_benchmark \
  --benchmark-config "$BENCHMARK_CONFIG" \
  --dry-run

echo "[protocol] Running focused benchmark execution"
"$PYTHON_BIN" -m survarena.run_benchmark \
  --benchmark-config "$BENCHMARK_CONFIG" \
  --dataset whas500 \
  --method coxph \
  --limit-seeds 1 \
  --output-dir "$WORK_DIR" \
  --max-retries 1

echo "[protocol] Verifying required artifacts"
test -f "$WORK_DIR"/experiment_manifest.json
test -f "$WORK_DIR"/coxph_fold_results.csv
test -f "$WORK_DIR"/coxph_leaderboard.csv
test -f "$WORK_DIR"/coxph_run_diagnostics.csv

echo "[protocol] Artifact validation passed: $WORK_DIR"
