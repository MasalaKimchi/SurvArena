#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/benchmark/manuscript_v1.yaml}"
WORK_DIR="${WORK_DIR:-results/protocol_validation}"
FOCUSED_BENCHMARK_CONFIG="$WORK_DIR/focused_benchmark.yaml"

echo "[protocol] Running dry-run config validation"
"$PYTHON_BIN" -m survarena.run_benchmark \
  --config "$BENCHMARK_CONFIG" \
  --dry-run

mkdir -p "$WORK_DIR"
"$PYTHON_BIN" - "$BENCHMARK_CONFIG" "$FOCUSED_BENCHMARK_CONFIG" <<'PY'
from pathlib import Path
import sys

import yaml

source = Path(sys.argv[1])
target = Path(sys.argv[2])
with source.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
cfg["benchmark_id"] = f"{cfg['benchmark_id']}_protocol_validation"
target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY

echo "[protocol] Running focused benchmark execution"
"$PYTHON_BIN" -m survarena.run_benchmark \
  --config "$FOCUSED_BENCHMARK_CONFIG" \
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
