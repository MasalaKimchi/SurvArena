#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK_CONFIG="${BENCHMARK_CONFIG:-configs/benchmark/manuscript_v1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/manuscript_dataset_model}"
MAX_RETRIES="${MAX_RETRIES:-1}"
LIMIT_SEEDS="${LIMIT_SEEDS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ ! -f "$BENCHMARK_CONFIG" ]]; then
  echo "[error] Benchmark config not found: $BENCHMARK_CONFIG" >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

readarray -t DATASETS < <("$PYTHON_BIN" - <<'PY' "$BENCHMARK_CONFIG"
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in cfg.get("datasets", []):
    print(item)
PY
)

readarray -t METHODS < <("$PYTHON_BIN" - <<'PY' "$BENCHMARK_CONFIG"
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in cfg.get("methods", []):
    print(item)
PY
)

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  echo "[error] No datasets found in $BENCHMARK_CONFIG" >&2
  exit 1
fi

if [[ "${#METHODS[@]}" -eq 0 ]]; then
  echo "[error] No methods found in $BENCHMARK_CONFIG" >&2
  exit 1
fi

printf '[plan] datasets=%s methods=%s config=%s\n' "${#DATASETS[@]}" "${#METHODS[@]}" "$BENCHMARK_CONFIG"

for dataset in "${DATASETS[@]}"; do
  dataset_output_dir="$OUTPUT_ROOT/$dataset"
  mkdir -p "$dataset_output_dir"

  for method in "${METHODS[@]}"; do
    run_tag="${dataset}__${method}"
    run_output_dir="$dataset_output_dir/$method"
    log_path="$dataset_output_dir/${method}.log"

    echo "[run] $run_tag"
    start_epoch="$(date +%s)"

    set +e
    "$PYTHON_BIN" -m survarena.run_benchmark \
      --config "$BENCHMARK_CONFIG" \
      --dataset "$dataset" \
      --method "$method" \
      --output-dir "$run_output_dir" \
      --resume \
      --max-retries "$MAX_RETRIES" \
      ${LIMIT_SEEDS:+--limit-seeds "$LIMIT_SEEDS"} \
      $EXTRA_ARGS \
      > "$log_path" 2>&1
    status=$?
    set -e

    end_epoch="$(date +%s)"
    elapsed="$((end_epoch - start_epoch))"

    if [[ "$status" -eq 0 ]]; then
      echo "[ok] $run_tag elapsed=${elapsed}s log=$log_path"
    else
      echo "[fail] $run_tag elapsed=${elapsed}s exit_code=$status log=$log_path" >&2
    fi

    printf '%s,%s,%s,%s,%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$dataset" \
      "$method" \
      "$status" \
      "$elapsed" \
      >> "$OUTPUT_ROOT/run_status.csv"
  done
done

echo "[done] matrix run complete. status table: $OUTPUT_ROOT/run_status.csv"
