#!/usr/bin/env bash
set +e

cd /Users/justin/Documents/SurvArena || exit 1

echo "[started] $(date)"
echo "[clinical] manuscript_autogluon_foundation_v1"
python -m survarena.run_benchmark \
  --config configs/benchmark/manuscript_autogluon_foundation_v1.yaml \
  --output-dir results/manuscript_autogluon_foundation_v1_full_20260527 \
  --regenerate-splits
clinical_status=$?
echo "[clinical_status] ${clinical_status}"

echo "[genomics] manuscript_genomics_autogluon_foundation_v1"
python -m survarena.run_benchmark \
  --config configs/benchmark/manuscript_genomics_autogluon_foundation_v1.yaml \
  --output-dir results/manuscript_genomics_autogluon_foundation_v1_full_20260527 \
  --regenerate-splits
genomics_status=$?
echo "[genomics_status] ${genomics_status}"
echo "[finished] $(date)"

if [[ "${clinical_status}" -ne 0 ]]; then
  exit "${clinical_status}"
fi
exit "${genomics_status}"
