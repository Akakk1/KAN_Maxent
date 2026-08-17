#!/usr/bin/env bash
# Level-1 smoke: tests + tiny Deep ablation (no full NCEAS).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

echo "== pytest =="
python -m pytest tests/ -q

OUT="${ROOT}/outputs/smoke_deep3_can3_$(date -u +%Y%m%dT%H%M%SZ)"
echo "== Deep smoke CAN 3 species → ${OUT} =="
python benchmarks/deepkan_ablation.py \
  --regions CAN \
  --species can02,can13,can14 \
  --seeds 0 \
  --models additive,deep2,deep3 \
  --hidden-width 4 \
  --lbfgs-steps 12 \
  --outdir "${OUT}"

echo "OK smoke complete: ${OUT}"
