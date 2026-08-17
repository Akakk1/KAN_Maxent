#!/usr/bin/env bash
# Copy minimal archival files from outputs/ → results/published/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DEST="${ROOT}/results/published"
mkdir -p "${DEST}"

copy_run() {
  local run="$1"; shift
  local src="${ROOT}/outputs/${run}"
  local dst="${DEST}/${run}"
  if [[ ! -d "${src}" ]]; then
    echo "SKIP missing: ${src}"
    return 0
  fi
  mkdir -p "${dst}"
  for f in "$@"; do
    if [[ -e "${src}/${f}" ]]; then
      mkdir -p "${dst}/$(dirname "${f}")"
      cp -a "${src}/${f}" "${dst}/${f}"
      echo "  + ${run}/${f}"
    else
      echo "  ? missing ${run}/${f}"
    fi
  done
}

echo "Snapshot → ${DEST}"

copy_run ginkgo_cv_20260714_phase1 \
  SUMMARY.md manifest.json metrics.csv

copy_run nceas_full_20260714_phase3 \
  SUMMARY.md manifest.json metrics_all.csv paired_delta_auc.csv \
  auc_by_region_model.csv valavi_maxnet_comparison_all.csv tgb_vs_random50k.csv

copy_run deepkan_fair_six_3seed_20260715 \
  SUMMARY.md manifest.json metrics_all.csv \
  paired_delta_auc_all.csv paired_delta_auc_species_mean.csv

copy_run deepkan_deep3_can20_fair_20260715 \
  SUMMARY.md manifest.json metrics.csv paired_delta_auc_all.csv \
  paired_delta_auc_deep2.csv paired_delta_auc_deep3.csv

copy_run phase5_curves_20260715 \
  SUMMARY.md nceas_kan_vs_gam_agreement_all.csv \
  figures/FigG1_ginkgo_curves.png figures/FigN1_nceas_sample_curves.png

echo "Done. Optional: sha256sum → results/published/SHA256SUMS.txt"
