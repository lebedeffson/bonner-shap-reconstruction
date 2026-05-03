#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON_BIN:-/home/lebedeffson/Code/venv_cuda/bin/python}
SEEDS=${SEEDS:-42,43,44,45,46}
BASE_CONFIG=${BASE_CONFIG:-configs/config_sml2010_ea_minimal.yaml}
TAG_PREFIX=${TAG_PREFIX:-sml_ablation}

$PY scripts/run_ea_ablation.py \
  --base-config "$BASE_CONFIG" \
  --seeds "$SEEDS" \
  --tag-prefix "$TAG_PREFIX" \
  --inprocess \
  --fast \
  --fast-save-model \
  --with-explainability \
  --mask permute \
  --eval-importance final \
  --k-list 1,2,3,4

MANIFEST="results/ablation/ablation_manifest_$(basename "${BASE_CONFIG%.*}")_${TAG_PREFIX}.json"
$PY scripts/report_ea_ablation.py --manifest "$MANIFEST"

echo "Ablation run + report finished."
