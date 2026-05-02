#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON_BIN:-/home/lebedeffson/Code/venv_cuda/bin/python}
SEEDS=${SEEDS:-42,43,44,45,46}

# Replace configs below with dedicated ablation configs when prepared.
$PY scripts/run_multiseed_autonomous.py --config configs/config_sml2010_ea_minimal.yaml --seeds "$SEEDS" --tag-prefix ablation_full --inprocess --fast --fast-save-model

echo "Ablation scaffold finished (full EA baseline run)."

