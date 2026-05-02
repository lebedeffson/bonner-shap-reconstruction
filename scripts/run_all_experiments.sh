#!/usr/bin/env bash
set -euo pipefail

PY=${PYTHON_BIN:-/home/lebedeffson/Code/venv_cuda/bin/python}

# Data prep
$PY scripts/prepare_sml2010.py
$PY scripts/prepare_naval_propulsion.py
$PY scripts/download_energy_efficiency.py

# EA multiseed
$PY scripts/run_multiseed_autonomous.py --config configs/config_sml2010_ea_minimal.yaml --seeds 42,43,44,45,46,47,48,49,50,51 --tag-prefix sml_ea10 --inprocess --fast --fast-save-model
$PY scripts/run_multiseed_autonomous.py --config configs/config_energy_ea_minimal.yaml --seeds 42,43,44,45,46,47,48,49,50,51 --tag-prefix energy_ea10 --inprocess --fast --fast-save-model
$PY scripts/run_multiseed_autonomous.py --config configs/config_naval_ea_minimal.yaml --seeds 42,43,44,45,46,47,48,49,50,51 --tag-prefix naval_ea10 --inprocess --fast --fast-save-model

# Baselines
$PY scripts/run_tabular_baselines.py --config configs/config_sml2010_ea_minimal.yaml --tag sml2010_10seed
$PY scripts/run_tabular_baselines.py --config configs/config_energy_ea_minimal.yaml --tag energy_10seed
$PY scripts/run_tabular_baselines.py --config configs/config_naval_ea_minimal.yaml --tag naval_10seed

# Baseline faithfulness
$PY scripts/run_baseline_faithfulness.py --config configs/config_sml2010_ea_minimal.yaml --tag sml2010_10seed --k-list 1,2,3,4 --mask permute --random-trials 20
$PY scripts/run_baseline_faithfulness.py --config configs/config_energy_ea_minimal.yaml --tag energy_10seed --k-list 1,2,3,4 --mask permute --random-trials 20
$PY scripts/run_baseline_faithfulness.py --config configs/config_naval_ea_minimal.yaml --tag naval_10seed --k-list 1,2,3,4 --mask permute --random-trials 20

# Manifest
$PY scripts/generate_results_manifest.py

echo "Done."

