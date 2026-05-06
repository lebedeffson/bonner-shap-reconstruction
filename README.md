# Bonner Spectrum Reconstruction

ANFIS-based repository for neutron spectrum reconstruction from Bonner sphere channels (`Q1..Q10`) with full SHAP regularization at training time.

## Scope
This repository contains the SHAP-regularization line for Bonner/spectra experiments.

EAAR-focused experiments are maintained separately:
- https://github.com/lebedeffson/eaar-regularization

## Main entry points
- `train_vanilla_real_only.py` — baseline training without regularization.
- `train.py` — two-stage training with regularization.
- `src/models/shap_trainer_improved.py` — SHAP/importance regularization logic.
- `scripts/report_faithfulness_top_random_bottom.py` — deletion-faithfulness report (`top/random/bottom`).
- `scripts/practical_readiness_gate.py` — practical gate (quality + regularization strength + faithfulness/alignment).
- `scripts/cleanup_results_heavy.py` — cleanup heavy local artifacts in `results/`.

## SHAP mode in this repo
Default and recommended mode is full SHAP regularization:
- `shap_reg.shap_estimator: exact_shap`
- `shap_reg.strict_exact_shap: true`

## Quick start
```bash
source /home/lebedeffson/Code/venv_cuda/bin/activate
pip install -r requirements.txt
```

Baseline:
```bash
python train_vanilla_real_only.py --config configs/config_vanilla_r2_09.yaml --tag vanilla_run
```

Regularized run (full SHAP):
```bash
python train.py --config configs/config_shap_exact_accuracy_first.yaml --tag shap_exact_run
```

Faithfulness report:
```bash
PYTHONPATH=. python scripts/report_faithfulness_top_random_bottom.py \
  --summary results/training_summary_<timestamp>.json \
  --k-max 4 --random-trials 20 --masking permute
```

Importance-alignment report (SHAP target vs internal importance):
```bash
PYTHONPATH=. python scripts/report_importance_alignment.py \
  --ref results/feature_importance_shap_<timestamp>.csv \
  --cand results/feature_importance_<timestamp>.csv \
  --label-ref shap_target --label-cand internal_grad \
  --k 3,5
```

Practical readiness gate:
```bash
PYTHONPATH=. python scripts/practical_readiness_gate.py \
  --summary results/training_summary_<timestamp>.json \
  --faithfulness results/faithfulness_<timestamp>.json \
  --alignment results/importance_alignment_<timestamp>.json
```

Cleanup heavy local artifacts (dry-run):
```bash
python scripts/cleanup_results_heavy.py
```

Cleanup heavy local artifacts (apply):
```bash
python scripts/cleanup_results_heavy.py --apply
```

## Key configs
- `configs/config_vanilla_r2_09.yaml`
- `configs/config_integrated_shap.yaml`
- `configs/config_shap_only_full.yaml`
- `configs/config_shap_tikhonov_balanced_full.yaml`
- `configs/config_shap_exact_accuracy_first.yaml`

## Results artifacts
Generated into `results/`:
- `training_summary_*.json`
- `metrics_*.csv`
- `feature_importance*.csv`
- `faithfulness_*.md` / `faithfulness_*.json`

Latest sanity snapshot:
- `results/run_check_spectra_bonner_20260506.md`

## Repository layout
```text
configs/
src/
scripts/
results/
train.py
train_vanilla_real_only.py
```

## Git hygiene
Commit:
- code (`src/`, `scripts/`, `configs/`)
- lightweight reports (`results/*.md`, `results/*summary*.json`, `results/metrics_*.csv`)

Do not commit:
- `results/**/*.pt`
- `results/**/*.npy`
- `results/**/*.png`
- `results/**/*.pdf`
