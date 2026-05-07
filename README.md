# Bonner Spectrum Reconstruction with SHAP Regularization

Official repository for ANFIS-based neutron spectrum reconstruction from Bonner sphere channels (`Q1..Q10`) with training-time SHAP regularization.

## 1) Scope

This repository is focused on the **SHAP-regularized Bonner/spectra line**.

EAAR experiments are maintained separately:
- https://github.com/lebedeffson/eaar-regularization

## 2) Method (release definition)

We train a model \(f_\theta\) with a task loss and a SHAP-aware regularizer:

\[
\mathcal{L}_{total}
=
\mathcal{L}_{task}
 \gamma \,\mathcal{L}_{shap}
 \lambda \,\mathcal{L}_{tik}
\]

where:
- \(\mathcal{L}_{task}\): spectrum regression loss,
- \(\mathcal{L}_{tik}\): Tikhonov smoothness penalty over output bins,
- \(\mathcal{L}_{shap}\): weighted SHAP regularization block.

Improved SHAP block:

\[
\mathcal{L}_{shap}
=
w_c \mathcal{L}_{consistency}
 w_s \mathcal{L}_{sparsity}
 w_f \mathcal{L}_{faithfulness}
 w_{st} \mathcal{L}_{stability}
\]

with:
- **consistency**: alignment of internal gradient importance with exact SHAP target,
- **sparsity**: compactness/entropy-gini shaping,
- **faithfulness**: first-order output-change consistency,
- **stability**: variance control across samples.

In release mode:
- `shap_estimator: exact_shap`
- `strict_exact_shap: true`

## 3) Main entry points

- `train_vanilla_real_only.py` — baseline training.
- `train.py` — two-stage pipeline (vanilla pretrain + SHAP-regularized fine-tune).
- `src/models/shap_trainer_improved.py` — core regularization logic.
- `scripts/report_faithfulness_top_random_bottom.py` — deletion faithfulness report.
- `scripts/report_importance_alignment.py` — SHAP/internal alignment report.
- `scripts/practical_readiness_gate.py` — practical PASS/FAIL gate.
- `scripts/cleanup_results_heavy.py` — local cleanup utility.

## 4) Quick start

```bash
source /home/lebedeffson/Code/venv_cuda/bin/activate
pip install -r requirements.txt
```

Baseline:
```bash
python train_vanilla_real_only.py \
  --config configs/config_vanilla_r2_09.yaml \
  --tag vanilla_run
```

Regularized run:
```bash
python train.py \
  --config configs/config_shap_exact_alignment_push_quickcheck.yaml \
  --tag shap_run
```

Faithfulness:
```bash
PYTHONPATH=. python scripts/report_faithfulness_top_random_bottom.py \
  --summary results/training_summary_<timestamp>.json \
  --k-max 4 --random-trials 20 --masking permute
```

Alignment:
```bash
PYTHONPATH=. python scripts/report_importance_alignment.py \
  --ref results/feature_importance_shap_<timestamp>.csv \
  --cand results/feature_importance_internal_<timestamp>.csv \
  --label-ref shap_target --label-cand internal_grad \
  --k 3,5
```

Practical gate:
```bash
PYTHONPATH=. python scripts/practical_readiness_gate.py \
  --summary results/training_summary_<timestamp>.json \
  --faithfulness results/faithfulness_<timestamp>.json \
  --alignment results/importance_alignment_<timestamp>.json
```

## 5) Release readiness criteria

A run is release-ready if practical gate is `PASS`:
- quality preserved (`r2_weighted` threshold),
- regularization contribution is non-negligible,
- faithfulness (`AUC gap`, `top/random`) is valid,
- SHAP/internal alignment is above threshold.

## 6) Repository layout

```text
configs/
scripts/
src/
tests/
results/
train.py
train_vanilla_real_only.py
```

## 7) Artifact policy

Commit:
- source code (`src/`, `scripts/`, `configs/`),
- lightweight summaries/reports (`.md`, `.json`, metric `.csv`).

Do not commit:
- `results/**/*.pt`
- `results/**/*.npy`
- `results/**/*.png`
- `results/**/*.pdf`

Use local cleanup:
```bash
python scripts/cleanup_results_heavy.py
python scripts/cleanup_results_heavy.py --apply
```
