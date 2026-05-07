# Bonner Spectrum Reconstruction with SHAP Regularization

This repository contains a reproducible pipeline for neutron spectrum reconstruction from Bonner sphere detector channels `Q1..Q10`. The main line of the project is based on ANFIS models and training-time SHAP regularization. The purpose of the repository is to preserve predictive quality while improving the agreement between the model's internal feature importance and exact SHAP-based attribution targets.

Bonner spectrum reconstruction is an ill-posed inverse problem. The input vector contains a small number of detector responses, while the output is a reconstructed neutron spectrum. In this setting, prediction accuracy is important, but it is not the only requirement. Each input channel has physical meaning, so the model should also provide a stable and interpretable structure of feature importance. This repository focuses on that requirement through SHAP-aware regularization during training.

EAAR experiments are maintained in a separate repository at https://github.com/lebedeffson/eaar-regularization. The present repository is dedicated to the SHAP-regularized Bonner spectrum reconstruction line.

## Latest Result Snapshot (2026-05-07)

The current repository state is finalized around a short multi-seed validation cycle and a method-refinement cycle.

- Multi-seed short benchmark (`seeds 42/43/44`):
  - SHAP `AUC gap mean = 0.011759`
  - SHAP wins vs vanilla by `AUC gap`: `3/3`
- Rank/early-stop refinement (`seeds 42/43`, fast PSO smoke):
  - SHAP `AUC gap mean = 0.020822`
  - Relative gain vs previous short baseline (`seeds 42/43`): `+63.90%`

Compact release artifacts:
- `results/release_final_pack_20260507.md`
- `results/results_manifest.json`

## Method

The model is trained with a reconstruction loss and an additional SHAP-aware regularization term. The total objective has the following form.

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{task}}
+
\gamma \mathcal{L}_{\mathrm{shap}}
+
\lambda \mathcal{L}_{\mathrm{tik}} .
```

Here, `L_task` denotes the main spectrum reconstruction loss. The term `L_tik` denotes the Tikhonov smoothness penalty over the reconstructed spectrum bins. The term `L_shap` denotes the SHAP-aware attribution regularization block. The coefficient `gamma` controls the strength of SHAP regularization, while `lambda` controls the strength of Tikhonov smoothing.

The improved SHAP regularization block combines several attribution-oriented constraints.

```math
\mathcal{L}_{\mathrm{shap}}
=
w_c \mathcal{L}_{\mathrm{consistency}}
+
w_s \mathcal{L}_{\mathrm{sparsity}}
+
w_f \mathcal{L}_{\mathrm{faithfulness}}
+
w_{st} \mathcal{L}_{\mathrm{stability}} .
```

The consistency component aligns internal gradient-based importance with the exact SHAP target. The sparsity component encourages a compact attribution structure through entropy and Gini-based shaping. The faithfulness component supports first-order consistency between attribution and output change. The stability component controls the variance of importance estimates across samples.

In release mode, the pipeline uses exact SHAP targets. The corresponding configuration is expected to contain the following settings.

```yaml
shap_estimator: exact_shap
strict_exact_shap: true
```

SHAP target semantics are explicit and configurable:

```yaml
shap_value_function: mean_output     # mean_output | sum_output | l2_output
shap_baseline_mode: feature_mean     # feature_mean | median | zero
shap_baseline_clip_nonnegative: true
```

## Repository Structure

The repository is organized around training, regularization, evaluation, and practical validation. The `src/` directory contains the core implementation, model definitions, training logic, and SHAP regularization modules. The `configs/` directory stores experiment configurations. The `scripts/` directory contains reporting, readiness, and cleanup utilities. The `tests/` directory is used for validation checks. The `results/` directory is intended for local experiment outputs and should not be used for committing heavy artifacts.

```text
configs/
scripts/
src/
tests/
results/
train.py
train_vanilla_real_only.py
```

The baseline training entry point is `train_vanilla_real_only.py`. The main two-stage pipeline is implemented in `train.py`, where vanilla pretraining is followed by SHAP-regularized fine-tuning. The core regularization logic is located in `src/models/shap_trainer_improved.py`.

## Quick Start

Activate the environment and install the required dependencies.

```bash
source /home/lebedeffson/Code/venv_cuda/bin/activate
pip install -r requirements.txt
```

After the environment is ready, the baseline model can be trained with the vanilla entry point.

```bash
python train_vanilla_real_only.py \
  --config configs/config_vanilla_r2_09.yaml \
  --tag vanilla_run
```

The SHAP-regularized training pipeline is launched through `train.py`.

```bash
python train.py \
  --config configs/config_shap_exact_alignment_push_quickcheck.yaml \
  --tag shap_run
```

This run performs vanilla pretraining and then applies SHAP-regularized fine-tuning. The regularized stage is designed to preserve reconstruction quality while improving the alignment between internal importance and exact SHAP targets.

## Faithfulness Evaluation

The repository includes a deletion-based faithfulness report. The evaluation compares how the model behaves when top-ranked, randomly selected, and bottom-ranked input channels are masked. A faithful importance ranking should assign higher impact to top-ranked channels than to bottom-ranked channels.

The report is generated with the following command.

```bash
python scripts/report_faithfulness_top_random_bottom.py \
  --summary results/training_summary_<timestamp>.json \
  --k-max 4 \
  --random-trials 20 \
  --masking permute
```

The resulting report is used to check whether the attribution ranking is functionally meaningful. The main diagnostic values are `AUC gap`, `top/random`, top-feature deletion effect, and bottom-feature deletion effect.

## SHAP and Internal Importance Alignment

The alignment report compares exact SHAP targets with internal gradient-based importance. This check is needed to verify that the regularized model forms an internal attribution structure that is consistent with the SHAP target used during training.

```bash
python scripts/report_importance_alignment.py \
  --ref results/feature_importance_shap_<timestamp>.csv \
  --cand results/feature_importance_internal_<timestamp>.csv \
  --label-ref shap_target \
  --label-cand internal_grad \
  --k 3,5
```

The reference file is the exact SHAP importance report. The candidate file is the internal model importance report. The output of this script helps determine whether the learned internal ranking follows the target attribution structure.

## Practical Readiness Gate

The practical readiness gate combines the main validation signals into a single release decision. It checks predictive quality, regularization behavior, deletion faithfulness, and SHAP/internal alignment.

```bash
python scripts/practical_readiness_gate.py \
  --summary results/training_summary_<timestamp>.json \
  --faithfulness results/faithfulness_<timestamp>.json \
  --alignment results/importance_alignment_<timestamp>.json
```

A run is considered release-ready only when the gate returns `PASS`. This means that the reconstruction quality is preserved according to the configured `r2_weighted` threshold, the regularization signal is non-negligible, the deletion faithfulness metrics are valid, and the internal importance is sufficiently aligned with the exact SHAP target.

## Recommended Workflow

A typical experiment starts with a vanilla baseline run. This baseline is used as a reference for reconstruction quality and later comparison with the regularized model.

```bash
python train_vanilla_real_only.py \
  --config configs/config_vanilla_r2_09.yaml \
  --tag vanilla_run
```

After the baseline is available, the SHAP-regularized pipeline is launched.

```bash
python train.py \
  --config configs/config_shap_exact_alignment_push_quickcheck.yaml \
  --tag shap_run
```

Once training is complete, the deletion faithfulness report is generated.

```bash
python scripts/report_faithfulness_top_random_bottom.py \
  --summary results/training_summary_<timestamp>.json \
  --k-max 4 \
  --random-trials 20 \
  --masking permute
```

The next step is to compare the exact SHAP target with the internal importance produced by the model.

```bash
python scripts/report_importance_alignment.py \
  --ref results/feature_importance_shap_<timestamp>.csv \
  --cand results/feature_importance_internal_<timestamp>.csv \
  --label-ref shap_target \
  --label-cand internal_grad \
  --k 3,5
```

The final step is the practical readiness gate.

```bash
python scripts/practical_readiness_gate.py \
  --summary results/training_summary_<timestamp>.json \
  --faithfulness results/faithfulness_<timestamp>.json \
  --alignment results/importance_alignment_<timestamp>.json
```

This sequence gives a complete validation path from baseline training to release readiness.

## Reviewer-Oriented Benchmarks

Classical/modern ML baselines on the same split:

```bash
python scripts/benchmark_ml_baselines.py \
  --config configs/config_integrated_shap.yaml \
  --seeds 42,43,44 \
  --output-json results/ml_baselines_benchmark_3seed_20260507.json \
  --output-md results/ml_baselines_benchmark_3seed_20260507.md
```

Unfolding-style proxy baselines (Tikhonov/NNLS):

```bash
python scripts/benchmark_unfolding_proxies.py \
  --config configs/config_integrated_shap.yaml \
  --seeds 42,43,44 \
  --lambdas 1e-5,1e-4,1e-3,1e-2,1e-1
```

SHAP semantics sweep:

```bash
python scripts/sweep_shap_semantics.py \
  --base-config configs/config_integrated_shap.yaml \
  --baseline-modes feature_mean,median,zero \
  --value-functions mean_output,sum_output,l2_output
```

`training_summary_*.json` now stores reproducibility metadata:
- `config_sha256`
- `effective_config_sha256`
- `split_hash`
- `diagnostics.shap_spec`
- `diagnostics.shap_compute`

## Artifact Policy

The repository should contain source code, configuration files, scripts, tests, and lightweight experiment summaries. Large local artifacts should remain outside version control. This includes model checkpoints, NumPy arrays, generated figures, and exported PDF reports from the `results/` directory.

The cleanup utility can be used to inspect heavy local artifacts before removal.

```bash
python scripts/cleanup_results_heavy.py
```

Cleanup can be applied with the following command.

```bash
python scripts/cleanup_results_heavy.py --apply
```

## Notes

This repository is the release-oriented SHAP regularization line for Bonner spectrum reconstruction. It is intended for controlled experiments where predictive quality, attribution faithfulness, and practical readiness are evaluated together. For masking-error-driven attribution regularization and EAAR experiments, use the separate EAAR repository at https://github.com/lebedeffson/eaar-regularization.
