# Bonner SHAP Final Pack (2026-05-07)

Submission-ready compact pack for the SHAP regularization line.

## 1) Core pipeline status

- Two-stage ANFIS pipeline is stable: vanilla pretrain + SHAP fine-tuning.
- Gap-aware checkpoint selection is active and validated on multi-seed short runs.
- Selection quality metric is fixed to `r2_var_weighted` (no `r2_mean` mismatch).

## 2) Multi-seed short benchmark (official)

Source: `results/gapaware_ms120_3seed_summary_20260507.md`

- SHAP AUC gap mean: `0.011759` (wins vs vanilla: `3/3`)
- Internal AUC gap mean: `0.001677` (wins vs vanilla: `3/3`)
- Top/random SHAP mean: `2.288389`
- R2_w SHAP mean: `0.787623`

## 3) Method refinement cycle (rank-margin + early-stop)

Source: `results/gapaware_rankes_fastpso_2seed_summary_20260507.md`

Added:
- consistency rank-margin term
- delayed regularization start
- early-stop on selection score

Observed:
- SHAP AUC gap mean improved to `0.020822` on seeds `42/43`
- Relative improvement vs previous seeds `42/43` short baseline: `+63.90%`
- Internal gap still near zero (`~0.000024`)

## 4) Honest boundary

- SHAP-target faithfulness is clearly improved.
- Internal-gradient faithfulness is not yet strong.
- For article claims: keep strong claims on SHAP-target mode, keep internal mode conservative.

## 5) Key reproducibility artifacts

- Config (main): `configs/config_integrated_shap.yaml`
- Configs (short multiseed): `configs/multiseed_gapaware_v2_short/`
- Training summaries:
  - `results/training_summary_20260507_112256_ms120_s42.json`
  - `results/training_summary_20260507_113428_ms120_s43.json`
  - `results/training_summary_20260507_114617_ms120_s44.json`
  - `results/training_summary_20260507_121253_dev_rank_es_fastpso.json`
  - `results/training_summary_20260507_121747_dev_rank_es_fastpso_s43.json`
- Faithfulness reports:
  - `results/faithfulness_ms120_s42_shap.json`
  - `results/faithfulness_ms120_s43_shap.json`
  - `results/faithfulness_ms120_s44_shap.json`
  - `results/faithfulness_vanilla_ms120_s42.json`
  - `results/faithfulness_vanilla_ms120_s43.json`
  - `results/faithfulness_vanilla_ms120_s44.json`
  - `results/faithfulness_dev_rank_es_fastpso_shap.json`
  - `results/faithfulness_dev_rank_es_fastpso_s43_shap.json`
