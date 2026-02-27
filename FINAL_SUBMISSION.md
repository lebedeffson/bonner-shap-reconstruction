# Final Submission Summary

This repository is ready for submission with the ANFIS + SHAP + Tikhonov pipeline.
Below are the exact artifacts, metrics, and reproducible commands.

## Configuration
- Main config: `configs/config_integrated_shap.yaml`
- Key settings:
  - `model.num_rules: 50`
  - `shap_reg.enabled: true`
  - `shap_reg.tikhonov.enabled: true`
  - `shap_reg.tikhonov.lambda: 0.001`
  - `shap_reg.tikhonov.order: 2`

## Final Model & Metrics
- Model checkpoint: `results/anfis_model_state_20260227_233124_tikhonov_final.pt`
- Training summary (metrics, bands, diagnostics): `results/training_summary_20260227_233124_tikhonov_final.json`
- Test metrics (from summary):
  - MSE: 0.010615
  - RMSE: 0.103029
  - MAE: 0.048607
  - R2 (weighted): 0.836842
  - R2 (mean): 0.556031

## Result Figures
- Error plots: `results/errors_energy_20260227_233124_tikhonov_final.png`, `results/errors_hist_20260227_233124_tikhonov_final.png`
- Scatter: `results/scatter_20260227_233124_tikhonov_final.png`
- Spectra samples: `results/spectra_samples_20260227_233124_tikhonov_final.png`
- Mean spectra: `results/spectra_mean_20260227_233124_tikhonov_final.png`
- SHAP feature importances:
  - `results/feature_importance_shap_20260227_233124_tikhonov_final.csv`
  - `results/feature_importance_shap_20260227_233124_tikhonov_final.png`

## Inference Outputs
- Inference outputs for one example:
  - `results/inference_final/predictions_20260227_233144.csv`
  - `results/inference_final/spectrum_20260227_233144_idx0.png`

## Uncertainty / Monte Carlo
- Sweep 0.5%–10% (step 0.5%), N=1000:
  - Summary CSV: `results/uncertainty_range/uncertainty_summary_20260227_234234.csv`
  - Summary plot: `results/uncertainty_range/uncertainty_summary_20260227_234234.png`
  - Per-error plots: `results/uncertainty_range/error_*/uncertainty_20260227_234234.png`

## Article Math
- Full method description: `article_math.tex`

## Repro Commands
```bash
~/Code/venv/bin/python train.py --config configs/config_integrated_shap.yaml --tag tikhonov_final
```
```bash
~/Code/venv/bin/python infer.py \
  --config configs/config_integrated_shap.yaml \
  --model results/anfis_model_state_20260227_233124_tikhonov_final.pt \
  --input "0.016725994,0.028362745,0.063373974,0.14926387,0.1718632,0.17809387,0.15080825,0.11797526,0.077408414,0.046124427" \
  --output-dir results/inference_final
```
```bash
~/Code/venv/bin/python uncertainty_analysis.py \
  --config configs/config_integrated_shap.yaml \
  --model results/anfis_model_state_20260227_233124_tikhonov_final.pt \
  --input "0.016725994,0.028362745,0.063373974,0.14926387,0.1718632,0.17809387,0.15080825,0.11797526,0.077408414,0.046124427" \
  --n-samples 1000 \
  --error-percent-range 0.5:10:0.5 \
  --plot-each \
  --output-dir results/uncertainty_range
```
