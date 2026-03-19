# Final Submission Summary

This repository is ready for submission with the main `ANFIS + SHAP + Tikhonov` pipeline.
Below are the current artifacts, comparison runs, figures, and reproducible commands.

## Configuration
- Main config: `configs/config_integrated_shap.yaml`
- Main experimental tag: `tikhonov_stronger_20260319`
- Key settings:
  - `model.num_rules: 50`
  - `shap_reg.enabled: true`
  - `shap_reg.use_true_shap: true`
  - `shap_reg.use_adaptive_weights: true`
  - `shap_reg.gamma_start: 0.02`
  - `shap_reg.gamma_end: 0.10`
  - `shap_reg.target_shap_ratio: 0.40`
  - `shap_reg.tikhonov.enabled: true`
  - `shap_reg.tikhonov.lambda: 0.002`
  - `shap_reg.tikhonov.order: 2`

## Main Model & Metrics
- Model checkpoint:
  - `results/anfis_model_state_20260319_190809_tikhonov_stronger_20260319.pt`
- Training summary:
  - `results/training_summary_20260319_190809_tikhonov_stronger_20260319.json`
- SHAP history:
  - `results/shap_history_20260319_190809_tikhonov_stronger_20260319.json`
- Test metrics:
  - MSE: `0.01060296`
  - RMSE: `0.10297068`
  - MAE: `0.04890357`
  - R2 (weighted): `0.83702629`
  - R2 (mean): `0.55409377`

## Vanilla Baseline
- Full baseline summary:
  - `results/vanilla_full_20260319/training_summary_20260319_202741_vanilla_full_20260319.json`
- Baseline metrics:
  - MSE: `0.01132709`
  - RMSE: `0.10642880`
  - MAE: `0.04652143`
  - R2 (weighted): `0.82589603`
  - R2 (mean): `0.54969990`

## Full Method Comparison
Full-size comparison on the same `75`-sample held-out real test split:

- `Vanilla`:
  - `results/vanilla_full_20260319/training_summary_20260319_202741_vanilla_full_20260319.json`
- `Tikhonov-only`:
  - `results/training_summary_20260319_210248_tikhonov_only_full_20260319.json`
- `SHAP-only`:
  - `results/training_summary_20260319_210638_shap_only_full_20260319.json`
- `SHAP+Tikhonov`:
  - `results/training_summary_20260319_190809_tikhonov_stronger_20260319.json`

Key summary:
- Best by `MSE`, `RMSE`, `R2 (weighted)`, `R2 (mean)`: `SHAP-only`
- Best by `MAE` and by Monte Carlo input-noise robustness: `Vanilla`
- Best Monte Carlo robustness among regularized variants: `Tikhonov-only`
- `SHAP+Tikhonov` remains the main compromise run: near-top accuracy with both interpretability and a physical smoothness prior

Method-comparison artifacts:
- `results/method_comparison_20260319/method_comparison_summary.csv`
- `results/method_comparison_20260319/method_comparison_summary.md`
- `results/method_comparison_20260319/fig_method_tradeoff.png`
- `results/method_comparison_20260319/fig_band_quality.png`
- `results/method_comparison_20260319/fig_uncertainty_methods.png`

## Final Figure Pack
Main publication figures are stored in:
- `results/final_figures_20260319/manifest.md`
- `results/final_figures_20260319/fig_01_metrics_comparison.png`
- `results/final_figures_20260319/fig_02_mean_spectra_comparison.png`
- `results/final_figures_20260319/fig_03_representative_spectrum.png`
- `results/final_figures_20260319/fig_05_shap_importance.png`
- `results/final_figures_20260319/fig_06_uncertainty_monte_carlo.png`

## Uncertainty / Monte Carlo
Full uncertainty sweep for `0.5, 1, 2, 5, 10%`, `N=1000`:
- `results/uncertainty_article_20260319/uncertainty_summary_20260319_202543.csv`
- `results/uncertainty_article_20260319/uncertainty_summary_20260319_202543.png`
- `results/uncertainty_article_20260319/error_0.5/`
- `results/uncertainty_article_20260319/error_1/`
- `results/uncertainty_article_20260319/error_2/`
- `results/uncertainty_article_20260319/error_5/`
- `results/uncertainty_article_20260319/error_10/`

Key uncertainty summary:
- `0.5%`: `mean_std=5.68e-05`, `max_std=3.59e-04`, `cv=0.00293`
- `1.0%`: `mean_std=1.13e-04`, `max_std=7.16e-04`, `cv=0.00585`
- `2.0%`: `mean_std=2.25e-04`, `max_std=1.42e-03`, `cv=0.01162`
- `5.0%`: `mean_std=5.38e-04`, `max_std=3.35e-03`, `cv=0.02776`
- `10.0%`: `mean_std=9.60e-04`, `max_std=5.74e-03`, `cv=0.04969`

Cross-method uncertainty on the `75` real-test inputs (`N=300` per level):
- `results/uncertainty_compare_vanilla_20260319/`
- `results/uncertainty_compare_tikhonov_only_20260319/`
- `results/uncertainty_compare_shap_only_20260319/`
- `results/uncertainty_compare_shap_tikhonov_20260319/`

Representative single-input Tikhonov-only Monte Carlo:
- `results/uncertainty_single_tikhonov_only_20260319/uncertainty_summary_20260319_211433.png`
- `results/uncertainty_single_tikhonov_only_20260319/error_10/uncertainty_20260319_211433.png`

## Article Math
- Main article body: `article_math.tex`
- Build wrapper: `article_math_full.tex`
- Compiled PDF: `article_math_full.pdf`
- Full implementation-aligned derivation: `math.tex`
- Compiled technical PDF: `math.pdf`

Current article status:
- includes 4-method comparison (`Vanilla`, `Tikhonov-only`, `SHAP-only`, `SHAP+Tikhonov`)
- includes cross-method Monte Carlo comparison
- explicitly documents that `SHAP` is strongest on accuracy, `Tikhonov` is strongest on robustness within the regularized family, and the combined model is a balanced compromise rather than a universal winner
- includes mathematically motivated next-step improvements: log-energy-aware Tikhonov, band-aware SHAP, and nonnegativity penalty

## Tests
- Full test suite:
  - `source /home/lebedeffson/Code/venv/bin/activate && pytest -q`
- Current result:
  - `76 passed, 13 subtests passed`

## Repro Commands
Main training run:
```bash
source /home/lebedeffson/Code/venv/bin/activate
python train.py --config configs/config_integrated_shap.yaml --tag tikhonov_stronger_20260319
```

Vanilla baseline:
```bash
source /home/lebedeffson/Code/venv/bin/activate
python train_vanilla_baseline.py --config configs/config_integrated_shap.yaml --tag vanilla_full_20260319
```

Inference:
```bash
source /home/lebedeffson/Code/venv/bin/activate
python infer.py \
  --config configs/config_integrated_shap.yaml \
  --model results/anfis_model_state_20260319_190809_tikhonov_stronger_20260319.pt \
  --input "0.016725994,0.028362745,0.063373974,0.14926387,0.1718632,0.17809387,0.15080825,0.11797526,0.077408414,0.046124427" \
  --output-dir results/inference_final
```

Monte Carlo uncertainty:
```bash
source /home/lebedeffson/Code/venv/bin/activate
python uncertainty_analysis.py \
  --config configs/config_integrated_shap.yaml \
  --model results/anfis_model_state_20260319_190809_tikhonov_stronger_20260319.pt \
  --input "0.016725994,0.028362745,0.063373974,0.14926387,0.1718632,0.17809387,0.15080825,0.11797526,0.077408414,0.046124427" \
  --n-samples 1000 \
  --error-percent-range 0.5,1,2,5,10 \
  --plot-each \
  --output-dir results/uncertainty_article_20260319
```

Publication figures:
```bash
source /home/lebedeffson/Code/venv/bin/activate
python prepare_publication_figures.py
```
