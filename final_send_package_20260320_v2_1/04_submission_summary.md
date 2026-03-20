# Final Submission Summary

This repository is ready for submission with the current main `ANFIS + SHAP + log-energy-aware Tikhonov + hybrid nonnegativity` V2.1 pipeline.
Below are the current artifacts, comparison runs, figures, and reproducible commands.

## Configuration
- Main config: `configs/config_integrated_shap.yaml`
- Mirrored V2 config: `configs/config_integrated_shap_v2.yaml`
- Exact run config used for the current main result: `configs/config_integrated_shap_v2_1_light_candidate.yaml`
- Main experimental tag: `v2_1_light_nonneg_20260320`
- Key settings:
  - `model.num_rules: 50`
  - `shap_reg.enabled: true`
  - `shap_reg.use_true_shap: true`
  - `shap_reg.use_adaptive_weights: true`
  - `shap_reg.gamma_start: 0.02`
  - `shap_reg.gamma_end: 0.099`
  - `shap_reg.target_shap_ratio: 0.3895`
  - `shap_reg.tikhonov.enabled: true`
  - `shap_reg.tikhonov.lambda: 0.001`
  - `shap_reg.tikhonov.order: 2`
  - `shap_reg.tikhonov.energy_aware: true`
  - `shap_reg.scalarization.mode: band_weighted`
  - `shap_reg.scalarization.band_weights: [1/3, 1/3, 1/3]`
  - `shap_reg.nonnegativity.enabled: true`
  - `shap_reg.nonnegativity.lambda: 0.0038`
  - `shap_reg.nonnegativity.mode: hybrid_mass_softcount`
  - `shap_reg.nonnegativity.soft_count_weight: 0.28`
  - `shap_reg.nonnegativity.soft_count_temperature: 0.012`

## Main Model & Metrics
- Model checkpoint:
  - `results/anfis_model_state_20260320_062903_v2_1_light_nonneg_20260320.pt`
- Training summary:
  - `results/training_summary_20260320_062903_v2_1_light_nonneg_20260320.json`
- SHAP history:
  - `results/shap_history_20260320_062903_v2_1_light_nonneg_20260320.json`
- Test metrics:
  - MSE: `0.01051771`
  - RMSE: `0.10255587`
  - MAE: `0.04741066`
  - R2 (weighted): `0.83833671`
  - R2 (mean): `0.55642551`
- Physicality diagnostics:
  - `negative_fraction = 0.11288889`
  - `negative_count = 508`
  - `dominant_regularizer = tikhonov`
  - `dominant_shap_component = consistency`

## Previous Official V2 Baseline
- Previous official V2 summary:
  - `results/training_summary_20260320_055350_v2_official_det_20260320.json`
- Main deltas vs previous official V2:
  - `MSE: 0.01056393 -> 0.01051771`
  - `RMSE: 0.10278100 -> 0.10255587`
  - `MAE: 0.04741353 -> 0.04741066`
  - `R2 weighted: 0.83762617 -> 0.83833671`
  - `R2 mean: 0.55768377 -> 0.55642551`
  - `negative_fraction: 0.19422222 -> 0.11288889`

## Baselines & Full Comparison
Full-size comparison on the same `75`-sample held-out real test split:

- `Vanilla`:
  - `results/vanilla_full_20260319/training_summary_20260319_202741_vanilla_full_20260319.json`
- `Tikhonov-only`:
  - `results/training_summary_20260319_210248_tikhonov_only_full_20260319.json`
- `SHAP-only`:
  - `results/training_summary_20260319_210638_shap_only_full_20260319.json`
- `V2.1 SHAP+Tikhonov+Hybrid-Nonnegativity`:
  - `results/training_summary_20260320_062903_v2_1_light_nonneg_20260320.json`

Key summary:
- Best by `MSE`, `RMSE`, `R2 (weighted)`: `V2.1 SHAP+Tikhonov`
- Best by `MAE`: `Vanilla`
- Best by `R2 (mean)`: `SHAP-only`
- Best Monte Carlo robustness overall: `Vanilla`
- Best Monte Carlo robustness among regularized variants: `Tikhonov-only`
- `V2.1` is the current official main run because it is the strongest overall compromise and substantially reduces negative bins relative to previous combined V2

Method-comparison artifacts:
- `results/method_comparison_20260320_v2_1/method_comparison_summary.csv`
- `results/method_comparison_20260320_v2_1/method_comparison_summary.md`
- `results/method_comparison_20260320_v2_1/fig_method_tradeoff.png`
- `results/method_comparison_20260320_v2_1/fig_band_quality.png`
- `results/method_comparison_20260320_v2_1/fig_uncertainty_methods.png`

## Final Figure Pack
Main publication figures are stored in:
- `results/final_figures_20260320_v2_1/manifest.md`
- `results/final_figures_20260320_v2_1/fig_01_metrics_comparison.png`
- `results/final_figures_20260320_v2_1/fig_02_mean_spectra_comparison.png`
- `results/final_figures_20260320_v2_1/fig_03_representative_spectrum.png`
- `results/final_figures_20260320_v2_1/fig_04_regularization_comparison.png`
- `results/final_figures_20260320_v2_1/fig_05_shap_importance.png`
- `results/final_figures_20260320_v2_1/fig_06_uncertainty_monte_carlo.png`

## Uncertainty / Monte Carlo
Single-reference Monte Carlo sweep for `0.5, 1, 2, 5, 10%`, `N=1000`:
- `results/uncertainty_article_v2_1_20260320/uncertainty_summary_20260320_065020.csv`
- `results/uncertainty_article_v2_1_20260320/error_0.5/`
- `results/uncertainty_article_v2_1_20260320/error_1/`
- `results/uncertainty_article_v2_1_20260320/error_2/`
- `results/uncertainty_article_v2_1_20260320/error_5/`
- `results/uncertainty_article_v2_1_20260320/error_10/`

Key uncertainty summary for V2.1:
- `0.5%`: `mean_std=3.95e-05`, `max_std=2.15e-04`, `cv=0.00211`
- `1.0%`: `mean_std=7.90e-05`, `max_std=4.31e-04`, `cv=0.00423`
- `2.0%`: `mean_std=1.58e-04`, `max_std=8.61e-04`, `cv=0.00847`
- `5.0%`: `mean_std=3.96e-04`, `max_std=2.15e-03`, `cv=0.02122`
- `10.0%`: `mean_std=7.77e-04`, `max_std=4.16e-03`, `cv=0.04166`

Cross-method uncertainty on the `75` real-test inputs (`N=300` per level):
- `results/uncertainty_compare_vanilla_20260319/`
- `results/uncertainty_compare_tikhonov_only_20260319/`
- `results/uncertainty_compare_shap_only_20260319/`
- `results/uncertainty_compare_v2_1_light_20260320/`

## Article Math
- Main article body: `article_math.tex`
- Build wrapper: `article_math_full.tex`
- Compiled PDF: `article_math_full.pdf`
- Full implementation-aligned derivation: `math.tex`
- Compiled technical PDF: `math.pdf`

Current article status:
- synchronized with the current `V2.1` run and the latest figures
- includes 4-method comparison (`Vanilla`, `Tikhonov-only`, `SHAP-only`, `V2.1 SHAP+Tikhonov`)
- documents band-aware SHAP, log-energy-aware Tikhonov, and hybrid nonnegativity
- honestly records the remaining trade-off: better primary metrics and lower negative fraction, but a small drop in `R2 mean` versus the previous official V2

## Tests
- Full test suite:
  - `source /home/lebedeffson/Code/venv/bin/activate && pytest -q`
- Current result:
  - `91 passed, 13 subtests passed`

## Repro Commands
Main training run:
```bash
source /home/lebedeffson/Code/venv/bin/activate
python train.py --config configs/config_integrated_shap.yaml --tag v2_1_light_nonneg_20260320
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
  --model results/anfis_model_state_20260320_062903_v2_1_light_nonneg_20260320.pt \
  --input "0.016725994,0.028362745,0.063373974,0.14926387,0.1718632,0.17809387,0.15080825,0.11797526,0.077408414,0.046124427" \
  --output-dir results/inference_v2_1
```

Monte Carlo uncertainty:
```bash
source /home/lebedeffson/Code/venv/bin/activate
python uncertainty_analysis.py \
  --config configs/config_integrated_shap_v2_1_light_candidate.yaml \
  --model results/anfis_model_state_20260320_062903_v2_1_light_nonneg_20260320.pt \
  --input-csv results/analysis_inputs/single_input_reference.csv \
  --n-samples 1000 \
  --error-percent-list 0.5,1,2,5,10 \
  --plot-each \
  --output-dir results/uncertainty_article_v2_1_20260320
```

Publication figures:
```bash
source /home/lebedeffson/Code/venv/bin/activate
python prepare_publication_figures.py \
  --shap-summary results/training_summary_20260320_062903_v2_1_light_nonneg_20260320.json \
  --vanilla-summary results/vanilla_full_20260319/training_summary_20260319_202741_vanilla_full_20260319.json \
  --uncertainty-dir results/uncertainty_article_v2_1_20260320 \
  --output-dir results/final_figures_20260320_v2_1
```
