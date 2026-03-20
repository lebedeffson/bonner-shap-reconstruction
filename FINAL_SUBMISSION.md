# Final Submission Summary

This repository now keeps only the latest V2.1 materials and the final send package.
Old intermediate runs, previous result dumps, and outdated send packages were removed to reduce clutter.

## Current Main Version
- Main config: `configs/config_integrated_shap.yaml`
- Exact tuned config: `configs/config_integrated_shap_v2_1_light_candidate.yaml`
- Main run tag: `v2_1_light_nonneg_20260320`

## Main Artifacts Kept In `results/`
- Model:
  - `results/anfis_model_state_20260320_062903_v2_1_light_nonneg_20260320.pt`
- Summary:
  - `results/training_summary_20260320_062903_v2_1_light_nonneg_20260320.json`
- SHAP history:
  - `results/shap_history_20260320_062903_v2_1_light_nonneg_20260320.json`
- Main per-run figures:
  - `results/metrics_20260320_062903_v2_1_light_nonneg_20260320.png`
  - `results/regularization_summary_20260320_062903_v2_1_light_nonneg_20260320.png`
  - `results/feature_importance_shap_20260320_062903_v2_1_light_nonneg_20260320.png`
  - `results/spectra_mean_20260320_062903_v2_1_light_nonneg_20260320.png`
  - `results/spectra_samples_20260320_062903_v2_1_light_nonneg_20260320.png`
  - `results/scatter_20260320_062903_v2_1_light_nonneg_20260320.png`

## Core Metrics
- MSE: `0.01051771`
- RMSE: `0.10255587`
- MAE: `0.04741066`
- R2 weighted: `0.83833671`
- R2 mean: `0.55642551`
- negative_fraction: `0.11288889`
- negative_count: `508`

## Current Supporting Result Packages Kept In `results/`
- Final figures:
  - `results/final_figures_20260320_v2_1/`
- Cross-method comparison:
  - `results/method_comparison_20260320_v2_1/`
- Single-reference Monte Carlo:
  - `results/uncertainty_article_v2_1_20260320/`
- Real-test Monte Carlo for V2.1:
  - `results/uncertainty_compare_v2_1_light_20260320/`
- Analysis inputs:
  - `results/analysis_inputs/`

## Current Documents
- Article source: `article_math.tex`
- Built article PDF: `article_math_full.pdf`
- Math appendix source: `math.tex`
- Built math appendix PDF: `math.pdf`

## Final Send Package
- Directory:
  - `final_send_package_20260320_v2_1/`
- Zip archive:
  - `final_send_package_20260320_v2_1.zip`

Main files inside the send package:
- `01_article_full.pdf`
- `02_technical_appendix.pdf`
- `03_final_materials_overview.pdf`
- `03_final_materials_overview.docx`
- `04_submission_summary.md`

## Tests
- Full suite:
  - `source /home/lebedeffson/Code/venv/bin/activate && pytest -q`
- Current status:
  - `91 passed, 13 subtests passed`

## Note On Cleanup
- Previous `20260319` send package was removed.
- Old `v1`, `v2`, candidate, tuning, smoke, and intermediate result directories were removed.
- Only the latest V2.1 result set and its final materials remain visible in the project tree.
