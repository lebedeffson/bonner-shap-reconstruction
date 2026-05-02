# Final Submission Summary

This package contains the current main `V2.1 ANFIS + SHAP + log-energy-aware Tikhonov + hybrid nonnegativity` materials prepared for sending.

## Main Version
- Main config: `configs/config_integrated_shap.yaml`
- Exact tuned config: `configs/config_integrated_shap_v2_1_light_candidate.yaml`
- Main run tag: `v2_1_light_nonneg_20260320`

## Main Artifacts
- `05_conference_talk_script_20260323.docx` — полный текст выступления для доклада
- TeX sources:
  - `tex_sources/article_math.tex`
  - `tex_sources/article_math_full.tex`
  - `tex_sources/math.tex`
- Main article PDF:
  - `01_article_full.pdf`
- Technical appendix PDF:
  - `02_technical_appendix.pdf`
- Overview PDF / DOCX:
  - `03_final_materials_overview.pdf`
  - `03_final_materials_overview.docx`
- This summary:
  - `04_submission_summary.md`

## Main Results
- Summary JSON:
  - `summaries/training_summary_20260320_062903_v2_1_light_nonneg_20260320.json`
- Key metrics:
  - `MSE = 0.01051771`
  - `RMSE = 0.10255587`
  - `MAE = 0.04741066`
  - `R2 weighted = 0.83833671`
  - `R2 mean = 0.55642551`
- Physicality diagnostics:
  - `negative_fraction = 0.11288889`
  - `negative_count = 508`

## Interpretation Of The Current Version
- `V2.1` is the strongest overall compromise among the combined variants.
- It improves `MSE`, `RMSE`, `MAE`, and `R2 weighted` relative to the previous official combined `V2` configuration.
- It also reduces the fraction of negative spectral bins from `0.1942` to `0.1129`.
- `Vanilla` still remains the most robust to input noise.
- `SHAP-only` still has a small advantage in `R2 mean`.

## Current Figures Included
The `figures/` directory contains the current publication figures, including:
- `fig_01_metrics_comparison.png`
- `fig_02_mean_spectra_comparison.png`
- `fig_03_representative_spectrum.png` — 4 representative spectra on one figure
- `fig_04_regularization_comparison.png`
- `fig_05_shap_importance.png` — rounded labels
- `fig_06_uncertainty_monte_carlo.png`
- `fig_band_quality.png`
- `fig_method_tradeoff.png`
- `fig_uncertainty_methods.png`

## Baseline Summaries Kept In This Package
For convenience, this package also keeps compact JSON summaries of the comparison baselines:
- `summaries/training_summary_20260319_202741_vanilla_full_20260319.json`
- `summaries/training_summary_20260319_210248_tikhonov_only_full_20260319.json`
- `summaries/training_summary_20260319_210638_shap_only_full_20260319.json`
- `summaries/training_summary_20260320_055350_v2_official_det_20260320.json`
- `summaries/training_summary_20260320_062903_v2_1_light_nonneg_20260320.json`

## Notes
- The main article now expands abbreviations at first mention, clarifies the SHAP formulas, adds the physical measurement introduction, and explicitly documents the current V2.1 trade-offs.
- The technical appendix is prepared as a supplementary appendix to the article.
- A dose-oriented metric is included in formula form; numerical reporting of that metric requires an external table of fluence-to-dose conversion coefficients.
