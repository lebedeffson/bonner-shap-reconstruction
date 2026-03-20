# Method Comparison

Сравнение методов по качеству, гладкости и регуляризации.

- Лучший по MSE: `V2.1 SHAP+Tikh`
- Лучший по R² weighted: `V2.1 SHAP+Tikh`
- Лучший по доле неотрицательных бинов: `V2.1 SHAP+Tikh`

## Таблица

label,summary_path,training_time_total,training_time_shap,mse,rmse,mae,r2_weighted,r2_mean,band_0_19_r2,band_20_39_r2,band_40_59_r2,regularization_share_mean,shap_contribution_mean,tikhonov_contribution_mean,nonnegativity_contribution_mean,negative_fraction,negative_count,dominant_regularizer,dominant_shap_component,d1_mean_sq,d2_mean_sq,d1_error_sq,d2_error_sq
V2.1 SHAP+Tikh,/home/lebedeffson/Code/bonner-shap-reconstruction/results/training_summary_20260320_062903_v2_1_light_nonneg_20260320.json,231.055145,71.216526,0.010518,0.102556,0.047411,0.838337,0.556426,0.484344,0.678263,0.506669,0.063968,0.000050,0.000351,0.000049,0.112889,508.000000,tikhonov,consistency,0.069640,0.206883,0.008912,0.019164
SHAP-only,/home/lebedeffson/Code/bonner-shap-reconstruction/results/training_summary_20260319_210638_shap_only_full_20260319.json,189.220366,54.927922,0.010596,0.102935,0.048590,0.837138,0.558447,0.481895,0.675309,0.518137,0.011345,0.000058,0.000000,0.000000,,,shap,consistency,,,,
Tikhonov-only,/home/lebedeffson/Code/bonner-shap-reconstruction/results/training_summary_20260319_210248_tikhonov_only_full_20260319.json,187.137932,37.383773,0.010630,0.103101,0.048674,0.836615,0.555909,0.472943,0.677606,0.517176,0.058075,0.000000,0.000423,0.000000,,,tikhonov,sparsity,,,,
Vanilla,/home/lebedeffson/Code/bonner-shap-reconstruction/results/vanilla_full_20260319/training_summary_20260319_202741_vanilla_full_20260319.json,117.587705,0.000000,0.011327,0.106429,0.046521,0.825896,0.549700,0.496667,0.655848,0.496585,0.000000,0.000000,0.000000,0.000000,,,none,none,0.063561,0.189654,0.009004,0.019491
