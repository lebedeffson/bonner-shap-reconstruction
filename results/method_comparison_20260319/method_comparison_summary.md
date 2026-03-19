# Method Comparison

Сравнение методов по качеству, гладкости и регуляризации.

- Лучший по MSE: `SHAP`
- Лучший по R² weighted: `SHAP`
- Лучший по совпадению кривизны спектра (D2 error): `SHAP`

## Таблица

label,summary_path,training_time_total,training_time_shap,mse,rmse,mae,r2_weighted,r2_mean,band_0_19_r2,band_20_39_r2,band_40_59_r2,regularization_share_mean,shap_contribution_mean,tikhonov_contribution_mean,dominant_regularizer,dominant_shap_component,d1_mean_sq,d2_mean_sq,d1_error_sq,d2_error_sq
SHAP,/home/lebedeffson/Code/bonner-shap-reconstruction/results/training_summary_20260319_210638_shap_only_full_20260319.json,189.220366,54.927922,0.010596,0.102935,0.048590,0.837138,0.558447,0.481895,0.675309,0.518137,0.011345,0.000058,0.000000,shap,consistency,0.072886,0.217613,0.008766,0.018632
SHAP+Tikhonov,/home/lebedeffson/Code/bonner-shap-reconstruction/results/training_summary_20260319_190809_tikhonov_stronger_20260319.json,164.372076,48.377712,0.010603,0.102971,0.048904,0.837026,0.554094,0.468325,0.678723,0.515233,0.065316,0.000055,0.000423,tikhonov,consistency,0.070235,0.208379,0.008848,0.018927
Tikhonov,/home/lebedeffson/Code/bonner-shap-reconstruction/results/training_summary_20260319_210248_tikhonov_only_full_20260319.json,187.137932,37.383773,0.010630,0.103101,0.048674,0.836615,0.555909,0.472943,0.677606,0.517176,0.058075,0.000000,0.000423,tikhonov,sparsity,0.070009,0.207775,0.008869,0.018956
Vanilla,/home/lebedeffson/Code/bonner-shap-reconstruction/results/vanilla_full_20260319/training_summary_20260319_202741_vanilla_full_20260319.json,117.587705,0.000000,0.011327,0.106429,0.046521,0.825896,0.549700,0.496667,0.655848,0.496585,0.000000,0.000000,0.000000,none,none,0.063561,0.189654,0.009004,0.019491
