# Bonner/Spectra Run Check (2026-05-06)

## Runs
- Vanilla: `training_summary_20260506_230524_vanilla_check_20260506.json`
- SHAP exact: `training_summary_20260506_232134_shap_exact_check_20260506.json`

## Predictive metrics (from training summaries)

| Model | MSE | RMSE | MAE | R² (weighted) | R² (mean) | Train time (sec) |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla | 0.008686 | 0.093200 | 0.043526 | 0.863836 | 0.591062 | 328.08 |
| SHAP exact | 0.011414 | 0.106839 | 0.050686 | 0.824553 | 0.531163 | 632.00 |

Δ (SHAP - Vanilla):
- `ΔR²(weighted) = -0.039283`
- `ΔR²(mean) = -0.059899`
- `ΔMSE = +0.002728`
- `time overhead = 1.93x`

## Faithfulness (top/random/bottom deletion, k=1..4, permute)

Files:
- `faithfulness_vanilla_check_20260506.json`
- `faithfulness_shap_exact_check_20260506.json`

| Model | AUC top | AUC random | AUC bottom | AUC gap (top-bottom) | Top/random |
|---|---:|---:|---:|---:|---:|
| Vanilla | 0.021206 | 0.009108 | 0.000148 | 0.021058 | 2.328371 |
| SHAP exact | 0.017219 | 0.006822 | 0.000842 | 0.016376 | 2.524146 |

Δ (SHAP - Vanilla):
- `ΔAUC gap = -0.004682`
- `ΔTop/random = +0.195776`

## Short take
- В этом запуске SHAP exact **снизил точность** относительно vanilla.
- По faithfulness картина смешанная: `top/random` выше у SHAP, но `AUC gap` ниже.
- Для текущего конфига это не даёт выигрыш “и по точности, и по faithfulness” одновременно.

