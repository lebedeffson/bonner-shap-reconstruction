# Neural Unfolding Benchmark

- Config: `/home/lebedeffson/Code/bonner-shap-reconstruction/configs/config_integrated_shap.yaml`
- Seeds: `[42, 43, 44]`
- Device: `cuda`
- Samples: `375`; Features: `10`; Bins: `60`

| Model | R2_w mean±std | R2_mean mean±std | Cosine mean±std | Rel-L1 mean±std | RMSE mean±std | MAE mean±std |
|---|---:|---:|---:|---:|---:|---:|
| CNN1D | 0.721035 ± 0.047493 | 0.397336 ± 0.048232 | 0.842390 ± 0.006484 | 0.634418 ± 0.010679 | 0.125992 ± 0.014771 | 0.058722 ± 0.006570 |
| MC_Dropout_MLP | 0.703123 ± 0.042156 | 0.378666 ± 0.036719 | 0.834881 ± 0.001453 | 0.622117 ± 0.012166 | 0.130067 ± 0.013350 | 0.058202 ± 0.006389 |
| MLP | 0.681394 ± 0.032670 | 0.350590 ± 0.016254 | 0.825017 ± 0.008921 | 0.679225 ± 0.024705 | 0.134662 ± 0.008740 | 0.062988 ± 0.004664 |
