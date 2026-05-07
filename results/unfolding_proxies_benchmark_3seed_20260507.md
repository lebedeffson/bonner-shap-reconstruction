# Unfolding Proxy Benchmark

- Config: `/home/lebedeffson/Code/bonner-shap-reconstruction/configs/config_integrated_shap.yaml`
- Seeds: `[42, 43, 44]`
- Lambdas: `[1e-05, 0.0001, 0.001, 0.01, 0.1]`
- Samples: `375`, Features: `10`, Bins: `60`

Best-per-seed aggregation (criterion: `r2_weighted`):

| Method | R2_w mean±std | R2_mean mean±std | RMSE mean±std | MAE mean±std |
|---|---:|---:|---:|---:|
| gravel_like | -0.363986 ± 0.030166 | -1.519298 ± 0.380741 | 0.278919 ± 0.011372 | 0.110441 ± 0.004232 |
| maxed_like | -0.208955 ± 0.015368 | -0.344271 ± 0.047781 | 0.262707 ± 0.012902 | 0.093887 ± 0.003757 |
| tikhonov_linear | -0.140177 ± 0.013755 | -1.627342 ± 1.737454 | 0.255108 ± 0.012157 | 0.095835 ± 0.003810 |
| tikhonov_nnls | -0.135416 ± 0.032571 | -0.197069 ± 0.093709 | 0.254572 ± 0.012983 | 0.088789 ± 0.004008 |
