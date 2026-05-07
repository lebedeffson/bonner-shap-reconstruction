# ML Baselines Benchmark

- Config: `/home/lebedeffson/Code/bonner-shap-reconstruction/configs/config_integrated_shap.yaml`
- Seeds: `[42, 43, 44]`
- Samples: `375`; Features: `10`

| Model | R2_w mean±std | R2_mean mean±std | RMSE mean±std | MAE mean±std |
|---|---:|---:|---:|---:|
| ExtraTrees | 0.832952 ± 0.066767 | 0.611609 ± 0.023994 | 0.095620 ± 0.017637 | 0.034051 ± 0.001845 |
| HGB | 0.763852 ± 0.056980 | 0.512683 ± 0.038547 | 0.115328 ± 0.014625 | 0.042911 ± 0.001951 |
| MLPRegressor | 0.575435 ± 0.047728 | 0.216326 ± 0.006004 | 0.155194 ± 0.007318 | 0.075948 ± 0.001775 |
| RandomForest | 0.810642 ± 0.069849 | 0.563543 ± 0.060483 | 0.102068 ± 0.017226 | 0.037762 ± 0.001518 |
