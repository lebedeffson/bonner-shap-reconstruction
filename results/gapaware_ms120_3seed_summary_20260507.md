# Gap-Aware Multi-Seed Short Summary (2026-05-07)

Config: `multiseed_gapaware_v2_short`, epochs=120, selection by validation deletion-gap with `quality_metric=r2_var_weighted`.

| Seed | AUC gap SHAP | AUC gap Internal | AUC gap Vanilla | Top/random SHAP | Top/random Internal | Top/random Vanilla | R2_w SHAP | R2_w Vanilla |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.011257 | 0.003591 | 0.002262 | 2.0873 | 0.7792 | 1.0540 | 0.834009 | 0.825896 |
| 43 | 0.014151 | -0.000217 | -0.008041 | 2.9187 | 0.0032 | 0.2494 | 0.821660 | 0.838027 |
| 44 | 0.009870 | 0.001657 | -0.001019 | 1.8592 | 0.3965 | 0.5571 | 0.707201 | 0.689649 |

## Aggregates
- AUC gap SHAP: mean=0.011759, std=0.001783
- AUC gap Internal: mean=0.001677, std=0.001555
- AUC gap Vanilla: mean=-0.002266, std=0.004298
- Top/random SHAP: mean=2.288389, std=0.455349
- Top/random Internal: mean=0.392961, std=0.316828
- Top/random Vanilla: mean=0.620155, std=0.331519
- R2_w SHAP: mean=0.787623, std=0.057090
- R2_w Vanilla: mean=0.784524, std=0.067269
- Wins SHAP vs Vanilla by AUC gap: 3/3
- Wins Internal vs Vanilla by AUC gap: 3/3
