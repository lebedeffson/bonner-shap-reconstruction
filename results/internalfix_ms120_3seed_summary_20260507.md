# Internal-Importance Fix Multi-Seed Summary (2026-05-07)

Config: `multiseed_gapaware_v2_short`, epochs=120, gap-aware selection + task-loss-gradient internal importance.

| Seed | AUC gap SHAP | AUC gap Internal | AUC gap Vanilla-importance | Top/random SHAP | Top/random Internal | Top/random Vanilla-importance | R2_w model |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.005222 | 0.018414 | -0.001561 | 1.0485 | 4.0035 | 0.4784 | 0.809471 |
| 43 | 0.010813 | 0.017359 | -0.009120 | 2.1232 | 3.6421 | 0.0770 | 0.829096 |
| 44 | 0.010167 | 0.010423 | -0.001515 | 2.0256 | 2.0330 | 0.4911 | 0.719495 |

## Aggregates
- AUC gap SHAP: mean=0.008734, std=0.002497
- AUC gap Internal: mean=0.015399, std=0.003545
- AUC gap Vanilla-importance: mean=-0.004065, std=0.003574
- Top/random SHAP: mean=1.732435, std=0.485264
- Top/random Internal: mean=3.226181, std=0.856514
- Top/random Vanilla-importance: mean=0.348858, std=0.192292
- R2_w model: mean=0.786020, std=0.047718
- Wins Internal vs Vanilla-importance by AUC gap: 3/3
- Wins SHAP vs Vanilla-importance by AUC gap: 3/3
- Internal Top/random > 1.0: 3/3
