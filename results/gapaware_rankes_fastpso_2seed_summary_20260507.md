# Gap-Aware Rank/ES FastPSO Summary (2026-05-07)

Config family: `multiseed_gapaware_v2_short` + `rank_consistency` + `delayed regularization start` + selection `early-stop`.

## Per-seed results

| Seed | AUC gap SHAP | Top/random SHAP | AUC gap Internal | Top/random Internal | R2_w |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.023988 | 3.6898 | -0.000132 | 0.1334 | 0.808787 |
| 43 | 0.017657 | 3.7244 | 0.000180 | 0.1555 | 0.822386 |

## Aggregates

- AUC gap SHAP (mean): `0.020822`
- AUC gap SHAP (std): `0.003166`
- Top/random SHAP (mean): `3.7071`
- AUC gap Internal (mean): `0.000024`
- R2_w (mean): `0.815587`

## Comparison to previous short baseline (same seeds 42/43)

- Previous SHAP AUC gap mean (ms120 short): `0.012704`
- New SHAP AUC gap mean (rank/es fastpso): `0.020822`
- Delta: `+0.008118` (`+63.90%`)

## Notes

- `seed43` triggered gap-aware early-stop and restored best checkpoint (`epoch=10`).
- SHAP-based faithfulness improved clearly.
- Internal-gradient faithfulness remains weak and requires separate tuning.
