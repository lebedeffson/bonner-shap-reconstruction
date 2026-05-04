# Covertype Per-Class F1 (Vanilla vs EAAR, 5 seeds)

Source: `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_covtype_cls_eaar5_pc.json`

| Class | F1 vanilla | F1 EAAR | ΔF1 | Support (mean) |
|---:|---:|---:|---:|---:|
| 0 | 0.832982 | 0.835550 | +0.002569 | 7292.0 |
| 1 | 0.870549 | 0.871540 | +0.000991 | 9752.0 |
| 2 | 0.834945 | 0.833674 | -0.001271 | 1231.0 |
| 3 | 0.729456 | 0.718701 | -0.010755 | 94.0 |
| 4 | 0.538710 | 0.531500 | -0.007210 | 327.0 |
| 5 | 0.651713 | 0.651498 | -0.000215 | 598.0 |
| 6 | 0.854196 | 0.858311 | +0.004115 | 706.0 |

- Balanced accuracy vanilla: `0.729552`
- Balanced accuracy EAAR: `0.725581`
- ΔBalanced accuracy: `-0.003970`
- Worst class ΔF1: class `3` with `-0.010755`