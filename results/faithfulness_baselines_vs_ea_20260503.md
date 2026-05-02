# Faithfulness: Baseline Models vs ANFIS EA-minimal

## 1) ANFIS (EA vs Vanilla)

| Dataset | Mode | AUC top | AUC random | AUC bottom | AUC gap (top-bottom) | AUC top/random |
|---|---|---:|---:|---:|---:|---:|
| sml2010 | EA-only | 0.4496 | 0.2280 | 0.0268 | 0.4228 | 1.9716 |
| sml2010 | Vanilla-only | 0.1258 | 0.2280 | 0.4808 | -0.3550 | 0.5516 |
| energy_efficiency | EA-only | 3.0387 | 2.3008 | 0.7831 | 2.2557 | 1.3207 |
| energy_efficiency | Vanilla-only | 1.6498 | 2.3008 | 2.2776 | -0.6279 | 0.7170 |
| naval_propulsion | EA-only | 0.1895 | 0.1384 | 0.0469 | 0.1426 | 1.3691 |
| naval_propulsion | Vanilla-only | 0.1750 | 0.1384 | 0.1713 | 0.0037 | 1.2640 |

## 2) Classic Baselines (10 seed, permute deletion)

| Dataset | Model | AUC top (mean) | AUC random (mean) | AUC bottom (mean) | AUC gap (mean) | AUC top/random (mean) |
|---|---|---:|---:|---:|---:|---:|
| sml2010 | mlp | 106.0475 | 23.1565 | 1.2270 | 104.8205 | 4.6227 |
| sml2010 | hgb | 57.9276 | 7.2047 | 0.0486 | 57.8790 | 8.6654 |
| sml2010 | rf | 57.3003 | 6.8677 | 0.0326 | 57.2678 | 8.8176 |
| sml2010 | et | 44.3347 | 5.5054 | 0.0881 | 44.2467 | 8.1850 |
| energy_efficiency | mlp | 940.5707 | 365.5939 | 21.6715 | 918.8992 | 2.6125 |
| energy_efficiency | hgb | 566.4770 | 179.6392 | 1.9677 | 564.5093 | 3.2559 |
| energy_efficiency | et | 477.0503 | 171.3831 | 5.8983 | 471.1520 | 2.8365 |
| energy_efficiency | rf | 212.8957 | 80.0837 | 9.3239 | 203.5717 | 2.6487 |
| naval_propulsion | mlp | 0.4063 | 0.2286 | 0.0661 | 0.3403 | 1.8023 |
| naval_propulsion | hgb | 0.0009 | 0.0004 | 0.0000 | 0.0008 | 2.1024 |
| naval_propulsion | rf | 0.0007 | 0.0003 | 0.0000 | 0.0007 | 2.1594 |
| naval_propulsion | et | 0.0004 | 0.0002 | 0.0001 | 0.0004 | 2.1701 |

## 3) Quick Take

- EA-minimal у ANFIS даёт корректный порядок `top > random > bottom` на SML2010 и Energy.
- По R² ансамбли сильнее, но faithfulness-профиль у ANFIS EA контролируемый и интерпретируемый.
- Naval остаётся ограничительным кейсом для ANFIS по качеству.