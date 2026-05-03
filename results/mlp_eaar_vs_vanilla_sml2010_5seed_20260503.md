# SML2010: MLP Vanilla vs MLP+EAAR (5 seeds)

Source:
- `results/mlp_eaar_multiseed_config_sml2010_mlp_ea_sml_mlp_eaar5.json`

## Aggregate

| Metric | Value |
|---|---:|
| n runs | 5 |
| ΔR² mean (EAAR - vanilla) | -0.000281 |
| ΔR² std | 0.003204 |
| R² wins/losses | 3 / 2 |
| Vanilla AUC gap mean | 16.2754 |
| EAAR AUC gap mean | 18.1475 |
| ΔAUC gap mean | +1.8721 |
| ΔAUC gap relative | +11.50% |
| Faithfulness wins/losses (AUC gap) | 4 / 1 |
| Vanilla top/random mean | 4.9169 |
| EAAR top/random mean | 5.4185 |

Interpretation:
- MLP+EAAR improves faithfulness (`AUC gap`) on most seeds (4/5).
- Predictive impact is small (`ΔR²` close to zero).
- This supports portability beyond ANFIS for a second differentiable model (MLP).

## Per-seed deltas

| Seed | ΔR² | Vanilla AUC gap | EAAR AUC gap | ΔAUC gap |
|---:|---:|---:|---:|---:|
| 42 | -0.005321 | 14.6623 | 18.1185 | +3.4562 |
| 43 | +0.000515 | 17.7631 | 19.0952 | +1.3322 |
| 44 | +0.003343 | 18.0061 | 21.6594 | +3.6533 |
| 45 | +0.000949 | 16.5267 | 16.3767 | -0.1500 |
| 46 | -0.000891 | 14.4187 | 15.4875 | +1.0688 |
