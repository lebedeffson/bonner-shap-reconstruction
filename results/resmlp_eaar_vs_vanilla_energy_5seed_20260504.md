# Energy: ResMLP Vanilla vs ResMLP+EAAR (5 seeds)

Source:
- `results/resmlp_eaar_multiseed_config_energy_resmlp_ea_energy_resmlp_eaar5.json`
- `results/significance_energy_resmlp_eaar_vs_vanilla_auc_gap_5seed.json`

## Aggregate

| Metric | Value |
|---|---:|
| n runs | 5 |
| ΔR² mean (EAAR - vanilla) | -0.002107 |
| ΔR² std | 0.006195 |
| R² wins/losses | 1 / 4 |
| Vanilla AUC gap mean | 148.1323 |
| EAAR AUC gap mean | 144.9934 |
| ΔAUC gap mean | -3.1389 |
| ΔAUC gap relative | -2.12% |
| Faithfulness wins/losses (AUC gap) | 3 / 2 |
| 95% CI ΔAUC gap | [-21.6565, 15.3787] |
| Wilcoxon p | 1.0000 |
| Cohen d | -0.1486 |

Interpretation:
- This is a boundary-case result: transfer of EAAR to ResMLP on Energy is mixed.
- Mean faithfulness gain is not positive in this setup, while per-seed direction is unstable.
- Claim remains: portability is dataset/model dependent and should be stated as initial evidence.

## Per-seed deltas

| Seed | ΔR² | Vanilla AUC gap | EAAR AUC gap | ΔAUC gap |
|---:|---:|---:|---:|---:|
| 42 | -0.007424 | 129.5824 | 148.0594 | +18.4769 |
| 43 | -0.001498 | 129.8510 | 131.8885 | +2.0375 |
| 44 | -0.008140 | 183.4764 | 150.3291 | -33.1473 |
| 45 | +0.007187 | 167.0585 | 151.6445 | -15.4141 |
| 46 | -0.000658 | 130.6931 | 143.0454 | +12.3524 |
