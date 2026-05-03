# Expanded Negative Controls (SML2010, ANFIS, mid3 core/anti)

Sources:
- `results/ablation_neg_mid3_core_summary.md`
- `results/ablation_neg_mid3_anti_summary.md`

## Core matrix (3 seeds, unmasked, ea_raw eval)

| Variant | ΔR² mean | AUC gap | Top/random |
|---|---:|---:|---:|
| full | +0.0791 | 0.3750 | 2.0651 |
| task_only | +0.0893 | 0.3711 | 2.0667 |
| random_target | +0.1091 | 0.4008 | 2.0971 |
| shuffled_q_err | +0.1040 | 0.3913 | 2.0710 |
| sparsity_only | +0.0351 | 0.3869 | 2.1630 |

## Anti-target check

| Variant | ΔR² mean | AUC gap | Top/random |
|---|---:|---:|---:|
| full | +0.0791 | 0.3750 | 2.0651 |
| anti_q_err | +0.0863 | 0.3712 | 2.0676 |

## Interpretation

- Expanded controls are now reproducible and visible in one place.
- On this setting, variants remain close; this still does **not** isolate a dominant `q_err`-specific effect.
- For strong-journal claim, keep wording conservative and continue with gamma/divergence sweeps.
