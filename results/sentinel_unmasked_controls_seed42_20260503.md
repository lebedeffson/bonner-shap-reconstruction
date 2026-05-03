# Sentinel Unmasked Controls (seed=42, high gamma, fallback off)

Configs:
- `results/ablation/configs/sentinel_unmasked/config_sml2010_ea_minimal_sentinel_unmasked_*.yaml`
- forced unmasking: `quality_first=false`, `reject_on_val_degrade=false`, `restore_best_state=false`, `accuracy_guard=false`, `acceptance_min_delta_r2=-1.0`

| Variant | target_mode | metrics_source | fallback | ΔR² (ea_raw-vanilla) | ea_ratio(last) | gradR(last) | p~q(last) | q_entropy(last) | q_gini(last) | AUC gap | top/random |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | none | shap | 0 | -0.074422 | 0.0365 | 0.000348 | 0.7014 | 0.9312 | 0.8844 | 0.3527 | 2.1925 |
| random_target | random_target | shap | 0 | -0.069271 | 0.1277 | 0.000053 | 0.9478 | 0.9198 | 0.8853 | 0.3205 | 2.0323 |
| shuffled_q_err | shuffled_q_err | shap | 0 | -0.100953 | 0.2157 | 0.000074 | 0.7481 | 0.9491 | 0.8834 | 0.3299 | 2.1744 |
| sparsity_only | n/a | shap | 0 | -0.054595 | 0.2176 | 0.000037 | n/a | n/a | n/a | 0.3417 | 2.2610 |

Quick read:
- unmasked setup worked (`metrics_source=shap`, `fallback=0` for all variants);
- variants now differ on faithfulness (`AUC gap`: `full` > `shuffled` > `random`);
- `sparsity_only` is competitive on this single seed, so final proof still needs multi-seed non-fast controls.
