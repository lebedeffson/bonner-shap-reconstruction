# SML2010 Non-fast Negative Controls (ANFIS, 3 seeds, unmasked, `eval=ea_raw`)

Multiseed (unmasked):
- `results/multiseed_config_sml2010_ea_minimal_sentinel_unmasked_full_sml_ablation_neg3_unmasked_full.json`
- `results/multiseed_config_sml2010_ea_minimal_sentinel_unmasked_random_target_sml_ablation_neg3_unmasked_random_target.json`
- `results/multiseed_config_sml2010_ea_minimal_sentinel_unmasked_shuffled_q_err_sml_ablation_neg3_unmasked_shuffled_q_err.json`
- `results/multiseed_config_sml2010_ea_minimal_sentinel_unmasked_sparsity_only_sml_ablation_neg3_unmasked_sparsity_only.json`

Explainability:
- `results/explainability_multiseed_config_sml2010_ea_minimal_sentinel_unmasked_full_sml_ablation_neg3_unmasked_full.json`
- `results/explainability_multiseed_config_sml2010_ea_minimal_sentinel_unmasked_random_target_sml_ablation_neg3_unmasked_random_target.json`
- `results/explainability_multiseed_config_sml2010_ea_minimal_sentinel_unmasked_shuffled_q_err_sml_ablation_neg3_unmasked_shuffled_q_err.json`
- `results/explainability_multiseed_config_sml2010_ea_minimal_sentinel_unmasked_sparsity_only_sml_ablation_neg3_unmasked_sparsity_only.json`

| Variant | ΔR² mean | Wins/Losses | AUC gap (top-bottom) | AUC top/random | N_eff | Mass@3 |
|---|---:|---:|---:|---:|---:|---:|
| full | -0.059594 | 0/3 | 0.499158 | 2.451916 | 2.898475 | 0.821027 |
| random_target | -0.102213 | 0/3 | 0.436877 | 2.590376 | 2.882565 | 0.821683 |
| shuffled_q_err | -0.080829 | 0/3 | 0.479756 | 2.556858 | 2.891749 | 0.821698 |
| sparsity_only | -0.120033 | 0/3 | 0.507686 | 2.481157 | 2.908863 | 0.818922 |

## Takeaway

- Unmasking worked: all runs are `metrics_source=shap`, fallback disabled.
- `random_target` is worse than `full` on AUC-gap (signal appears), but `sparsity_only` is not worse.
- So negative controls are now **partially informative**, but still not clean enough for strong causal claim that only `q_err` drives effect.
