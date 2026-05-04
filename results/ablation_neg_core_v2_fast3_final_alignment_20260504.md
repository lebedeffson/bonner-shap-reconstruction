# Negative Controls v2 (fast3) — Alignment-Oriented Table

Source: `results/ablation_neg_core_v2_fast3_final.csv`

| Variant | ΔR² mean | AUC gap | Top/random | Entropy | N_eff | Mass@3 | Fallback rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_rho1 | -0.030117 | 0.303043 | 1.512385 | 2.049293 | 5.913884 | 0.650207 | 0.00 |
| random_target | -0.030119 | 0.303041 | 1.512392 | 2.049294 | 5.913886 | 0.650207 | 0.00 |
| shuffled_q_err | -0.030115 | 0.303046 | 1.512382 | 2.049291 | 5.913861 | 0.650209 | 0.00 |
| uniform_target | -0.030116 | 0.303043 | 1.512384 | 2.049293 | 5.913884 | 0.650207 | 0.00 |
| anti_q_err | -0.030116 | 0.303043 | 1.512391 | 2.049294 | 5.913886 | 0.650207 | 0.00 |
| sparsity_only | -0.095325 | 0.204276 | 1.364775 | 2.055192 | 5.957342 | 0.647866 | 0.00 |
| task_only | -0.030118 | 0.303042 | 1.512383 | 2.049293 | 5.913882 | 0.650207 | 0.00 |

Note: `p~q correlation` is not serialized in these fast summaries; this table uses the saved alignment proxies (entropy, N_eff, Mass@3).