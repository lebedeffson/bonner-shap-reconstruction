# SML2010 Faithfulness Modes (time-block split, non-fast 5 seeds)

Multiseed:
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_nf.json`
- unstable runs: `0` (seed `[]`)

R² aggregation:

| Aggregation | n_runs | ΔR² mean | wins/losses | fallback_rate |
|---|---:|---:|---:|---:|
| all runs | 5 | -0.000155 | 0/5 | 0.40 |
| stable only | 5 | -0.000155 | 0/5 | 0.40 |

Faithfulness (permute deletion, random_trials=20):

| Mode | AUC top | AUC random | AUC bottom | AUC gap | Top/random |
|---|---:|---:|---:|---:|---:|
| Vanilla gradient | 0.0228 | 0.1248 | 0.3141 | -0.2913 | 0.1224 |
| Vanilla permutation | 0.4010 | 0.1248 | -0.0103 | +0.4113 | 3.2618 |
| EAAR internal (ea-only) | 0.3392 | 0.1248 | 0.0145 | +0.3248 | 2.7892 |
| EAAR final policy | 0.2074 | 0.1248 | 0.1350 | +0.0724 | 1.8364 |

Interpretation:
- Full non-fast time-block confirms the core pattern (`vanilla gradient` incorrect, EAAR modes correct).
- External permutation remains stronger than EAAR internal.
- Final policy is now positive on this protocol, but still weaker than EAAR internal.
