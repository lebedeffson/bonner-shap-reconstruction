# SML2010 Faithfulness Modes (time-block split, non-fast 3 seeds)

Multiseed:
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock3_nf.json`
- unstable runs: `0` (seed `[]`)

R² aggregation:

| Aggregation | n_runs | ΔR² mean | wins/losses | fallback_rate |
|---|---:|---:|---:|---:|
| all runs | 3 | -0.000151 | 0/3 | 0.00 |
| stable only | 3 | -0.000151 | 0/3 | 0.00 |

Faithfulness (permute deletion, random_trials=20):

| Mode | AUC top | AUC random | AUC bottom | AUC gap | Top/random |
|---|---:|---:|---:|---:|---:|
| Vanilla gradient | 0.0118 | 0.1150 | 0.3072 | -0.2953 | 0.0253 |
| Vanilla permutation | 0.3791 | 0.1150 | -0.0144 | +0.3935 | 3.3812 |
| EAAR internal (ea-only) | 0.3195 | 0.1150 | 0.0086 | +0.3109 | 2.8819 |
| EAAR final policy | 0.3195 | 0.1150 | 0.0086 | +0.3109 | 2.8819 |

Interpretation:
- Non-fast mini confirms the same sign pattern under time-block split.
- `vanilla gradient` remains functionally inconsistent, while EAAR modes remain functionally correct.
- External permutation baseline remains stronger than EAAR internal on this protocol.
