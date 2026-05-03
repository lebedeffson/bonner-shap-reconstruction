# SML2010 Faithfulness Modes (non-fast, 5 seeds, stability guard)

Multiseed:
- `results/multiseed_config_sml2010_ea_minimal_sml_eaar_stability_guard5.json`
- unstable runs: `1` (seed `[45]`)

R² aggregation:

| Aggregation | n_runs | ΔR² mean | wins/losses | fallback_rate |
|---|---:|---:|---:|---:|
| all runs | 5 | +0.001056 | 2/3 | 0.60 |
| stable only | 4 | -0.000087 | 1/3 | 0.75 |

Faithfulness (permute deletion, random_trials=20):

| Mode | AUC top | AUC random | AUC bottom | AUC gap | Top/random |
|---|---:|---:|---:|---:|---:|
| Vanilla gradient | 0.0796 | 0.2143 | 0.4947 | -0.4151 | 0.3687 |
| Vanilla permutation | 0.5713 | 0.2143 | 0.0053 | +0.5661 | 2.6381 |
| EAAR internal (ea-only) | 0.4788 | 0.2143 | 0.0122 | +0.4666 | 2.1792 |
| EAAR final policy | 0.1786 | 0.2143 | 0.3525 | -0.1739 | 0.7763 |

Interpretation:
- Unstable seed is now explicitly flagged and excluded in `stable_only`.
- EAAR internal remains functionally correct (`top > random > bottom`), but still below external permutation baseline.
- Final policy remains quality-gated and weaker on faithfulness due to fallback/rejection behavior.
