# Practical Readiness Gate

- summary: `results/training_summary_20260507_094018_alignment_push_internalmetric_20260507.json`
- faithfulness: `results/faithfulness_alignment_push_internalmetric_20260507.json `
- alignment: `results/importance_alignment_internalmetric_20260507.json `

- overall: `PASS`

| check | value | rule | status |
|---|---:|---:|---|
| r2_weighted | 0.838840 | >= 0.8 | PASS |
| regularization_share_mean | 0.089767 | >= 0.05 | PASS |
| shap_contribution_mean | 0.000566 | >= 0.0001 | PASS |
| faithfulness_auc_gap | 0.016084 | >= 0.0 | PASS |
| faithfulness_top_random_ratio | 1.934409 | >= 1.0 | PASS |
| alignment_cosine | 0.654560 | >= 0.65 | PASS |
