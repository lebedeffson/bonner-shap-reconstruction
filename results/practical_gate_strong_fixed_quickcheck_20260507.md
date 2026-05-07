# Practical Readiness Gate

- summary: `results/training_summary_20260507_001106_strong_fixed_quickcheck_20260507.json`
- faithfulness: `results/faithfulness_strong_fixed_quickcheck_20260507.json `
- alignment: `results/importance_alignment_strong_fixed_quickcheck_20260507.json `

- overall: `FAIL`

| check | value | rule | status |
|---|---:|---:|---|
| r2_weighted | 0.838623 | >= 0.8 | PASS |
| regularization_share_mean | 0.257947 | >= 0.05 | PASS |
| shap_contribution_mean | 0.002599 | >= 0.0001 | PASS |
| faithfulness_auc_gap | 0.007990 | >= 0.0 | PASS |
| faithfulness_top_random_ratio | 1.177920 | >= 1.0 | PASS |
| alignment_cosine | 0.598358 | >= 0.65 | FAIL |
