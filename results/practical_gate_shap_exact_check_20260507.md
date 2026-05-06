# Practical Readiness Gate

- summary: `results/training_summary_20260506_232134_shap_exact_check_20260506.json`
- faithfulness: `results/faithfulness_shap_exact_check_20260506.json `
- alignment: `results/importance_alignment_shap_vs_internal_20260506.json `

- overall: `FAIL`

| check | value | rule | status |
|---|---:|---:|---|
| r2_weighted | 0.824553 | >= 0.8 | PASS |
| regularization_share_mean | 0.037145 | >= 0.05 | FAIL |
| shap_contribution_mean | 0.000005 | >= 0.0001 | FAIL |
| faithfulness_auc_gap | 0.016376 | >= 0.0 | PASS |
| faithfulness_top_random_ratio | 2.524146 | >= 1.0 | PASS |
| alignment_cosine | 0.794848 | >= 0.65 | PASS |
