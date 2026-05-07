# Revision Snapshot (2026-05-07)

## A) Current short multi-seed evidence (SML2010, seeds 42/43/44)

| Seed | R2_w SHAP | R2_w Vanilla | AUC gap SHAP | AUC gap Internal | AUC gap Vanilla | Top/random SHAP | Top/random Internal | Top/random Vanilla |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.834009 | 0.825896 | 0.011257 | 0.003591 | 0.002262 | 2.0873 | 0.7792 | 1.0540 |\n| 43 | 0.821660 | 0.838027 | 0.014151 | -0.000217 | -0.008041 | 2.9187 | 0.0032 | 0.2494 |\n| 44 | 0.707201 | 0.689649 | 0.009870 | 0.001657 | -0.001019 | 1.8592 | 0.3965 | 0.5571 |\n
- R2_w wins (SHAP > Vanilla): 2/3\n- AUC gap wins (SHAP > Vanilla): 3/3\n- Mean R2_w SHAP: 0.787623 (std 0.057090)\n- Mean R2_w Vanilla: 0.784524 (std 0.067269)\n- Mean AUC gap SHAP: 0.011759 (std 0.001783)\n- Mean AUC gap Internal: 0.001677 (std 0.001555)\n- Mean AUC gap Vanilla: -0.002266 (std 0.004298)\n
## B) Compute cost (from saved training summaries)

| Seed | Total time SHAP (sec) | SHAP stage time (sec) | Total time Vanilla (sec) | Overhead total (x) |
|---:|---:|---:|---:|---:|
| 42 | 524.77 | 402.87 | 121.41 | 4.32 |\n| 43 | 520.05 | 397.40 | 117.68 | 4.42 |\n| 44 | 516.89 | 397.76 | 120.17 | 4.30 |\n\n- Mean overhead (total time): 4.35x\n
## C) Reviewer-facing boundary (for manuscript text)

- Keep strong claim only for SHAP-target faithfulness (deletion AUC gap).\n- Keep predictive-quality claim as near-neutral, not SOTA gain.\n- State explicitly that internal-gradient faithfulness remains weak and is future work.\n