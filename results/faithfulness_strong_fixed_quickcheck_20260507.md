# Faithfulness Report (Top/Random/Bottom)

- Summary: `results/training_summary_20260507_001106_strong_fixed_quickcheck_20260507.json`
- Config: `/home/lebedeffson/Code/bonner-shap-reconstruction/configs/config_shap_exact_strong_fixed_quickcheck.yaml`
- Model: `/home/lebedeffson/Code/bonner-shap-reconstruction/results/anfis_model_state_20260507_001106_strong_fixed_quickcheck_20260507.pt`
- Masking: `permute`
- k_max: `4`
- random_trials: `20`

- base MSE: `0.072452`
- AUC top: `0.009530`
- AUC random: `0.008091`
- AUC bottom: `0.001540`
- AUC gap (top-bottom): `0.007990`
- Top/random ratio: `1.177920`

| k | ΔMSE top | ΔMSE random | ΔMSE bottom |
|---:|---:|---:|---:|
| 1 | 0.000665 | 0.000992 | 0.000329 |
| 2 | 0.002454 | 0.002333 | 0.000463 |
| 3 | 0.002957 | 0.003118 | 0.000488 |
| 4 | 0.007572 | 0.004286 | 0.000850 |
