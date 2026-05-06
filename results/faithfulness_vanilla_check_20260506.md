# Faithfulness Report (Top/Random/Bottom)

- Summary: `results/training_summary_20260506_230524_vanilla_check_20260506.json`
- Config: `configs/config_vanilla_r2_09.yaml`
- Model: `/home/lebedeffson/Code/bonner-shap-reconstruction/results/anfis_model_state_20260506_230524_vanilla_check_20260506.pt`
- Masking: `permute`
- k_max: `4`
- random_trials: `20`

- base MSE: `0.071869`
- AUC top: `0.021206`
- AUC random: `0.009108`
- AUC bottom: `0.000148`
- AUC gap (top-bottom): `0.021058`
- Top/random ratio: `2.328371`

| k | ΔMSE top | ΔMSE random | ΔMSE bottom |
|---:|---:|---:|---:|
| 1 | 0.000739 | 0.001054 | 0.000121 |
| 2 | 0.007561 | 0.002505 | 0.000197 |
| 3 | 0.008680 | 0.003824 | -0.000045 |
| 4 | 0.009192 | 0.004504 | -0.000129 |
