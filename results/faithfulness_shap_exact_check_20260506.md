# Faithfulness Report (Top/Random/Bottom)

- Summary: `results/training_summary_20260506_232134_shap_exact_check_20260506.json`
- Config: `/home/lebedeffson/Code/bonner-shap-reconstruction/configs/config_shap_exact_accuracy_first.yaml`
- Model: `/home/lebedeffson/Code/bonner-shap-reconstruction/results/anfis_model_state_20260506_232134_shap_exact_check_20260506.pt`
- Masking: `permute`
- k_max: `4`
- random_trials: `20`

- base MSE: `0.072022`
- AUC top: `0.017219`
- AUC random: `0.006822`
- AUC bottom: `0.000842`
- AUC gap (top-bottom): `0.016376`
- Top/random ratio: `2.524146`

| k | ΔMSE top | ΔMSE random | ΔMSE bottom |
|---:|---:|---:|---:|
| 1 | 0.002150 | 0.000675 | -0.000050 |
| 2 | 0.002315 | 0.001763 | 0.000443 |
| 3 | 0.009211 | 0.002898 | 0.000265 |
| 4 | 0.009235 | 0.003645 | 0.000318 |
