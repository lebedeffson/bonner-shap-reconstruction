# Faithfulness Report (Top/Random/Bottom)

- Summary: `results/training_summary_20260507_092801_alignment_push_quickcheck_20260507.json`
- Config: `/home/lebedeffson/Code/bonner-shap-reconstruction/configs/config_shap_exact_alignment_push_quickcheck.yaml`
- Model: `/home/lebedeffson/Code/bonner-shap-reconstruction/results/anfis_model_state_20260507_092801_alignment_push_quickcheck_20260507.pt`
- Masking: `permute`
- k_max: `4`
- random_trials: `20`

- base MSE: `0.071861`
- AUC top: `0.017230`
- AUC random: `0.008907`
- AUC bottom: `0.001146`
- AUC gap (top-bottom): `0.016084`
- Top/random ratio: `1.934409`

| k | ΔMSE top | ΔMSE random | ΔMSE bottom |
|---:|---:|---:|---:|
| 1 | 0.000176 | 0.001085 | -0.000036 |
| 2 | 0.003900 | 0.002414 | 0.000417 |
| 3 | 0.008833 | 0.003616 | 0.000312 |
| 4 | 0.008820 | 0.004668 | 0.000868 |
