# SHAP Semantics Sweep

- Base config: `/home/lebedeffson/Code/bonner-shap-reconstruction/configs/multiseed_gapaware_v2_short/config_integrated_shap_short40_seed42_dev_fastpso.yaml`

| Baseline mode | Value function | Status | R2_w | R2_mean | Sel AUC gap | Sel top/random | Summary |
|---|---|---|---:|---:|---:|---:|---|
| feature_mean | mean_output | ok | 0.808787 | 0.487354 | 0.064042 | 1.030570 | results/training_summary_20260507_130055_q1_semantics_s40_s42_feature_mean_mean_output.json |
| feature_mean | sum_output | ok | 0.808787 | 0.487354 | 0.064042 | 1.030570 | results/training_summary_20260507_130325_q1_semantics_s40_s42_feature_mean_sum_output.json |
| feature_mean | l2_output | ok | 0.808823 | 0.487950 | 0.064011 | 1.030300 | results/training_summary_20260507_130553_q1_semantics_s40_s42_feature_mean_l2_output.json |
| median | mean_output | ok | 0.808804 | 0.487363 | 0.064045 | 1.030505 | results/training_summary_20260507_130822_q1_semantics_s40_s42_median_mean_output.json |
| median | sum_output | ok | 0.808804 | 0.487362 | 0.064045 | 1.030505 | results/training_summary_20260507_131051_q1_semantics_s40_s42_median_sum_output.json |
| median | l2_output | ok | 0.808825 | 0.487936 | 0.064015 | 1.030380 | results/training_summary_20260507_131316_q1_semantics_s40_s42_median_l2_output.json |
| zero | mean_output | ok | 0.808836 | 0.487529 | 0.064067 | 1.030638 | results/training_summary_20260507_131544_q1_semantics_s40_s42_zero_mean_output.json |
| zero | sum_output | ok | 0.808836 | 0.487529 | 0.064067 | 1.030638 | results/training_summary_20260507_131813_q1_semantics_s40_s42_zero_sum_output.json |
| zero | l2_output | ok | 0.808879 | 0.487938 | 0.064020 | 1.029898 | results/training_summary_20260507_132037_q1_semantics_s40_s42_zero_l2_output.json |
