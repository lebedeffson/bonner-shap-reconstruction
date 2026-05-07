# SHAP Scaling Report

- Source glob: `results/training_summary_*.json`
- Rows: `16`

| Tag | Train | Test | Time total (s) | Time SHAP (s) | Exact calls | Exact utility evals |
|---|---:|---:|---:|---:|---:|---:|
| tikhonov_stronger_20260319 | 225 | 75 | 164.37207555770874 | 48.37771248817444 | None | None |
| tikhonov_only_full_20260319 | 225 | 75 | 187.13793182373047 | 37.38377285003662 | None | None |
| shap_only_full_20260319 | 225 | 75 | 189.22036623954773 | 54.92792248725891 | None | None |
| vanilla_check_20260506 | None | 94 | None | None | None | None |
| shap_exact_check_20260506 | 225 | 75 | 631.9991140365601 | 487.31020641326904 | None | None |
| strong_fixed_quickcheck_20260507 | 225 | 75 | 423.1249952316284 | 346.59192085266113 | None | None |
| alignment_push_quickcheck_20260507 | 225 | 75 | 438.0709500312805 | 357.3369722366333 | None | None |
| alignment_push_internalmetric_20260507 | 225 | 75 | 418.39706468582153 | 342.12817454338074 | None | None |
| gapaware_v1 | 225 | 75 | 1169.7903599739075 | 1039.6779069900513 | None | None |
| gapaware_v2_r2w | 225 | 75 | 1109.0978441238403 | 984.0037412643433 | None | None |
| ms120_s42 | 225 | 75 | 524.7711980342865 | 402.8678922653198 | None | None |
| ms120_s43 | 225 | 75 | 520.0455491542816 | 397.4035758972168 | None | None |
| ms120_s44 | 225 | 75 | 516.8926227092743 | 397.75973773002625 | None | None |
| dev_rank_es_fastpso | 225 | 75 | 140.7777931690216 | 134.60999035835266 | None | None |
| dev_rank_es_fastpso_s43 | 225 | 75 | 140.43738627433777 | 134.2825207710266 | None | None |
| q1_shap_compute_smoke | 225 | 75 | 23.61593770980835 | 22.97253942489624 | 9 | 9216 |
