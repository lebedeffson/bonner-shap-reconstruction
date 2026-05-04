# Q1/Q2 Extended Validation (2026-05-04)

## 1) Non-fast gamma sweep (ANFIS, SML2010, 3 seed, unmasked)

Files:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_gamma_sweep_v2_nf3_q1q2.json`
- `results/ablation_gamma_sweep_v2_nf3_q1q2.md`
- `results/ablation_gamma_sweep_v2_nf3_q1q2.csv`

Key:
- `task_only`: `ΔR²=-0.0594`, `AUC gap=0.5392`
- `gamma_x03/x10/x30/x100_rho1`: `ΔR²=-0.0556`, `AUC gap=0.5380` (почти одинаково)
- `fallback_rate=1.0` для всех вариантов

Conclusion:
- В текущем контуре чувствительность по gamma остаётся слабой (метрики почти совпадают).
- Ключевой флаг: `fallback_rate=1.0` у всех вариантов, поэтому это скорее quality-gated collapse, чем чистая диагностика `q_err`.

## 2) Non-fast divergence sweep (ANFIS, SML2010, 3 seed, unmasked)

Files:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_divergence_sweep_v1_nf3_q1q2.json`
- `results/ablation_divergence_sweep_v1_nf3_q1q2.md`
- `results/ablation_divergence_sweep_v1_nf3_q1q2.csv`

Variants:
- `div_cosine_mse`: `ΔR²=-0.0556`, `AUC gap=0.5380`
- `div_js`: `ΔR²=-0.0588`, `AUC gap=0.5392`
- `div_mse`: `ΔR²=-0.0362`, `AUC gap=0.5392`
- `div_js_mse`: `ΔR²=-0.0283`, `AUC gap=0.5392`

Conclusion:
- По AUC gap различия минимальные; divergence choice пока не даёт явного separation.
- При `fallback_rate=1.0` интерпретация divergence тоже ограничена: policy сводит варианты к близкому режиму.

## 3) Deep negative ablation core (ANFIS, SML2010, 3 seed, unmasked)

Files:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_neg_core_v2_nf3_q1q2.json`
- `results/ablation_neg_core_v2_nf3_q1q2.md`
- `results/ablation_neg_core_v2_nf3_q1q2.csv`

Variants:
- `full_rho1`: `ΔR²=-0.0556`, `AUC gap=0.5380`
- `random_target`: `ΔR²=-0.0536`, `AUC gap=0.5392`
- `shuffled_q_err`: `ΔR²=-0.0403`, `AUC gap=0.5380`
- `uniform_target`: `ΔR²=-0.0556`, `AUC gap=0.5380`
- `anti_q_err`: `ΔR²=-0.0556`, `AUC gap=0.5380`
- `sparsity_only`: `ΔR²=-0.0501`, `AUC gap=0.5392`
- `task_only`: `ΔR²=-0.0594`, `AUC gap=0.5392`

Conclusion:
- В non-fast unmasked режиме на этой конфигурации уникальный вклад `q_err` всё ещё не изолирован.
- Это не равносильно выводу "`q_err` бесполезен": текущий прогон диагностически схлопнут quality/fallback-политикой.
- Для Q1/Q2 остаются обязательными: `ROAR/KAR-lite` и `SAGE` (либо мощный replacement-блок).

## 4) Correct interpretation of current ablation

- `fallback_rate=1.0` + `wins=0/losses=3` + отрицательный `ΔR²` у всех вариантов = **quality-gated diagnostic collapse**.
- Поэтому эти таблицы нельзя трактовать как механизмный тест "уникального вклада `q_err`".
- Корректный вывод: текущий контур не отделяет эффекты (`q_err` vs compactness) из-за одинаковой финальной policy-траектории.

## 5) Next diagnostic step (must)

- Прогнать ablation в режиме, где policy не может схлопнуть варианты:
  - `eval=ea_raw`
  - `quality_first=false`
  - `reject_on_val_degrade=false`
  - `restore_best_state=false`
  - `accuracy_guard.enabled=false`
  - фиксировать `selected_mode` и `effective_config_sha256` в summary.

## 6) Internal-only no-fallback check (done)

Files:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_neg_core_v2_internal_only_no_fallback_nf3_q1q2.json`
- `results/ablation_neg_core_v2_internal_only_no_fallback_nf3_q1q2.md`
- `results/ablation_neg_core_v2_internal_only_no_fallback_s1_q1q2.md`

Key:
- `fallback_rate=0.0` for all variants (policy collapse removed).
- `nf3` run: deletion AUC fields mostly empty (model-deletion stage skipped in that contour).
- `s1` run (with deletion):
  - `full_rho1`: `AUC gap=0.3858`, `top/random=2.1521`
  - `random_target`: `AUC gap=0.3240`, `top/random=2.0873`
  - `sparsity_only`: `AUC gap=0.3011`, `top/random=2.2922`

Conclusion:
- after removing fallback, `full_rho1` is above `random_target` and `task_only` on this seed;
- but separation vs `sparsity_only` is still not conclusive -> mechanism claim remains bounded.

## 7) ROAR-lite SML2010 (surrogate retrain, 3 seed, k=1,2)

Files:
- `results/roar_lite_sml2010_surrogate_3seed_20260504.json`
- `results/roar_lite_sml2010_surrogate_3seed_20260504.csv`
- `results/roar_lite_sml2010_surrogate_3seed_20260504.md`

Aggregate (`roar_gap_top_minus_bottom`, mean):
- `eaar_internal`: `k1=-0.00019`, `k2=-0.00039`
- `permutation`: `k1=+0.00091`, `k2=+0.00231`
- `vanilla_gradient`: `k1=-0.00247`, `k2=+0.00120` (high variance)

Conclusion:
- retrain-after-removal signal is weak/noisy on 3 seeds for internal methods;
- permutation remains most stable positive ROAR-lite signal in this block.

## 8) SAGE baseline SML2010 (ANFIS checkpoints, 3 seed)

Files:
- `results/sage_sml2010_3seed_20260504.json`
- `results/sage_sml2010_3seed_20260504_meta.csv`
- `results/sage_sml2010_3seed_20260504_features.csv`
- `results/sage_sml2010_3seed_20260504.md`

Seed-level meta:
- `corr(SAGE, vanilla_grad)` = `[-0.041, +0.398, -0.371]` (unstable)
- `corr(SAGE, EAAR_internal)` = `[+0.284, +0.274, +0.237]` (consistently positive)

Top SAGE features (mean):
- `meteo_exterior_sol_sud` `0.2662`
- `lighting_comedor_sensor` `0.1225`
- `meteo_exterior_sol_oest` `0.1061`

Conclusion:
- SAGE baseline is now present for SML2010;
- alignment with EAAR-internal ranking is weak-to-moderate positive across all tested seeds.
