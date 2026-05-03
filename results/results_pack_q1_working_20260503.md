# Q1 Working Results Pack (2026-05-03)

Короткий рабочий пакет: где лежат главные результаты и какие числа брать в текст/таблицы.

## 1) ANFIS + EAAR: SML2010 non-fast (stability guard, 5 seed)

Основной файл:
- `results/sml2010_faithfulness_mode_compare_stability_guard5_20260503.md`

Ключевые числа:
- `unstable runs = 1` (seed `45`) — явно помечен.
- `ΔR² stable_only = -0.000087` (качество почти сохранено).
- Faithfulness modes:
  - Vanilla gradient: `AUC gap = -0.4151`
  - Vanilla permutation: `AUC gap = +0.5661`
  - EAAR internal: `AUC gap = +0.4666`
  - EAAR final policy: `AUC gap = -0.1739`

Интерпретация для статьи:
- EAAR чинит внутреннюю атрибуцию (`top > random > bottom`), но пока ниже внешнего permutation baseline.
- Current final-policy (quality-gated) ослабляет faithfulness.

## 2) Stability sensitivity

Файлы:
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_sml_eaar_stability_guard5_20260503.md`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_sml_eaar_stability_guard5_20260503.json`

Ключ:
- `all runs`: `ΔR² mean = +0.001056`
- `stable only`: `ΔR² mean = -8.659e-05`

Это и есть корректный sensitivity-анализ (не скрывать unstable-run, а отделять).

## 3) Portability: MLP vanilla vs MLP+EAAR (SML2010, 5 seed)

Файлы:
- `results/mlp_eaar_vs_vanilla_sml2010_5seed_20260503.md`
- `results/mlp_eaar_multiseed_config_sml2010_mlp_ea_sml_mlp_eaar5.json`

Ключевые числа:
- `ΔR² mean = -0.000281` (negligible impact)
- `R² wins/losses = 3/2`
- `vanilla AUC gap mean = 16.2754`
- `EAAR AUC gap mean = 18.1475`
- `ΔAUC gap = +1.8721` (`+11.50%`)
- Faithfulness wins/losses (AUC gap): `4/1`

Это главный новый аргумент переносимости за пределы ANFIS.

## 3.1) Новая задача: multiclass classification (Covertype, 100k subset, 3 seed)

Файлы:
- `results/covtype_mlp_eaar_vs_vanilla_3seed_20260503.md`
- `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_covtype_cls_eaar3.json`

Ключевые числа:
- `ΔAccuracy mean = +0.000650`
- `ΔMacro-F1 mean = -0.004929`
- `vanilla AUC gap (CE) = 2.3461`
- `EAAR AUC gap (CE) = 2.4187`
- `ΔAUC gap (CE) = +0.0726`
- Faithfulness wins/losses: `3/0`

Интерпретация:
- EAAR улучшает internal faithfulness и на классификации.
- Качество по accuracy сохраняется практически на месте.

## 4) Negative controls (fast sanity)

Файлы:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_sml_ablation_neg3_fast.json`
- `results/ablation_summary_ablation_manifest_config_sml2010_ea_minimal_sml_ablation_neg3_fast.md`

Варианты:
- `full`
- `random_target`
- `shuffled_q_err`
- `sparsity_only`

Статус:
- fast-режим дал почти одинаковые агрегаты → использовать только как dev-sanity.
- Для Q1-доказательности нужен non-fast mini на `full/random_target/shuffled_q_err` с `eval=ea_raw`.

## 5) Statistical evidence (ANFIS main comparison)

Файл:
- `results/significance_sml2010_ea_vs_vanilla_auc_gap.json`

Ключевые числа:
- `delta_mean = 0.7778`
- `CI95 = [0.6408, 0.9149]`
- `wins/losses = 10/0`
- `wilcoxon_p = 0.001953`
- `ttest_p = 1.46e-06`
- `cohen_d = 3.518`

## 6) R² baselines across datasets

Файл:
- `results/methods_compare_multidataset_20260503.md`

Ключ:
- ensemble baselines (`ET/HGB/RF`) существенно выше ANFIS по R².
- корректный claim: не “SOTA accuracy”, а faithfulness/internal attribution repair.

## 7) Manifest (быстрый вход в артефакты)

Файл:
- `results/results_manifest.json`

## 8) Что запускать следующим шагом

1. non-fast negative controls (3 seed):
   - `full, random_target, shuffled_q_err`
   - `eval=ea_raw`, `mask=permute`, `random_trials=20`
2. добавить CI/Wilcoxon для MLP `ΔAUC gap`.
3. (после этого) обновить Q1-док финальной таблицей:
   - ANFIS modes + MLP portability + negative controls.

## 9) Что коммитить (без мусора)

Минимальный чистый набор для репозитория:

- `results/results_manifest.json`
- `results/results_pack_q1_working_20260503.md`
- `results/methods_compare_multidataset_20260503.md`
- `results/faithfulness_baselines_vs_ea_20260503.md`
- `results/sml2010_faithfulness_mode_compare_stability_guard5_20260503.md`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_sml_eaar_stability_guard5_20260503.md`
- `results/mlp_eaar_vs_vanilla_sml2010_5seed_20260503.md`
- `results/ablation_nf3_ea_raw_summary_20260503.md`
- `results/significance_sml2010_ea_vs_vanilla_auc_gap.json`

Тяжелые артефакты не коммитить:

- `results/*/*.pt`
- `results/*/*.npy`
- `results/*/*.png`
- сырые каталоги с тысячами файлов (`results/sml2010_ea_minimal/*`, `results/eaar_v2_ablation_fast3/*`, если не нужны в статье напрямую)
