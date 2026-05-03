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

## 2.1) Time-block split sensitivity (SML2010, fast 5 seed)

Файлы:
- `results/sml2010_faithfulness_mode_compare_timeblock5_sm_20260503.md`
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_sm.json`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_sm_20260503.md`

Ключ:
- `all runs ΔR² mean = -0.000244`
- `stable only ΔR² mean = -0.000251`
- `AUC gap`: vanilla gradient `-0.4245`, vanilla permutation `+0.6141`, EAAR internal `+0.4744`, EAAR final policy `+0.3103`.

Вывод:
- even under time-block split, EAAR internal remains functionally correct; permutation remains strongest external baseline.

## 2.2) Time-block split confirmation (SML2010, non-fast 3 seed)

Файлы:
- `results/sml2010_faithfulness_mode_compare_timeblock3_nf_20260503.md`
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock3_nf.json`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock3_nf_20260503.md`

Ключ:
- `unstable runs = 0`
- `ΔR² mean = -0.000151`
- `AUC gap`: vanilla gradient `-0.2953`, vanilla permutation `+0.3935`, EAAR internal `+0.3109`, EAAR final policy `+0.3109`.

Вывод:
- non-fast mini подтверждает time-block результат без смены знаков.

## 2.3) Time-block split full (SML2010, non-fast 5 seed)

Файлы:
- `results/sml2010_faithfulness_mode_compare_timeblock5_nf_20260503.md`
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_nf.json`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_nf_20260503.md`

Ключ:
- `unstable runs = 0`
- `ΔR² mean = -0.000155`, `wins/losses = 0/5`, `fallback_rate = 0.40`
- `AUC gap`: vanilla gradient `-0.2913`, vanilla permutation `+0.4113`, EAAR internal `+0.3248`, EAAR final `+0.0724`.

Вывод:
- full non-fast time-block подтверждает robustness основного паттерна;
- final-policy положительный, но заметно слабее `ea-only`.

## 3) Portability: MLP vanilla vs MLP+EAAR (SML2010, 5 seed)

Файлы:
- `results/mlp_eaar_vs_vanilla_sml2010_5seed_20260503.md`
- `results/mlp_eaar_multiseed_config_sml2010_mlp_ea_sml_mlp_eaar5.json`
- `results/significance_mlp_eaar_vs_vanilla_auc_gap.json`

Ключевые числа:
- `ΔR² mean = -0.000281` (negligible impact)
- `R² wins/losses = 3/2`
- `vanilla AUC gap mean = 16.2754`
- `EAAR AUC gap mean = 18.1475`
- `ΔAUC gap = +1.8721` (`+11.50%`)
- Faithfulness wins/losses (AUC gap): `4/1`

Это главный новый аргумент переносимости за пределы ANFIS.

## 3.1) Новая задача: multiclass classification (Covertype, 100k subset, 5 seed)

Файлы:
- `results/covtype_mlp_eaar_vs_vanilla_5seed_20260503.md`
- `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_covtype_cls_eaar5.json`
- `results/significance_covtype_mlp_eaar_vs_vanilla_auc_gap_ce_5seed.json`

Ключевые числа:
- `ΔAccuracy mean = +0.001270`
- `ΔMacro-F1 mean = -0.001682`
- `ΔCE mean = -0.001051`
- `vanilla AUC gap (CE) = 2.3420`
- `EAAR AUC gap (CE) = 2.4138`
- `ΔAUC gap (CE) = +0.0718`
- Faithfulness wins/losses: `5/0`

Интерпретация:
- EAAR улучшает internal faithfulness и на классификации.
- Качество по accuracy сохраняется практически на месте.
- Статистика по `ΔAUC gap (CE)`: `CI95 = [0.0384, 0.1052]`, `wins/losses = 5/0` (N=5).

## 3.2) Final-policy refresh (SML2010, fast 5 seed)

Файлы:
- `results/sml2010_faithfulness_mode_compare_policy_refresh5_20260503.md`
- `results/multiseed_config_sml2010_ea_minimal_sml_policy_refresh5.json`

Ключ:
- `EAAR final policy AUC gap = +0.0161` (было отрицательно в старом non-fast отчете).
- `Top/random = 1.1585` (выше 1, но заметно слабее `ea-only`).
- `stable_only ΔR² ≈ +0.000003`.

Вывод:
- faithfulness-aware policy убрала критический отрицательный final-gap,
- но final-policy всё ещё слабее, чем `EAAR internal`.

## 3.3) Classification scaling (Covertype 300k, 3 seed)

Файлы:
- `results/covtype300k_mlp_eaar_vs_vanilla_3seed_20260503.md`
- `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_300k_covtype300k_cls_eaar3.json`
- `results/significance_covtype300k_mlp_eaar_vs_vanilla_auc_gap_ce.json`

Ключ:
- `ΔAccuracy mean = -0.000211`
- `ΔMacro-F1 mean = -0.000717`
- `ΔCE mean = +0.002098`
- `ΔAUC gap (CE) = +0.0941`
- `CI95(ΔAUC gap CE) = [0.0736, 0.1145]`, `wins/losses = 3/0`

Вывод:
- faithfulness-эффект сохраняется и на более крупном масштабе (300k subset).

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

## 4.1) Negative controls (non-fast, 3 seed, ANFIS)

Файл:
- `results/ablation_neg3_nonfast_controls_20260503.md`

Ключ:
- unmasked режим (fallback off) показал различия:
  - `full AUC gap = 0.4992`
  - `random_target AUC gap = 0.4369`
  - `shuffled_q_err AUC gap = 0.4798`
  - `sparsity_only AUC gap = 0.5077`
- вывод: block стал информативнее (random хуже full), но контроль все ещё не идеальный, т.к. `sparsity_only` не хуже.

## 4.2) Expanded negative-controls snapshot

Файл:
- `results/ablation_neg_controls_expanded_20260503.md`

Статус:
- собрана единая матрица `task-only / random / shuffled / sparsity / anti`;
- варианты остаются близкими по `AUC gap`, поэтому для сильного claim нужен gamma/divergence sweep.

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

1. доработать negative controls (gamma/divergence sweep; `full` должен стабильно выигрывать у `random/shuffled/sparsity/task-only`).
2. time-block sensitivity (`5 seed non-fast`) закрыт; при необходимости добавить rolling-split.
3. (после этого) обновить Q1-док финальной таблицей:
   - ANFIS modes + MLP portability + negative controls.

## 8.1) Timing table

Файл:
- `results/timing_table_q1_20260503.md`

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
