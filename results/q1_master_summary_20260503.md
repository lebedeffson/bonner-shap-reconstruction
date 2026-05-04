# EAAR Q1 Master Summary (2026-05-03)

Единый сводный документ: метод, результаты, ограничения, статус готовности и список артефактов для статьи.

---

## 0) Главный тезис (финальный на текущий момент)

**EAAR (Error-Aware Attribution Regularization)**:

\[
L = L_{task} + \gamma D(p_\theta, q_{err})
\]

где \(q_{err}\) — importance через рост ошибки при маскировании признака, \(p_\theta\) — внутренняя важность модели.

Корректный claim:
- EAAR **не заявляется как SOTA-бустер точности**.
- EAAR **не заменяет** permutation importance как внешний post-hoc baseline.
- Основной эффект EAAR: **repair внутренней атрибуции** и согласование важности с функциональным влиянием на ошибку.

---

## 1) ANFIS + EAAR: SML2010 non-fast (stability guard, 5 seed)

Основной файл:
- `results/sml2010_faithfulness_mode_compare_stability_guard5_20260503.md`

Ключевые числа:
- unstable runs: `1` (seed `45`)
- `ΔR² stable_only = -0.000087`

Faithfulness modes (permute deletion, random_trials=20):

| Mode | AUC top | AUC random | AUC bottom | AUC gap | Top/random |
|---|---:|---:|---:|---:|---:|
| Vanilla gradient | 0.0796 | 0.2143 | 0.4947 | -0.4151 | 0.3687 |
| Vanilla permutation | 0.5713 | 0.2143 | 0.0053 | +0.5661 | 2.6381 |
| EAAR internal (ea-only) | 0.4788 | 0.2143 | 0.0122 | +0.4666 | 2.1792 |
| EAAR final policy | 0.1786 | 0.2143 | 0.3525 | -0.1739 | 0.7763 |

Вывод:
- EAAR чинит внутреннюю важность (`top > random > bottom`).
- External permutation baseline пока сильнее по gap.
- Current final-policy (quality-gated) ослабляет faithfulness.

### 1.1) Final-policy refresh (fast 5 seed, faithfulness-aware gate)

Файлы:
- `results/sml2010_faithfulness_mode_compare_policy_refresh5_20260503.md`
- `results/multiseed_config_sml2010_ea_minimal_sml_policy_refresh5.json`

Ключ:
- `EAAR final policy AUC gap = +0.0161`
- `Top/random = 1.1585`
- `stable_only ΔR² ≈ +0.000003`

Вывод:
- критический отрицательный final-gap снят;
- final-policy стал функционально валидным, но пока существенно слабее `EAAR internal`.

### 1.2) Time-block split sensitivity (SML2010, fast 5 seed, save-model)

Файлы:
- `results/sml2010_faithfulness_mode_compare_timeblock5_sm_20260503.md`
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_sm.json`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_sm_20260503.md`

Ключ:
- unstable runs: `1` (seed `[44]`)
- `all-runs ΔR² mean = -0.000244`
- `stable-only ΔR² mean = -0.000251`
- Faithfulness modes:
  - Vanilla gradient: `AUC gap = -0.4245`
  - Vanilla permutation: `AUC gap = +0.6141`
  - EAAR internal: `AUC gap = +0.4744`
  - EAAR final policy: `AUC gap = +0.3103`

Вывод:
- На time-block split сохраняется основной паттерн (`vanilla gradient` неверен, `EAAR internal` функционально корректен).
- External permutation baseline по-прежнему сильнее EAAR internal.

### 1.3) Time-block split confirmation (SML2010, non-fast 3 seed)

Файлы:
- `results/sml2010_faithfulness_mode_compare_timeblock3_nf_20260503.md`
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock3_nf.json`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock3_nf_20260503.md`

Ключ:
- unstable runs: `0`
- `ΔR² mean = -0.000151`
- Faithfulness modes:
  - Vanilla gradient: `AUC gap = -0.2953`
  - Vanilla permutation: `AUC gap = +0.3935`
  - EAAR internal: `AUC gap = +0.3109`
  - EAAR final policy: `AUC gap = +0.3109`

Вывод:
- Non-fast mini подтверждает time-block паттерн без смены знаков.
- Это усиливает split/leakage robustness-блок для manuscript.

### 1.4) Time-block split full (SML2010, non-fast 5 seed)

Файлы:
- `results/sml2010_faithfulness_mode_compare_timeblock5_nf_20260503.md`
- `results/multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_nf.json`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_timeblock_sml_timeblock5_nf_20260503.md`

Ключ:
- unstable runs: `0`
- `ΔR² mean = -0.000155`, `wins/losses = 0/5`, `fallback_rate = 0.40`
- Faithfulness modes:
  - Vanilla gradient: `AUC gap = -0.2913`
  - Vanilla permutation: `AUC gap = +0.4113`
  - EAAR internal: `AUC gap = +0.3248`
  - EAAR final policy: `AUC gap = +0.0724`

Вывод:
- Full non-fast time-block подтверждает основной sign-pattern.
- Final-policy остается слабее `ea-only`, но уже функционально валиден (`AUC gap > 0`).

---

## 2) Stability sensitivity (ANFIS, SML2010)

Файлы:
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_sml_eaar_stability_guard5_20260503.md`
- `results/stability_sensitivity_multiseed_config_sml2010_ea_minimal_sml_eaar_stability_guard5_20260503.json`

Ключ:
- all runs: `ΔR² mean = +0.001056`
- stable only: `ΔR² mean = -8.659e-05`

Это корректный sensitivity-анализ: unstable-run не скрывается, а отделяется.

---

## 3) Portability: MLP regression (SML2010, 5 seed)

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

Вывод:
- Переносимость за пределы ANFIS подтверждена на второй дифференцируемой модели (MLP).

---

## 3.1) Новая задача: multiclass classification (Covertype, 100k subset, 5 seed)

Файлы:
- `results/covtype_mlp_eaar_vs_vanilla_5seed_20260503.md`
- `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_covtype_cls_eaar5.json`
- `results/significance_covtype_mlp_eaar_vs_vanilla_auc_gap_ce_5seed.json`
- `results/covtype_per_class_full_5seed_20260504.md`

Ключевые числа:
- `ΔAccuracy mean = +0.001270`
- `ΔMacro-F1 mean = -0.001682`
- `ΔCE mean = -0.001051`
- `vanilla AUC gap (CE) = 2.3420`
- `EAAR AUC gap (CE) = 2.4138`
- `ΔAUC gap (CE) = +0.0718`
- Faithfulness wins/losses: `5/0`

Вывод:
- EAAR улучшает internal faithfulness и на классификации.
- Это пока **initial evidence**: accuracy/CE сохраняются, macro-F1 требует мониторинга.
- Статистика по `ΔAUC gap (CE)`: `CI95=[0.0384, 0.1052]`, `wins/losses=5/0`, `N=5`.
- По per-class блоку: `ΔBalanced accuracy = -0.003970`, worst class `ΔF1 = -0.010755`.

## 3.2) Classification scaling check (Covertype, 300k subset, 3 seed)

Файлы:
- `results/covtype300k_mlp_eaar_vs_vanilla_3seed_20260503.md`
- `results/mlp_classifier_eaar_multiseed_config_covtype_mlp_eaar_300k_covtype300k_cls_eaar3.json`
- `results/significance_covtype300k_mlp_eaar_vs_vanilla_auc_gap_ce.json`

Ключевые числа:
- `ΔAccuracy mean = -0.000211`
- `ΔMacro-F1 mean = -0.000717`
- `ΔCE mean = +0.002098`
- `vanilla AUC gap (CE) = 2.7437`
- `EAAR AUC gap (CE) = 2.8378`
- `ΔAUC gap (CE) = +0.0941`
- Faithfulness wins/losses: `3/0`

Вывод:
- На 300k subset faithfulness-эффект сохраняется.
- Predictive-метрики практически на месте, но CE слегка хуже в среднем.

---

## 3.3) Extra model portability check: ResMLP regression (Energy, 5 seed)

Файлы:
- `results/resmlp_eaar_vs_vanilla_energy_5seed_20260504.md`
- `results/resmlp_eaar_multiseed_config_energy_resmlp_ea_energy_resmlp_eaar5.json`
- `results/significance_energy_resmlp_eaar_vs_vanilla_auc_gap_5seed.json`

Ключевые числа:
- `ΔR² mean = -0.002107`
- `vanilla AUC gap mean = 148.1323`
- `EAAR AUC gap mean = 144.9934`
- `ΔAUC gap = -3.1389` (wins/losses `3/2`)
- `CI95(ΔAUC gap) = [-21.6565, 15.3787]`, `wilcoxon_p = 1.0000`

Вывод:
- Это boundary-case: перенос EAAR на `ResMLP + Energy` смешанный.
- Вклад useful для честного claim boundary и ограничения универсальности.

---

## 4) Negative controls

### 4.1 Fast sanity

Файлы:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_sml_ablation_neg3_fast.json`
- `results/ablation_summary_ablation_manifest_config_sml2010_ea_minimal_sml_ablation_neg3_fast.md`

Статус:
- fast-режим давал почти одинаковые агрегаты (использовать только как dev-sanity).

### 4.2 Non-fast controls (ANFIS, 3 seed, unmasked)

Файл:
- `results/ablation_neg3_nonfast_controls_20260503.md`

Ключ:
- `full AUC gap = 0.4992`
- `random_target AUC gap = 0.4369`
- `shuffled_q_err AUC gap = 0.4798`
- `sparsity_only AUC gap = 0.5077`

Вывод:
- блок стал информативнее (random хуже full),
- но контроль пока не идеальный (`sparsity_only` не хуже full).

### 4.3 Expanded controls snapshot (mid3 core/anti)

Файл:
- `results/ablation_neg_controls_expanded_20260503.md`

Ключ:
- матрица `task_only / random_target / shuffled_q_err / sparsity_only / anti_q_err` собрана в единой сводке;
- на текущей настройке варианты близки по `AUC gap` (dominant `q_err`-effect пока не изолирован).

### 4.4 Fast completion runs (v2, 3 seed)

Файлы:
- `results/ablation_neg_core_v2_fast3_final.md`
- `results/ablation_gamma_sweep_v2_fast3_final.md`
- `results/ablation_neg_core_v2_fast3_final_alignment_20260504.md`

Ключ:
- fast-контур завершен стабильно, но варианты почти совпали по агрегатам.

Вывод:
- эти прогоны использовать как sanity/stability check;
- основной механизмный вывод оставляем за non-fast блоком.

### 4.5 Internal-only no-fallback (order-fix, qerr-focus, non-fast 3 seed)

Файлы:
- `results/ablation/ablation_manifest_config_sml2010_ea_minimal_qerr_focus_orderfix_nf3.json`
- `results/qerr_focus_orderfix_nf3_summary.md`

Ключ:
- `fallback_rate = 0.0` для всех вариантов (policy collapse устранён).
- `AUC gap`:
  - `full_rho1 = 0.5144`
  - `random_target = 0.4566`
  - `sparsity_only = 0.5288`
  - `task_only = 0.4744`

Вывод:
- абляция стала корректной механистически (без fallback-артефакта);
- `q_err` вклад остаётся частично неотделённым от compactness/sparsity (full не доминирует над sparsity_only).
---

## 5) Statistical evidence (ANFIS main)

Файл:
- `results/significance_sml2010_ea_vs_vanilla_auc_gap.json`

Ключевые числа:
- `delta_mean = 0.7778`
- `CI95 = [0.6408, 0.9149]`
- `wins/losses = 10/0`
- `wilcoxon_p = 0.001953`
- `ttest_p = 1.46e-06`
- `cohen_d = 3.518`

Вывод:
- разница ANFIS EAAR vs vanilla по faithfulness статистически сильная.

---

## 6) R² baselines across datasets

Файл:
- `results/methods_compare_multidataset_20260503.md`

Вывод:
- ET/HGB/RF существенно выше ANFIS по R².
- Поэтому claim статьи: **не SOTA accuracy**, а **faithfulness/internal attribution repair**.

---

## 7) Claim boundary (что заявляем / что не заявляем)

Заявляем:
1. EAAR исправляет функционально неверную внутреннюю атрибуцию.
2. Эффект подтвержден на ANFIS и переносится на MLP (регрессия).
3. Есть первичный перенос на классификацию (Covertype subset).

Не заявляем:
1. “EAAR лучше permutation importance как внешнего post-hoc baseline”.
2. “EAAR дает существенный рост точности”.
3. “Метод уже оптимизирован для production-policy (final-policy)”.

---

## 8) Ограничения (честно)

1. Final-policy сейчас слабее по faithfulness из-за quality-gate/fallback.
2. Negative controls пока не дают идеального разрыва между всеми вариантами.
3. Classification-часть подтверждена на subset-уровне (100k/300k), но без full-Covertype.
3.1. Дополнительная проверка `ResMLP + Energy` не дала устойчивого среднего прироста faithfulness.
4. Timing-таблица добавлена; полноценный scaling-блок (N/d sweep) ещё нужен.
5. Deletion-протокол пока без полного ROAR/KAR retraining.
6. SAGE baseline пока не реализован.
7. Нужна явная спецификация статистической единицы сравнения (seed/target/fold).

## 8.1) Reproducibility protocol and timing

Файлы:
- `results/experimental_protocol_sml2010_eaar_q1.md`
- `results/timing_table_q1_20260503.md`
- `results/predictive_baselines_full_metrics_20260503.md`

Что закрыто:
- добавлен явный протокол split/preprocessing/hyperparams/policy/stat-unit;
- добавлена timing-таблица ANFIS/MLP-reg/MLP-cls;
- добавлена полная baseline-таблица `R²/RMSE/MAE`.

---

## 9) Что уже готово для статьи

Готово:
- ANFIS core result (SML2010 non-fast + stability guard)
- baseline-faithfulness mode comparison
- MLP portability regression
- initial classification portability
- statistical evidence для ANFIS main pair
- unified manifest результатов

Файл-манифест:
- `results/results_manifest.json`

---

## 10) Что добить до более сильной Q1-версии

1. Дожать negative controls (gamma/divergence sweeps), чтобы `full` стабильно выигрывал у `random/shuffled/sparsity/task-only`.
2. Time-block sensitivity закрыт (`non-fast 5 seed`); при необходимости расширить horizon/rolling split.
3. Добавить XGB/LGBM baseline (желательно).
4. По возможности: full-Covertype 1-3 seed как финальный scaling-check.

---

## 11) Быстрые формулировки для manuscript

### Short abstract claim
EAAR aligns internal model attribution with error-aware feature impact under masking. It improves internal faithfulness while preserving predictive quality, with evidence on ANFIS and MLP regression and initial multiclass classification transfer.

### Main result sentence (ANFIS SML2010)
Vanilla internal attribution was functionally inconsistent (`AUC gap = -0.4151`), while EAAR internal attribution became functionally consistent (`AUC gap = +0.4666`), approaching the strong external permutation baseline (`+0.5661`).

### Portability sentence (MLP)
On SML2010 MLP, EAAR improved faithfulness (`ΔAUC gap = +1.8721`, +11.50%) with negligible predictive change (`ΔR² = -0.000281`).

### Classification sentence (Covertype subset)
On Covertype multiclass subset, EAAR improved CE-based faithfulness (`ΔAUC gap = +0.0718`, wins 5/0) while preserving accuracy (`ΔAccuracy = +0.001270`); the 300k subset scaling check kept a positive faithfulness gain (`+0.0941`).

---

## 12) Clean commit scope (без мусора)

Коммитить:
- `results/results_manifest.json`
- `results/results_pack_q1_working_20260503.md`
- `results/q1_master_summary_20260503.md`
- `results/covtype_mlp_eaar_vs_vanilla_5seed_20260503.md`
- `results/covtype300k_mlp_eaar_vs_vanilla_3seed_20260503.md`
- `results/significance_covtype_mlp_eaar_vs_vanilla_auc_gap_ce_5seed.json`
- `results/significance_covtype300k_mlp_eaar_vs_vanilla_auc_gap_ce.json`
- `results/timing_table_q1_20260503.md`
- `results/experimental_protocol_sml2010_eaar_q1.md`
- `results/predictive_baselines_full_metrics_20260503.md`
- ключевые summary/md/json/csv, входящие в таблицы статьи

Не коммитить:
- `results/*/*.pt`
- `results/*/*.npy`
- `results/*/*.png`
- сырые директории с большим шумом, если не участвуют в финальных таблицах.

---

## 13) Bottom line

Текущее состояние: **major-revision Q1-oriented package** с честной рамкой claims.

Формула финальной позиции:
- не “мы точнее всех”,
- а “мы делаем внутреннюю важность модели функционально достоверной и переносимой за пределы одного класса моделей/одной задачи”.

---

## 14) Critical blockers before submission (hard gate)

Перед подачей в сильный журнал обязательно закрыть:

1. **Final-policy practical gap (partially fixed)**
   - В старом non-fast отчете final-policy был отрицательным.
   - В policy-refresh final-policy уже `AUC gap > 0`, но значительно слабее `ea-only`.
   - Перед подачей: или делаем final-policy secondary mode в claim, или усиливаем gate и валидируем на non-fast.

2. **Negative controls not conclusive**
   - `sparsity_only` не хуже `full`, `shuffled_q_err` близко к `full`.
   - Это не доказывает специфическую роль `q_err`.
   - Нужна расширенная ablation-матрица:
     - task-only
     - sparsity-only
     - EAAR true-qerr
     - EAAR shuffled-qerr
     - EAAR random-target
     - EAAR frozen-qerr
     - gamma sweep
     - divergence sweep (`KL/JS/L2/cosine`).

3. **ROAR/KAR limitation**
   - Сейчас deletion без retraining.
   - Для Q1/Q2 желательно mini-ROAR на SML2010 (хотя бы 1-2 `k` и 3 seed).

4. **SAGE gap**
   - Если SAGE не делаем, уменьшаем его роль в related work claim.
   - Если делаем, добавляем хотя бы lightweight baseline на SML2010.

5. **Statistics clarity**
   - Прямо прописать единицу статистики:
     - seed-level / target-level / fold-level,
     - откуда `10/0` при `5` seed.

---

## 15) Safe wording for abstract/conclusion

Использовать:
- “initial evidence”
- “preliminary classification evidence”
- “major revision needed for strong-journal submission”

Не использовать:
- “ready for strong journal as is”
- “proved general applicability”
- “production-ready final policy”.
