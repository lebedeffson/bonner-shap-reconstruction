# EAAR: One-File Article Pack (2026-05-03)

Единый файл для текста статьи: claim, ключевые результаты, протокол, ограничения, артефакты.

---

## 1) Главный claim (без оверклейма)

**EAAR (Error-Aware Attribution Regularization)**:

\[
L = L_{task} + \gamma D(p_\theta, q_{err})
\]

- Мы **не** заявляем SOTA по точности.
- Мы **не** заявляем, что EAAR лучше external permutation importance.
- Мы заявляем: EAAR делает **internal attribution функционально корректной** (по deletion-faithfulness).

---

## 2) Core result: ANFIS, SML2010 (random split, non-fast, 5 seeds, stability-guard)

Источник: `results/sml2010_faithfulness_mode_compare_stability_guard5_20260503.md`

| Mode | AUC gap | Top/random |
|---|---:|---:|
| Vanilla gradient | -0.4151 | 0.3687 |
| Vanilla permutation | +0.5661 | 2.6381 |
| EAAR internal (ea-only) | +0.4666 | 2.1792 |
| EAAR final policy (old quality-gated) | -0.1739 | 0.7763 |

Дополнительно:
- unstable runs: `1` (seed `45`)
- stable-only `ΔR² = -0.000087`

Вывод: EAAR internal чинит знак faithfulness; permutation все еще сильнее; старый final-policy ломал practical story.

---

## 3) Final-policy refresh (faithfulness-aware gate)

Источник: `results/sml2010_faithfulness_mode_compare_policy_refresh5_20260503.md`

- `EAAR final policy AUC gap = +0.0161`
- `Top/random = 1.1585`
- stable-only `ΔR² ≈ +0.000003`

Вывод: отрицательный final-gap снят, но final-policy остается слабее `ea-only`.

---

## 4) Time-block robustness (SML2010)

### 4.1 Non-fast 5 seeds (главный robustness check)
Источник: `results/sml2010_faithfulness_mode_compare_timeblock5_nf_20260503.md`

| Mode | AUC gap |
|---|---:|
| Vanilla gradient | -0.2913 |
| Vanilla permutation | +0.4113 |
| EAAR internal | +0.3248 |
| EAAR final policy | +0.0724 |

Итог:
- unstable runs: `0`
- `ΔR² mean = -0.000155` (wins/losses `0/5`, fallback `0.40`)

### 4.2 Non-fast 3 seeds (подтверждение)
Источник: `results/sml2010_faithfulness_mode_compare_timeblock3_nf_20260503.md`

| Mode | AUC gap |
|---|---:|
| Vanilla gradient | -0.2953 |
| Vanilla permutation | +0.3935 |
| EAAR internal | +0.3109 |
| EAAR final policy | +0.3109 |

Вывод: паттерн стабилен на time-block split.

---

## 5) Portability: MLP regression (SML2010, 5 seeds)

Источник: `results/mlp_eaar_vs_vanilla_sml2010_5seed_20260503.md`

| Metric | Value |
|---|---:|
| ΔR² mean | -0.000281 |
| Vanilla AUC gap mean | 16.2754 |
| EAAR AUC gap mean | 18.1475 |
| ΔAUC gap | +1.8721 (+11.50%) |
| Faithfulness wins/losses | 4/1 |

Significance (`results/significance_mlp_eaar_vs_vanilla_auc_gap.json`):
- 95% CI: `[0.4380, 3.3062]`
- Wilcoxon p: `0.1250` (small N, trend-level)
- Cohen's d: `1.1442`

---

## 6) Portability: MLP classification (Covertype)

### 6.1 Covertype-100k, 5 seeds
Источник: `results/covtype_mlp_eaar_vs_vanilla_5seed_20260503.md`

| Metric | Value |
|---|---:|
| ΔAccuracy mean | +0.001270 |
| ΔMacro-F1 mean | -0.001682 |
| ΔCE mean | -0.001051 |
| ΔAUC gap (CE) mean | +0.0718 |
| Faithfulness wins/losses | 5/0 |
| CI95(ΔAUC gap CE) | [0.0384, 0.1052] |

Per-class / class-balance:
- `results/covtype_per_class_full_5seed_20260504.md`
- `ΔBalanced accuracy = -0.003970`
- worst class `ΔF1 = -0.010755` (class 3)

### 6.2 Covertype-300k, 3 seeds
Источник: `results/covtype300k_mlp_eaar_vs_vanilla_3seed_20260503.md`

| Metric | Value |
|---|---:|
| ΔAccuracy mean | -0.000211 |
| ΔMacro-F1 mean | -0.000717 |
| ΔCE mean | +0.002098 |
| ΔAUC gap (CE) mean | +0.0941 |
| Faithfulness wins/losses | 3/0 |

Вывод: классификация — **initial evidence** (faithfulness +, predictive near-flat).

---

## 6.3) Extra portability check: ResMLP regression (Energy, 5 seeds)

Источник: `results/resmlp_eaar_vs_vanilla_energy_5seed_20260504.md`

| Metric | Value |
|---|---:|
| ΔR² mean | -0.002107 |
| Vanilla AUC gap mean | 148.1323 |
| EAAR AUC gap mean | 144.9934 |
| ΔAUC gap mean | -3.1389 |
| Faithfulness wins/losses | 3/2 |
| CI95(ΔAUC gap) | [-21.6565, 15.3787] |
| Wilcoxon p | 1.0000 |

Вывод: это честный **boundary-case**. На паре ResMLP+Energy перенос EAAR смешанный и не даёт устойчивого среднего прироста faithfulness.

---

## 7) Negative controls (главный незакрытый блок)

Источник: `results/ablation_neg3_nonfast_controls_20260503.md`

| Variant | AUC gap |
|---|---:|
| full | 0.4992 |
| random_target | 0.4369 |
| shuffled_q_err | 0.4798 |
| sparsity_only | 0.5077 |

Интерпретация для статьи:
- random хуже full (полезно),
- но sparsity-only не хуже full, shuffled близко к full,
- значит специфическая роль `q_err` еще не доказана жестко.

Дополнительные fast-completion прогоны:
- `results/ablation_neg_core_v2_fast3_final.md`
- `results/ablation_gamma_sweep_v2_fast3_final.md`
- `results/ablation_neg_core_v2_fast3_final_alignment_20260504.md`

Замечание:
- в fast-режиме варианты почти совпали, поэтому эти прогоны используем как sanity/stability check, а не как главный механизмный аргумент.

---

## 8) Predictive baselines (R²/RMSE/MAE)

Источник: `results/predictive_baselines_full_metrics_20260503.md`

Коротко:
- На всех 3 регрессионных датасетах ансамбли (`ET/HGB/RF`) выше ANFIS по R².
- Поэтому позиционирование корректное: **faithfulness/internal repair**, не accuracy-SOTA.

---

## 9) Timing / compute cost

Источник: `results/timing_table_q1_20260503.md`

| Task | Model | Dataset | Vanilla sec | EAAR sec | Overhead |
|---|---|---|---:|---:|---:|
| Regression | ANFIS | SML2010 | 2.7114 | 13.4471 | 4.9600x |
| Regression | MLP | SML2010 | 3.8423 | 23.8376 | 6.2041x |
| Classification | MLP | Covertype-100k | 18.2091 | 112.7511 | 6.1920x |

\[
O(E \cdot B \cdot d \cdot C_f)
\]

---

## 10) Experimental protocol / hyperparams (для reproducibility)

Основной ANFIS EAAR config: `configs/config_sml2010_ea_minimal.yaml`  
Протокол: `results/experimental_protocol_sml2010_eaar_q1.md`

### 10.1 ANFIS + EAAR (SML2010)
- num_rules: `12`
- mf_class: `Gaussian`
- reg_lambda: `0.01`
- optimizer: `OriginalPSO`, pso_epochs/pop: `80/60`
- epochs: `30`, batch_size: `64`, lr: `8e-05`
- gamma: `0.001`
- divergence/alignment: `cosine_mse` (alpha `0.5`)
- masking: `permute`
- q_err target scope: `train`
- q_err positive clipping: `True`
- policy mode: `quality_and_faithfulness`
- acceptance_min_delta_r2: `-0.001`
- auc_gap_margin: `0.02`
- min_ea_auc_gap: `0.0`
- min_top_random_ratio: `1.0`

### 10.2 MLP regression (SML2010)
- model: `Linear(19->128) -> ReLU -> Dropout(0.1) -> Linear(128->64) -> ReLU -> Linear(64->2)`
- optimizer: Adam, lr `0.001`
- epochs `140`, batch_size `128`
- EAAR: gamma `0.05`, masking `permute`, alignment `cosine_mse`, alpha `0.5`, warmup `0.25`

### 10.3 MLP classification (Covertype)
- model: `Linear(54->128) -> ReLU -> Dropout(0.1) -> Linear(128->64) -> ReLU -> Linear(64->7)`
- optimizer: Adam, lr `0.001`
- epochs `50`, batch_size `512`
- EAAR: gamma `0.02`, masking `permute`, alignment `cosine_mse`, alpha `0.5`, warmup `0.25`, score `logprob`, grad combine `projected`
- summary now logs: `per_class` precision/recall/f1/support + `balanced_accuracy`

### 10.4 Split protocol
- ANFIS SML2010 pipeline: train/val/test = `0.6/0.2/0.2`
- MLP pipelines: train/test = `0.8/0.2` (stratified for classification)

### 10.5 Runtime/repro metadata in summaries
- `runtime`: device/cpu/gpu/cuda/torch/python/ram
- `config_sha256`, `effective_config_sha256`
- `split_hash`, `data_hash` (or `data_hashes` for ANFIS pipeline)
- `model_mode`, `fallback_used`, `unstable_prediction_flag`

---

## 11) Dataset sizes used in this package

| Dataset | Task | N used | Features | Outputs/classes | Split used |
|---|---|---:|---:|---:|---|
| SML2010 | Regression | 4137 | 19 | 2 outputs | ANFIS: 60/20/20; MLP: 80/20 |
| Covertype-100k | Classification | 100000 | 54 | 7 classes | 80/20 stratified |
| Covertype-300k | Classification | 300000 | 54 | 7 classes | 80/20 stratified |

Derived test sizes:
- SML2010 MLP: ~`3309/828` train/test
- Covertype-100k: `80000/20000`
- Covertype-300k: `240000/60000`

---

## 12) Statistics unit (важно для рецензента)

- Основная единица сравнения в большинстве multiseed-таблиц: **seed-level paired**.
- В `significance_sml2010_ea_vs_vanilla_auc_gap.json` (`wins/losses=10/0`) использованы **10 пар target-level наблюдений** (5 seeds × 2 outputs), поэтому N в тесте больше числа seeds.

---

## 13) Что осталось слабым (честно)

1. **Negative controls (главный риск интерпретации).**  
   `sparsity-only (0.5077)` не хуже `full (0.4992)`, а `shuffled_q_err (0.4798)` близко к `full`.  
   Следствие: сейчас нельзя утверждать, что эффект единолично объясняется `q_err`; корректно говорить про совместный вклад error-aware сигнала и compactness.

2. **Final-policy слабее `ea-only`.**  
   Отрицательный gap уже снят (policy-refresh/time-block), но final-policy остается заметно слабее internal `ea-only` режима.  
   Следствие: главный научный результат — `EAAR internal`; final-policy — вторичный практический режим.

3. **Нет полного ROAR/KAR и SAGE baseline.**  
   Это явно ограничивает силу causality-claim по faithfulness в сильной международной подаче.

4. **Classification пока subset-level.**  
   Есть 100k/300k evidence, но нет full-Covertype прогона.

5. **ResMLP+Energy boundary-case.**  
   На этой паре перенос EAAR не дал устойчивого среднего улучшения faithfulness.

---

## 14) Что писать в статье безопасно

Использовать:
- “initial evidence”
- “preliminary classification evidence”
- “major-revision package for strong-journal submission”

Не использовать:
- “EAAR better than permutation baseline”
- “production-ready final policy”
- “proved general applicability”

---

## 15) Вердикт по готовности

- Для **РИНЦ/прикладного журнала**: пакет уже адекватный для отправки.
- Для **сильного Q1/Q2**: нужна добивка (expanded ablation/gamma-divergence sweep, mini-ROAR/KAR, SAGE или снижение claim, по возможности full-Covertype).

---

## 16) Перед отправкой: ручной чек

1. Заполнить: `Поступила в редакцию __.__.2026`.
2. Раздел «Благодарности/финансирование»: заполнить или удалить.
3. Проверить email ответственного автора (если требует шаблон журнала).
4. Проверить требования журнала к английскому title/abstract.

---

## 17) Main artifact index

- Master summary: `results/q1_master_summary_20260503.md`
- Working pack: `results/results_pack_q1_working_20260503.md`
- Manifest: `results/results_manifest.json`
- Protocol: `results/experimental_protocol_sml2010_eaar_q1.md`
- Timing: `results/timing_table_q1_20260503.md`
- Baselines: `results/predictive_baselines_full_metrics_20260503.md`
- ResMLP Energy: `results/resmlp_eaar_vs_vanilla_energy_5seed_20260504.md`
