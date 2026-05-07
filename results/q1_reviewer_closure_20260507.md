# Q1 Reviewer Closure Snapshot (2026-05-07)

Цель: закрыть критичные замечания рецензии кодом и воспроизводимыми прогонами.

## 1) Unfolding baselines (classical/proxy block)

Файлы:
- `results/unfolding_proxies_benchmark_3seed_20260507.md`
- `results/unfolding_proxies_benchmark_3seed_20260507.json`

Методы (best-per-seed, 3 seed):
- `tikhonov_linear`
- `tikhonov_nnls`
- `maxed_like` (entropy-regularized proxy)
- `gravel_like` (iterative multiplicative proxy)

Итог по `R2_w mean±std`:
- `tikhonov_linear`: `-0.140177 ± 0.013755`
- `tikhonov_nnls`: `-0.135416 ± 0.032571`
- `maxed_like`: `-0.208955 ± 0.015368`
- `gravel_like`: `-0.363986 ± 0.030166`

Вывод:
- classical/proxy unfolding block добавлен и воспроизводим;
- на текущем датасете этот блок существенно слабее основной ANFIS-линии по качеству восстановления.

## 2) SHAP semantics sensitivity sweep

Файлы:
- `results/shap_semantics_sweep_q1_s40_seed42_20260507.md`
- `results/shap_semantics_sweep_q1_s40_seed42_20260507.json`

Сетка:
- `shap_baseline_mode`: `feature_mean | median | zero`
- `shap_value_function`: `mean_output | sum_output | l2_output`
- всего `9` комбинаций (short40 config, seed 42)

Диапазоны по sweep:
- `R2_w`: `0.808787 ... 0.808879`
- `Sel AUC gap`: `0.064011 ... 0.064067`
- `Sel top/random`: `1.029898 ... 1.030638`

Вывод:
- метод устойчив к выбору baseline/value-function в проверенной сетке;
- sensitivity-block добавлен и закрыт количественно.

## 3) Что закрыто относительно рецензии

1. Формальная спецификация SHAP-цели (`baseline mode`, `value function`) теперь явная в коде и конфиге.
2. Есть sensitivity sweep по SHAP-семантике.
3. Есть отдельный классический unfolding baseline-блок (включая MAXED/GRAVEL-like proxy).
4. Скрипты запускаются напрямую (`python scripts/...`) без `PYTHONPATH` костылей.
5. Внутренняя важность переведена на `task-loss gradient` режим и теперь проходит deletion-тест на multi-seed.

## 4) Internal-importance fix (target-aware gradient mode)

Файлы:
- `results/internalfix_ms120_3seed_summary_20260507.md`
- `results/faithfulness_internalfix_ms120_s42_internal.json`
- `results/faithfulness_internalfix_ms120_s43_internal.json`
- `results/faithfulness_internalfix_ms120_s44_internal.json`

Ключ:
- `AUC gap Internal mean = +0.015399` (std `0.003545`)
- `AUC gap Vanilla-importance mean = -0.004065` (std `0.003574`)
- `Wins Internal vs Vanilla-importance = 3/3`
- `Top/random Internal > 1.0` во всех `3/3` запусках

Вывод:
- критичный пункт рецензии по deletion-тесту внутренней карты закрыт практическим кодовым фиксом;
- внутренняя важность стала функционально валидной на multi-seed.

## 5) Остаточные ограничения (честно)

1. MAXED/GRAVEL реализованы как proxy-версии, не как полные reference-реализации исторических пакетов.
2. Sensitivity sweep выполнен на `short40` контуре (оперативный Q1 validation loop).
3. Для строгого Q1/Q2 можно расширить sweep до multi-seed.
