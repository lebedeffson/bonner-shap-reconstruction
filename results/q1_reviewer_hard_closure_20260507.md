# Q1 Hard Closure (2026-05-07)

Ниже закрытие критичных замечаний рецензии с новыми кодовыми артефактами.

## 1) "Нет сравнения с другими нейросетевыми/ML-анфолдинг моделями"

Закрыто добавлением новых baseline-блоков:

- `results/neural_unfolding_benchmark_3seed_20260507.md`
  - `MLP`: `R2_w = 0.681394 ± 0.032670`
  - `1D-CNN`: `R2_w = 0.721035 ± 0.047493`
  - `MC_Dropout_MLP` (bayesian-like proxy): `R2_w = 0.703123 ± 0.042156`
- `results/ml_baselines_benchmark_3seed_20260507.md`
  - `ExtraTrees`, `RF`, `HGB`, `MLPRegressor`
- `results/unfolding_proxies_benchmark_3seed_20260507.md`
  - `tikhonov_linear`, `tikhonov_nnls`, `maxed_like`, `gravel_like`

Контекст основной SHAP-линии:
- `results/gapaware_ms120_3seed_summary_20260507.md`
  - `R2_w SHAP = 0.787623 ± 0.057090`
  - `AUC gap SHAP = 0.011759 ± 0.001783`

Вывод:
- сравнение с нейросетями и классическими proxy-анфолдинг методами теперь есть на одинаковом split-протоколе.

## 2) "Недоопределенность данных и протокола"

Закрыто:
- `results/dataset_protocol_bonner_20260507.md`
  - `N=375`
  - `features=10`, `bins=60`
  - `Train/Val/Test = 225/75/75`
  - `unique spectra = 374`, `duplicate spectra = 1`
  - `metadata columns outside feature/target set = 0`

Вывод:
- протокол и размерности теперь зафиксированы явно отдельным артефактом.

## 3) "Скаляризация и baseline в SHAP-недоопределены"

Закрыто:
- формализация в коде:
  - `shap_baseline_mode`: `feature_mean|median|zero`
  - `shap_value_function`: `mean_output|sum_output|l2_output`
- sensitivity sweep:
  - `results/shap_semantics_sweep_q1_s40_seed42_20260507.md`
  - диапазоны:
    - `R2_w: 0.808787 ... 0.808879`
    - `Sel AUC gap: 0.064011 ... 0.064067`
    - `Sel top/random: 1.029898 ... 1.030638`

Вывод:
- выбор baseline/scalarization перестал быть "скрытым"; чувствительность количественно проверена.

## 4) "Внутренняя важность проваливает deletion"

Статус: **частично закрыто / честно ограничено**.

Из `results/gapaware_ms120_3seed_summary_20260507.md`:
- `AUC gap Internal mean = 0.001677` (выше vanilla `-0.002266`, но существенно ниже SHAP-ветки `0.011759`)
- `top/random Internal mean = 0.392961` (слабее SHAP-ветки)

Вывод:
- внутренняя карта улучшена относительно vanilla по знаку gap, но пока не дотягивает до внешней SHAP-ветки.
- в статье корректно позиционировать как SHAP-distillation/consistency regularization, а не как "полностью самодостаточная внутренняя интерпретируемость".

## 5) "Дозовая метрика не вычислена"

Закрыто минимально:
- `results/dose_metrics_proxy_20260507.md`
  - `Dose MAPE mean = 0.074625`
  - `Dose Pearson corr = 0.968476`

Важно:
- в текущем файле использован `flat_ones_proxy` (интегральный флюенс proxy), так как в репозитории нет внешнего файла физ. коэффициентов дозопреобразования.
- как только будут утвержденные коэффициенты, пересчет делается тем же скриптом:
  - `scripts/report_dose_metrics.py --coeffs <path_to_coeffs>`

## 6) "Нет анализа масштабируемости"

Закрыто:
- `results/shap_scaling_report_20260507.md`
- есть run с реальными счетчиками exact-SHAP:
  - `tag=q1_shap_compute_smoke`
  - `exact_calls=9`
  - `exact_total_utility_evals=9216`
  - `training_time_total=23.62s`, `training_time_shap=22.97s`

Вывод:
- вычислительная сложность и overhead теперь подтверждаются реальными логами исполнения, а не только декларативно.

---

## Итоговое состояние

Что закрыто сильно:
1. сравнение с нейросетевыми baseline;
2. сравнение с classical/proxy unfolding baseline;
3. протокол данных и split-таблица;
4. SHAP semantics sensitivity;
5. dose-like scalar block (proxy + готовый путь к физическим коэффициентам);
6. scaling/runtime diagnostics для exact SHAP.

Что остается ограничением:
1. внутренняя importance-ветка пока слабее SHAP-ветки по functional faithfulness;
2. dose-метрика пока proxy до подключения официальных dose conversion coefficients.
