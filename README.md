# ANFIS Bonner Spectra Reconstruction

Проект по восстановлению энергетических спектров нейтронов по показаниям многошарового спектрометра Боннера с использованием ANFIS (Adaptive Neuro-Fuzzy Inference System) и SHAP-регуляризации. Система включает устойчивое обучение, автоматическую генерацию графиков и формирование отчётов для инженеров (PDF и Word).

---

## Основные возможности
- Мультирегрессия на 60 энергетических бинов с нормализацией/денормализацией по сумме сигналов (`SUM`).
- Двухэтапное обучение: vanilla ANFIS (PSO) → SHAP-регуляризация с градиентным клиппингом.
- Контроль NaN/Inf коэффициентов и подробные диагностические сводки.
- Визуализация спектров, ошибок и важности признаков в инженерном стиле (логарифмическая ось, ступенчатые графики).
- Генерация Markdown, PDF и DOCX отчётов по результатам.
- Набор unit-тестов и интеграционных тестов.

---

## Структура репозитория
```
ОИЯИ-ШАБД/
├── README.md                  # Этот файл
├── requirements.txt           # Зависимости (Python 3.10+)
├── configs/                   # Конфигурации обучения
├── src/                       # Исходники модели и утилиты
│   ├── models/
│   │   ├── anfis_manager.py
│   │   └── shap_trainer.py
│   └── utils/
│       ├── data_loader.py
│       └── config_loader.py
├── train.py                   # Основной CLI для обучения
├── analyze.py                 # Анализ JSON-сводки
├── plot_results.py            # Построение графиков и отчётов
├── generate_engineering_report.py  # Итоговый Word-отчёт
├── run_all_tests.py           # Быстрый прогон тестов
├── results/                   # Артефакты (в .gitignore)
└── docs/
    └── ЧижовК_статья_в_Системный_анализ_в_науке_и_образовании_2025.pdf
```

---

## Данные
- `normalized_linear_data_with_q_500K.csv` – синтетические спектры (≈500k примеров, FRUIT).
- `normalized_data_with_q_375.csv` – референсные реальные спектры (МАГАТЭ и др.).
- Каждая запись: 10 входных сигналов `Q1…Q10`, 60 выходов спектра в диапазоне `10^-3…10^8` эВ.
- Нормализация: входы и выходы делятся на `SUM = ΣQi`. Денормализация выполняется автоматически при сохранении предсказаний.

---

## Установка и запуск
```bash
git clone <repo-url>
cd ОИЯИ-ШАБД
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Мини-прогон для проверки
```bash
python train.py --config configs/config_debug.yaml --tag debug_check
python plot_results.py --summary results/training_summary_<timestamp>_debug_check.json
```

### Полный прогон
```bash
python train.py --config configs/config.yaml --tag full_blend_shap
python plot_results.py --summary results/training_summary_<timestamp>_full_blend_shap.json \
  --output-dir results/plots_full_blend_shap --report-file report_full_blend_shap.md
python generate_engineering_report.py --summary results/training_summary_<timestamp>_full_blend_shap.json
```

---

## Основные CLI-инструменты
- `train.py` – обучение модели. Ключевые опции:
  - `--config` – путь к YAML-конфигурации;
  - `--train-limit`, `--train-fraction` – подвыборка для ускоренных прогонов;
  - `--tag` – суффикс для артефактов.
- `plot_results.py` – построение спектров, ошибок, метрик, истории SHAP; формирует Markdown-отчёт.
- `analyze.py` – компактный консольный анализ JSON сводки.
- `generate_engineering_report.py` – создаёт Word-отчёт со всеми графиками.
- `run_all_tests.py` – прогон unit-тестов (`pytest`).

---

## Настройка обучения
Основные параметры задаются в `configs/config.yaml` или `configs/config_debug.yaml`:
- `model.num_rules` – число нечетких правил ANFIS.
- `model.vanishing_strategy` – стратегия устранения затухания (по умолчанию `blend`).
- `optim_params.epoch`, `optim_params.pop_size` – настройки PSO.
- `shap_reg.enabled`, `shap_reg.train_samples`, `shap_reg.lr`, `shap_reg.grad_clip` – параметры SHAP-этапа.
- `dataset.mix_with_real`, `mix_ratio` – смешивание реальных спектров.
- `output.save_samples`, `output.sample_size` – сохранение спектров для отчётов.

---

## Результаты последнего полного прогона
`2025-11-08, tag=full_blend_shap`  
Нормализованные метрики:

| Metric        | Value  |
|---------------|--------|
| MSE           | 0.0117 |
| RMSE          | 0.1083 |
| MAE           | 0.0402 |
| R2_weighted   | 0.5489 |
| R2_mean       | 0.5203 |

разбивка по диапазонам (норм.):  
`bins 0–19 → RMSE 0.079, R² 0.605`  
`bins 20–39 → RMSE 0.105, R² 0.496`  
`bins 40–59 → RMSE 0.133, R² 0.460`

Полный набор метрик и артефактов находится в `results/training_summary_20251108_213257_full_blend_shap.json` и `results/plots_full_blend_shap/`.

---

## Артефакты после запуска
- JSON сводка обучения (`training_summary_*.json`)
- Сохранённое состояние модели (`anfis_model_state_*.pt`)
- Предсказания (норм/денорм) и выборка спектров (`*.npy`)
- Графики спектров, ошибок, важностей (`results/plots_<tag>/...`)
- Markdown-отчёт (`report_<tag>.md`), инженерный PDF и финальный DOCX.

---

## Тестирование и контроль качества
```bash
pytest
```
Тесты покрывают загрузку данных, расчёт метрик и базовую интеграцию. Перед публикацией рекомендуется прогнать `run_all_tests.py` и убедиться в отсутствии `NaN/Inf` в коэффициентах (информация выводится в JSON-сводке).

---

## Дальнейшие задачи
- Сформировать таблицу экспериментов (`mix_ratio`, `num_rules`, `reg_lambda`) для отчёта.
- Добавить CLI-режим «fast train» с фиксированными параметрами подвыборки.
- Расширить сравнение с литературой (например, оценка доз `Ḣiso`).
- Подготовить CI/CD сценарий для автоматической генерации отчётов.

---

## Ссылки и благодарности
- К.А. Чижов и др., *Системный анализ в науке и образовании*, 2025 – исходная публикация.
- Ю. Трофимов – постановка задачи и идея нормализации на сумму сигналов.

Лицензия: проект распространяется для научного использования; условия можно уточнить перед открытой публикацией.

