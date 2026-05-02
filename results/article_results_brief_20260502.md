## EA-minimal: итоговое описание для статьи (черновик, без текста статьи)

Дата фиксации: `2026-05-02`.

### 1) Метод (финальный)

Используется только базовый контур:

```text
L = L_main + γ D(p, q_err)
```

где `q_err` — важность признака через прирост ошибки при маскировании, `p` — внутренняя чувствительность ANFIS.

Расширения (hinge-rank, ranknet, teacher/prior) проверялись отдельно и **не включены** в основной метод.

---

### 2) Данные и очистка

Воспроизводимые скрипты подготовки:

```bash
/home/lebedeffson/Code/venv_cuda/bin/python scripts/prepare_sml2010.py
/home/lebedeffson/Code/venv_cuda/bin/python scripts/prepare_naval_propulsion.py
/home/lebedeffson/Code/venv_cuda/bin/python scripts/download_energy_efficiency.py
```

Результат:
- `data/sml2010.csv`: `4137 x 24`
- `data/naval_propulsion.csv`: `11934 x 14` (убраны константы/дубликаты/почти коллинеарные)
- `data/energy_efficiency.csv`: `768 x 10`

---

### 3) Главный датасет статьи: SML2010

#### Fast multi-seed (10 seed)
Файл: `results/multiseed_config_sml2010_ea_minimal_sml_ea10_ckpt.json`

- `ΔR² mean = +7.206e-06`
- `wins/losses = 10/0`
- `metrics_source: shap 10/10`

Faithfulness (permute deletion):
- EA: `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea10_ckpt_permute_final_v2.json`
- Vanilla: `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea10_ckpt_permute_vanilla_v2.json`

Ключ:
- Vanilla: `top < random < bottom`
- EA: `top > random > bottom`

#### Non-fast подтверждение (5 seed, random_trials=20)
Файл: `results/multiseed_config_sml2010_ea_minimal_sml_ea5_full.json`

- `ΔR² mean = +0.001056` (смешанный policy output)
- `metrics_source`: `shap 2/5`, `vanilla_fallback 3/5`

Faithfulness:
- EA-only: `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea5_full_permute_shap_v3.json`
- Vanilla-only: `results/explainability_multiseed_config_sml2010_ea_minimal_sml_ea5_full_permute_vanilla_v3.json`

`AUC_gap(top-bottom)`:
- EA-only: `+0.4666`, CI95 `[+0.3234, +0.6099]`
- Vanilla-only: `-0.4151`, CI95 `[-0.5574, -0.2727]`

Вывод по SML2010:
- точность сохраняется (без существенного роста),
- ранжирование важности переходит из функционально неверного в корректное.

---

### 4) Дополнительные датасеты (non-fast, 3 seed, random_trials=20)

#### Energy Efficiency
- Multi-seed: `results/multiseed_config_energy_ea_minimal_energy_ea3_full.json`
  - `ΔR² mean = +0.000199`
  - `wins/losses = 2/1`, `shap 2`, `fallback 1`
- Faithfulness:
  - EA-only (`..._permute_shap_v3.json`): `AUC_gap = +2.2557`, `top/random = 1.337`
  - Vanilla (`..._permute_vanilla_v3.json`): `AUC_gap = -0.6279`, `top/random = 0.725`

#### Naval Propulsion
- Multi-seed: `results/multiseed_config_naval_ea_minimal_naval_ea3_full.json`
  - `ΔR² mean = -2.93e-05`
  - `wins/losses = 1/2`, `shap 1`, `fallback 2`
- Faithfulness:
  - EA-only (`..._permute_shap_v3.json`): `AUC_gap = +0.1426`, `top/random = 1.367`
  - Vanilla (`..._permute_vanilla_v3.json`): `AUC_gap = +0.0037`, `top/random = 1.267`

Интерпретация:
- Energy: сильный faithfulness-эффект, качество без существенной просадки.
- Naval: quality-gain не подтвержден, но есть улучшение faithfulness относительно vanilla.

---

### 5) Bonner (контрольный прогон актуального пайплайна)

Файл: `results/multiseed_config_integrated_shap_v2_1_light_candidate_bonner_v21_fast1.json`

- Режим: fast 1-seed
- `ΔR² = -0.0434`, `metrics_source = vanilla_fallback`

Вывод:
- для Боннера текущий heavy SHAP-контур требует отдельной настройки; в текущем виде quality guard откатывает к vanilla.
- это не влияет на основной инженерный тезис EA-minimal по SML/Energy/Naval.

---

### 6) Что заявлять в статье

Корректный тезис:

> EA-SHAP-регуляризация не заявляется как сильный бустер точности. Основной эффект — функциональная достоверность и компактность важности: признаки с высокой важностью действительно сильнее влияют на ошибку при удалении (`top > random > bottom`).

Не заявлять:
- “существенный рост R²”
- “рост стабильности между seed”
- “ранговые штрафы как основной метод”

---

### 7) Ответ на жёсткую рецензию и план до Q1

Короткий ответ:

> Критика по baseline/ablation/статистике/масштабируемости принята.  
> Текущая версия — сильный proof-of-concept для 2.3.5 (Поволжье), но для Q1 нужна расширенная валидация.

Что уже закрыто:
- Добавлено сравнение по `R²` с внешними baseline (`RF`, `ET`, `HGB`, `MLP`) на SML2010 / Energy / Naval:
  - `results/methods_compare_multidataset_20260503.md`
- Добавлен baseline-faithfulness (deletion top/random/bottom):
  - `results/faithfulness_baselines_vs_ea_20260503.md`

Что остается обязательно добить для Q1:

1. **Жесткая экспериментальная архитектура**
   - В каждом run хранить: `model_mode`, `fallback_used`, `metrics_source`, `effective_config`, `config_hash`, `seed`, `split_hash`.

2. **Full non-fast multi-seed**
   - SML2010: 10 seed (обязательно)
   - Energy: 10 seed (желательно)
   - Naval: 5–10 seed (ограничительный кейс)

3. **Ablation по EA-компонентам**
   - `EA train-target`, `EA val-target`, `без EMA`, `без grad balancing`, `full EA`.
   - Метрики: `R²`, `AUC_gap(top-bottom)`, `AUC top/random`, `N_eff`, `Mass@3`, fallback-rate.

4. **Статистическая доказательность**
   - `mean±std`, `median`, `CI95`, `wins/losses`, `Wilcoxon`, effect size.
   - Главный тест: `AUC_gap` у vanilla < 0, у EA > 0, разница статистически стабильна.

5. **Insertion**
   - Либо чинить формализацию и знак gain, либо исключить из основного тезиса и оставить deletion-only.

6. **Масштабируемость**
   - Оценка вычислительной сложности и wall-time.
   - Отдельный эксперимент на более крупном датасете (по возможности >50K объектов).

7. **Воспроизводимость Q1-уровня**
   - `requirements/environment`, `run_all_experiments.sh`, `results_manifest.json`, фиксированные seeds/splits.

Позиционирование для подачи:
- **Сейчас:** Поволжье / 2.3.5 (готово почти полностью).
- **Q1-версия:** после расширенного baseline+ablation+statistics+scaling.
