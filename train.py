#!/usr/bin/env python3
"""Обучение ANFIS модели и сохранение результатов"""

import argparse
import json
import os
import sys
import copy
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from src.models.anfis_manager import ANFISManager
from src.models.shap_trainer_improved import ShapAwareANFISTrainerImproved as ShapAwareANFISTrainer
from src.utils.config_loader import load_config
from src.utils.data_loader import (
    load_training_dataset,
    prepare_features_targets,
    split_data,
    denormalize_predictions
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


REAL_TEST_FRACTION = 0.2
REAL_VALIDATION_FRACTION_OF_TEMP = 0.25
REAL_DATA_SPLIT = {
    'train': 0.6,
    'validation': 0.2,
    'test': 0.2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение ANFIS модели для восстановления спектра")
    parser.add_argument("--config", default="configs/config_integrated_shap.yaml", help="Путь к YAML конфигурации")
    parser.add_argument("--train-limit", type=int, dest="train_limit",
                        help="Переопределяет dataset.train_limit в конфигурации")
    parser.add_argument("--train-fraction", type=float, dest="train_fraction",
                        help="Переопределяет dataset.train_fraction в конфигурации")
    parser.add_argument("--tag", help="Дополнительный суффикс к timestamp (для отладки)")
    return parser.parse_args()


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def _compute_band_metrics(y_true, y_pred, bands):
    if y_true is None or y_pred is None:
        return {}
    if y_true.shape != y_pred.shape:
        return {}

    y_true = np.nan_to_num(np.asarray(y_true, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    metrics = {}
    for name, band_slice in bands:
        if band_slice.stop is not None and band_slice.stop > y_true.shape[1]:
            continue
        y_true_band = y_true[:, band_slice]
        y_pred_band = y_pred[:, band_slice]
        if y_true_band.size == 0 or y_pred_band.size == 0:
            continue
        mse = mean_squared_error(y_true_band, y_pred_band, multioutput='uniform_average')
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_band, y_pred_band, multioutput='uniform_average')
        try:
            r2 = r2_score(y_true_band, y_pred_band, multioutput='uniform_average')
        except ValueError:
            r2 = float('nan')

        metrics[name] = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
    return metrics


def _resolve_output_bands(n_outputs):
    n_outputs = int(n_outputs)
    if n_outputs <= 0:
        return []
    if n_outputs == 60:
        return [
            ("band_0_19", slice(0, 20)),
            ("band_20_39", slice(20, 40)),
            ("band_40_59", slice(40, 60)),
        ]
    if n_outputs <= 3:
        return [("all_outputs", slice(0, n_outputs))]

    step = max(n_outputs // 3, 1)
    bands = [
        ("band_0", slice(0, step)),
        ("band_1", slice(step, min(2 * step, n_outputs))),
        ("band_2", slice(min(2 * step, n_outputs), n_outputs)),
    ]
    return [(name, sl) for name, sl in bands if sl.stop > sl.start]


def _prepare_feature_importance(values, feature_names, *, normalize=False):
    """Санитизирует важности признаков, проверяет размерность и при необходимости нормализует."""
    feature_names = list(feature_names)
    importance = np.asarray(values, dtype=float).reshape(-1)
    importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
    importance = np.maximum(importance, 0.0)

    if importance.size != len(feature_names):
        raise ValueError(
            f"Размер importance ({importance.size}) не совпадает с количеством признаков ({len(feature_names)})"
        )

    if normalize and importance.size > 0:
        total = float(np.sum(importance))
        if not np.isfinite(total) or total <= 1e-12:
            importance = np.full(importance.shape, 1.0 / importance.size, dtype=float)
        else:
            importance = importance / total

    return pd.Series(importance, index=feature_names, dtype=float)


def _summarize_regularization_history(shap_history, shap_config):
    """Извлекает компактную сводку о вкладе SHAP и Tikhonov из истории обучения."""
    if not shap_history:
        return {}

    component_names = ('consistency', 'sparsity', 'faithfulness', 'stability')

    def _stats(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        return {
            'mean': float(np.mean(arr)),
            'last': float(arr[-1]),
            'max': float(np.max(arr)),
        }

    summary = {
        'active_components': list(shap_config.get('active_components', ['consistency', 'sparsity', 'faithfulness', 'stability']))
    }

    for key in [
        'shap_contribution',
        'tikhonov_contribution',
        'nonnegativity_contribution',
        'regularization_share',
        'shap_scale_factor',
        'shap_loss_normalized',
        'nonnegativity_loss',
    ]:
        stats = _stats(shap_history.get(key))
        if stats is not None:
            summary[key] = stats

    component_terms = {}
    component_weights = {}
    weighted_component_signal = {}
    for name in component_names:
        term_key = f'shap_{name}'
        weight_key = f'shap_weight_{name}'

        term_stats = _stats(shap_history.get(term_key))
        if term_stats is not None:
            component_terms[name] = term_stats

        weight_stats = _stats(shap_history.get(weight_key))
        if weight_stats is not None:
            component_weights[name] = weight_stats

        terms = np.asarray(shap_history.get(term_key, []), dtype=float)
        weights = np.asarray(shap_history.get(weight_key, []), dtype=float)
        if terms.size and weights.size:
            size = min(len(terms), len(weights))
            signal = terms[:size] * weights[:size]
            signal = signal[np.isfinite(signal)]
            if signal.size:
                weighted_component_signal[name] = {
                    'mean': float(np.mean(signal)),
                    'last': float(signal[-1]),
                    'max': float(np.max(signal)),
                }

    if component_terms:
        summary['component_terms'] = component_terms
    if component_weights:
        summary['component_weights'] = component_weights
    if weighted_component_signal:
        summary['weighted_component_signal'] = weighted_component_signal
        dominant_name = max(weighted_component_signal.items(), key=lambda item: item[1]['mean'])[0]
        summary['dominant_shap_component'] = dominant_name

    shap_mean = summary.get('shap_contribution', {}).get('mean', 0.0)
    tikh_mean = summary.get('tikhonov_contribution', {}).get('mean', 0.0)
    nonneg_mean = summary.get('nonnegativity_contribution', {}).get('mean', 0.0)
    dominant = max(
        [('shap', shap_mean), ('tikhonov', tikh_mean), ('nonnegativity', nonneg_mean)],
        key=lambda item: item[1]
    )
    if dominant[1] <= 0:
        summary['dominant_regularizer'] = 'balanced'
    else:
        summary['dominant_regularizer'] = dominant[0]

    return summary


def _split_real_data_for_shap(X_real, y_real, SUM_real, *, normalize_sum=False, random_state=42):
    """Разделяет реальные данные на train/val/test и сохраняет выравнивание SUM с test-частью."""
    if normalize_sum and SUM_real is not None:
        X_temp, X_real_test, y_temp, y_real_test, _, SUM_real_test = train_test_split(
            X_real, y_real, SUM_real, test_size=REAL_TEST_FRACTION, random_state=random_state
        )
    else:
        X_temp, X_real_test, y_temp, y_real_test = train_test_split(
            X_real, y_real, test_size=REAL_TEST_FRACTION, random_state=random_state
        )
        SUM_real_test = None

    X_real_shap, X_real_val, y_real_shap, y_real_val = train_test_split(
        X_temp, y_temp, test_size=REAL_VALIDATION_FRACTION_OF_TEMP, random_state=random_state
    )

    return (
        X_real_shap,
        X_real_val,
        X_real_test,
        y_real_shap,
        y_real_val,
        y_real_test,
        SUM_real_test,
    )


def train_and_save(args):
    """Обучает модель и сохраняет артефакты"""

    print("=" * 80)
    print("🤖 ОБУЧЕНИЕ ANFIS МОДЕЛИ (train.py)")
    print("=" * 80)

    config_path = args.config
    print(f"\n⚙️  Конфигурация: {config_path}")
    config = load_config(config_path)
    config_text = Path(config_path).read_text(encoding="utf-8")
    config_sha256 = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    effective_config_sha256 = hashlib.sha256(
        json.dumps(_to_serializable(config), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dataset_config = config['dataset']
    normalize_sum = dataset_config.get('normalize_sum', False)

    if args.train_limit is not None:
        dataset_config['train_limit'] = args.train_limit
        print(f"   ➤ Переопределён dataset.train_limit = {args.train_limit}")

    if args.train_fraction is not None:
        dataset_config['train_fraction'] = args.train_fraction
        print(f"   ➤ Переопределён dataset.train_fraction = {args.train_fraction}")

    # В проекте поддерживается только основной двухэтапный режим:
    # vanilla PSO на train-данных -> SHAP + Tikhonov fine-tune на real train split.
    shap_config = config.get('shap_reg', {})
    run_meta = config.get('_run_meta', {})
    if shap_config.get('integrated_training', False):
        raise ValueError(
            "integrated_training больше не поддерживается в текущем train.py. "
            "Используйте основной two-stage режим с integrated_training=false."
        )
    training_mode = shap_config.get('training_mode', 'real_only')
    
    # Загружаем реальные данные для SHAP обучения и тестирования
    from src.utils.data_loader import load_validation_data
    
    real_data_path = dataset_config.get('validation_data')
    if not real_data_path or not os.path.exists(real_data_path):
        raise FileNotFoundError(f"Файл с реальными данными не найден: {real_data_path}")
    
    print("\n📂 Загрузка реальных данных...")
    X_real, y_real, SUM_real = load_validation_data(
        real_data_path,
        normalize_sum=normalize_sum,
        dataset_config=dataset_config
    )
    
    # Стандартный режим: загружаем синтетические данные для vanilla-этапа.
    print("\n📂 Загрузка обучающих данных...")
    try:
        data = load_training_dataset(dataset_config)
        X, y, SUM_train = prepare_features_targets(
            data, normalize_sum=normalize_sum, dataset_config=dataset_config
        )
        
        print("\n🔀 Разделение данных...")
        X_train, X_test, y_train, y_test = split_data(
            X, y,
            test_size=dataset_config.get('test_size', 0.25),
            random_state=dataset_config.get('random_state', 42)
        )
        
        # Сохраняем SUM для теста, если нужно
        if normalize_sum and SUM_train is not None:
            if hasattr(X_train, 'index'):
                SUM_test = SUM_train.loc[X_test.index].values if hasattr(SUM_train, 'loc') else SUM_train[X_test.index]
            else:
                n_train = len(X_train)
                SUM_test = SUM_train[n_train:]
        else:
            SUM_test = None
    except FileNotFoundError as e:
        if training_mode != 'real_only':
            raise FileNotFoundError(
                f"Синтетические данные не найдены, но режим {training_mode} требует их. Ошибка: {e}"
            )
        print(f"   ⚠️  Синтетические данные не найдены, используем только реальные данные")
        X_train = X_real
        y_train = y_real
        SUM_train = SUM_real
        X_test = X_real
        y_test = y_real
        SUM_test = SUM_real
    
    # Разделяем реальные данные: 60% обучение, 20% валидация, 20% финальный тест
    random_state = dataset_config.get('random_state', 42)
    (
        X_real_shap,
        X_real_val,
        X_real_test,
        y_real_shap,
        y_real_val,
        y_real_test,
        SUM_real_test,
    ) = _split_real_data_for_shap(
        X_real,
        y_real,
        SUM_real,
        normalize_sum=normalize_sum,
        random_state=random_state,
    )
    
    print(f"   ▶️ Обучение (Train): {len(X_real_shap)} реальных образцов (60%)")
    print(f"   ▶️ Валидация (Val): {len(X_real_val)} реальных образцов (20%)")
    print(f"   ▶️ Тестирование (Test): {len(X_real_test)} реальных образцов (20%)")
    
    # Преобразуем в массивы для SHAP обучения
    X_real_shap_array = np.array(X_real_shap) if not isinstance(X_real_shap, np.ndarray) else X_real_shap
    y_real_shap_array = np.array(y_real_shap) if not isinstance(y_real_shap, np.ndarray) else y_real_shap
    
    X_real_shap_array = np.nan_to_num(X_real_shap_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_real_shap_array = np.nan_to_num(y_real_shap_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Преобразуем валидационные данные
    X_real_val_array = np.array(X_real_val) if not isinstance(X_real_val, np.ndarray) else X_real_val
    y_real_val_array = np.array(y_real_val) if not isinstance(y_real_val, np.ndarray) else y_real_val
    X_real_val_array = np.nan_to_num(X_real_val_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_real_val_array = np.nan_to_num(y_real_val_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Преобразуем тренировочные данные в массивы
    X_train_array = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_array = np.nan_to_num(y_train_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Проверяем режим обучения
    shap_config = config.get('shap_reg', {})
    if not shap_config.get('enabled', False):
        raise ValueError("SHAP регуляризация должна быть включена (shap_reg.enabled=true)")
    
    # ДВУХЭТАПНОЕ ОБУЧЕНИЕ: vanilla + SHAP/Tikhonov fine-tune.
    print("\n🛠️  ДВУХЭТАПНОЕ ОБУЧЕНИЕ")
    print("=" * 80)
    
    print("\n🛠️  Этап 1: Обучение базовой ANFIS модели на синтетических данных...")
    manager = ANFISManager(config)
    if hasattr(X_train, 'columns'):
        manager.set_feature_names(X_train.columns)
    results = manager.train_vanilla_model(X_train_array, y_train_array, X_real_val_array, y_real_val_array)

    # Сохраняем веса vanilla-этапа, чтобы откатиться при деградации после SHAP.
    vanilla_state_dict = None
    if hasattr(results.get('model'), 'network') and results['model'].network is not None:
        vanilla_state_dict = copy.deepcopy(results['model'].network.state_dict())

    # Базовые метрики vanilla на финальном test (20% real test) для сравнения с SHAP.
    X_real_test_array = np.array(X_real_test) if not isinstance(X_real_test, np.ndarray) else X_real_test
    y_real_test_array = np.array(y_real_test) if not isinstance(y_real_test, np.ndarray) else y_real_test
    X_real_test_array = np.nan_to_num(X_real_test_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_real_test_array = np.nan_to_num(y_real_test_array, nan=0.0, posinf=0.0, neginf=0.0)

    vanilla_test_predictions = results['model'].predict(X_real_test_array)
    vanilla_test_predictions = manager._sanitize_predictions(
        vanilla_test_predictions,
        reference_shape=y_real_test_array.shape,
        context="vanilla_final_test"
    )
    vanilla_test_metrics = manager._calculate_metrics(y_real_test_array, vanilla_test_predictions)
    
    print("\n🧭 Этап 2: SHAP-регуляризация с улучшенной регуляризацией (4 компонента)...")
    
    # Используем подвыборку для SHAP обучения, если указано
    shap_subset = shap_config.get('train_samples')
    if shap_subset is not None:
        shap_subset = int(shap_subset)
        if shap_subset > 0 and shap_subset < len(X_real_shap_array):
            rng = np.random.default_rng(dataset_config.get('random_state', 42))
            subset_idx = rng.choice(len(X_real_shap_array), size=shap_subset, replace=False)
            shap_X_train = X_real_shap_array[subset_idx]
            shap_y_train = y_real_shap_array[subset_idx]
            print(f"   ▶️ SHAP будет обучаться на подвыборке {shap_subset} образцов")
        else:
            shap_X_train = X_real_shap_array
            shap_y_train = y_real_shap_array
    else:
        shap_X_train = X_real_shap_array
        shap_y_train = y_real_shap_array
    
    shap_trainer = ShapAwareANFISTrainer(
        results['model'],
        config,
        gamma=shap_config.get('gamma', 0.5),
        verbose=True
    )

    teacher_targets_train = None
    teacher_cfg = shap_config.get('teacher_distill', {})
    if teacher_cfg.get('enabled', False):
        n_estimators = int(teacher_cfg.get('n_estimators', 600))
        random_state = int(teacher_cfg.get('random_state', dataset_config.get('random_state', 42)))
        min_samples_leaf = int(teacher_cfg.get('min_samples_leaf', 1))
        max_depth = teacher_cfg.get('max_depth', None)
        if max_depth is not None:
            max_depth = int(max_depth)
        print(
            f"   ▶️ Teacher distillation: ExtraTrees(n_estimators={n_estimators}, "
            f"min_samples_leaf={min_samples_leaf}, max_depth={max_depth})"
        )
        teacher_model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
        )
        teacher_model.fit(shap_X_train, shap_y_train)
        teacher_targets_train = teacher_model.predict(shap_X_train)
        teacher_val_pred = teacher_model.predict(X_real_val_array)
        teacher_val_metrics = manager._calculate_metrics(y_real_val_array, teacher_val_pred)
        print(
            f"   ▶️ Teacher val R²: {teacher_val_metrics.get('r2', float('nan')):.6f} "
            f"(для ориентира SHAP-stage)"
        )
    
    shap_history = shap_trainer.fit(
        shap_X_train,
        shap_y_train,
        epochs=shap_config.get('epochs', 25),
        batch_size=shap_config.get('batch_size', 32),
        lr=shap_config.get('lr', 0.003),
        X_val=X_real_val_array,
        y_val=y_real_val_array,
        y_teacher_train=teacher_targets_train,
    )
    
    results['shap_history'] = shap_history
    
    # Тестирование на ВСЕХ реальных данных
    print("\n🧪 Финальное тестирование на ВСЕХ реальных данных...")
    
    # Предсказания и метрики
    shap_predictions = shap_trainer.predict(X_real_test_array)
    shap_predictions = manager._sanitize_predictions(
        shap_predictions,
        reference_shape=y_real_test_array.shape,
        context="shap"
    )
    
    shap_metrics = manager._calculate_metrics(y_real_test_array, shap_predictions)
    results['vanilla_metrics'] = vanilla_test_metrics
    results['shap_metrics_raw'] = shap_metrics
    
    # Вычисляем важность признаков на данных для SHAP
    shap_importance_data = X_real_shap_array
    
    shap_importance = shap_trainer.get_global_shap_importance(shap_importance_data)
    
    # Проверка на NaN/Inf
    metrics_array = np.array(list(shap_metrics.values()), dtype=float)
    importance_array = np.array(shap_importance, dtype=float)
    if not np.isfinite(metrics_array).all() or not np.isfinite(importance_array).all():
        print("⚠️  SHAP-регуляризация дала некорректные значения (NaN/Inf).")
        raise ValueError("SHAP обучение завершилось с ошибкой: NaN/Inf в метриках")
    
    # Safety gate: если SHAP резко ухудшил R² на финальном test, откатываемся к vanilla.
    shap_config = config.get('shap_reg', {})
    min_delta_r2 = float(shap_config.get('acceptance_min_delta_r2', -0.02))
    vanilla_r2 = float(vanilla_test_metrics.get('r2', float('-inf')))
    shap_r2 = float(shap_metrics.get('r2', float('-inf')))
    use_shap = (shap_r2 - vanilla_r2) >= min_delta_r2

    if use_shap:
        results['predictions'] = shap_predictions
        results['metrics'] = shap_metrics
        results['metrics_source'] = 'shap'
    else:
        print(
            f"⚠️  SHAP отклонен: R² ухудшился с {vanilla_r2:.6f} до {shap_r2:.6f}. "
            f"Порог: {min_delta_r2:+.4f}. Использую vanilla."
        )
        if vanilla_state_dict is not None and hasattr(results.get('model'), 'network'):
            results['model'].network.load_state_dict(vanilla_state_dict)
        results['predictions'] = vanilla_test_predictions
        results['metrics'] = vanilla_test_metrics
        results['metrics_source'] = 'vanilla_fallback'

    results['feature_importance_shap'] = np.asarray(shap_importance, dtype=float)
    results['shap_history'] = shap_history
    results['training_time_shap'] = shap_trainer.training_time
    results['training_time'] += shap_trainer.training_time
    
    # Обновляем тестовые данные на реальные
    y_test_array = y_real_test_array
    X_test_array = X_real_test_array
    SUM_test = SUM_real_test

    output_bands = _resolve_output_bands(y_test_array.shape[1])
    band_metrics_norm = _compute_band_metrics(y_test_array, np.asarray(results['predictions']), output_bands)

    # Денормализация метрик
    metrics_denorm = None
    y_test_denorm = None
    y_pred_denorm = None
    if normalize_sum and SUM_test is not None:
        print("\n🔄 Денормализация предсказаний...")
        y_pred_denorm = denormalize_predictions(results['predictions'], SUM_test)
        # Используем y_real_test_array для денормализации, так как тестируем на реальных данных
        y_test_denorm = denormalize_predictions(y_real_test_array, SUM_test)
        y_pred_denorm = np.nan_to_num(y_pred_denorm, nan=0.0, posinf=0.0, neginf=0.0)
        y_test_denorm = np.nan_to_num(y_test_denorm, nan=0.0, posinf=0.0, neginf=0.0)
        metrics_denorm = manager._calculate_metrics(y_test_denorm, y_pred_denorm)
        results['predictions_denorm'] = y_pred_denorm
        results['metrics_denorm'] = metrics_denorm

    band_metrics_denorm = _compute_band_metrics(
        y_test_denorm if y_test_denorm is not None else None,
        y_pred_denorm if y_pred_denorm is not None else None,
        output_bands
    )

    # Папка результатов и артефакты
    output_config = config.get('output', {})
    results_dir = output_config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = f"{timestamp_base}_{args.tag}" if args.tag else timestamp_base

    save_model = bool(output_config.get('save_model', True))
    model_state_path = os.path.join(results_dir, f"anfis_model_state_{timestamp}.pt")

    if save_model:
        print(f"\n💾 Сохранение модели: {model_state_path}")
        torch.save(results['model'].network.state_dict(), model_state_path)
    else:
        print("\n💾 Сохранение модели отключено конфигом")
        model_state_path = None

    saved_files = {}

    # Сохранение предсказаний и эталонов
    predictions_array = np.asarray(results['predictions'], dtype=float)
    targets_array = np.asarray(y_test_array, dtype=float)

    predictions_array = np.nan_to_num(predictions_array, nan=0.0, posinf=0.0, neginf=0.0)
    targets_array = np.nan_to_num(targets_array, nan=0.0, posinf=0.0, neginf=0.0)

    prediction_stats = {
        'mean': float(np.nanmean(predictions_array)),
        'std': float(np.nanstd(predictions_array)),
        'min': float(np.nanmin(predictions_array)),
        'max': float(np.nanmax(predictions_array)),
        'zero_fraction': float(np.mean(np.isclose(predictions_array, 0.0))),
        'negative_fraction': float(np.mean(predictions_array < 0.0)),
        'negative_count': int(np.sum(predictions_array < 0.0)),
    }

    target_stats = {
        'mean': float(np.nanmean(targets_array)),
        'std': float(np.nanstd(targets_array)),
        'min': float(np.nanmin(targets_array)),
        'max': float(np.nanmax(targets_array))
    }

    if output_config.get('save_predictions', False):
        predictions_path = os.path.join(results_dir, f"predictions_{timestamp}.npy")
        np.save(predictions_path, predictions_array)
        saved_files['predictions'] = os.path.basename(predictions_path)

        targets_test_path = os.path.join(results_dir, f"targets_test_{timestamp}.npy")
        np.save(targets_test_path, targets_array)
        saved_files['targets_test'] = os.path.basename(targets_test_path)

        if 'predictions_denorm' in results:
            predictions_denorm_path = os.path.join(results_dir, f"predictions_denorm_{timestamp}.npy")
            np.save(predictions_denorm_path, np.asarray(results['predictions_denorm'], dtype=float))
            saved_files['predictions_denorm'] = os.path.basename(predictions_denorm_path)
        if normalize_sum and SUM_test is not None:
            targets_denorm_path = os.path.join(results_dir, f"targets_test_denorm_{timestamp}.npy")
            np.save(targets_denorm_path, np.asarray(y_test_denorm, dtype=float))
            saved_files['targets_denorm'] = os.path.basename(targets_denorm_path)

    # Сохранение подвыборки для графиков
    if output_config.get('save_samples', False):
        print("\n💾 Сохранение образцов для графиков...")
        sample_size = int(output_config.get('sample_size', 5))
        print(f"   • Запрошено образцов: {sample_size}")
        sample_size = max(sample_size, 0)
        if sample_size > 0:
            print(f"   • Доступно для выбора: {X_test_array.shape[0]}")
            sample_size = min(sample_size, X_test_array.shape[0])
            rng = np.random.default_rng(dataset_config.get('random_state', 42))
            sample_indices = np.sort(rng.choice(X_test_array.shape[0], size=sample_size, replace=False))
            print(f"   • Выбраны индексы: {sample_indices}")

            sample_prefix = os.path.join(results_dir, f"samples_{timestamp}")
            np.save(f"{sample_prefix}_X.npy", np.asarray(X_test_array[sample_indices], dtype=float))
            np.save(f"{sample_prefix}_y.npy", np.asarray(y_test_array[sample_indices], dtype=float))
            np.save(f"{sample_prefix}_pred.npy", np.asarray(results['predictions'][sample_indices], dtype=float))

            sample_record = {
                'indices': sample_indices.tolist(),
                'X': os.path.basename(f"{sample_prefix}_X.npy"),
                'y': os.path.basename(f"{sample_prefix}_y.npy"),
                'pred': os.path.basename(f"{sample_prefix}_pred.npy")
            }
            if SUM_test is not None:
                sum_array = np.asarray(SUM_test)
                np.save(f"{sample_prefix}_sum.npy", np.asarray(sum_array[sample_indices], dtype=float))
                sample_record['sum'] = os.path.basename(f"{sample_prefix}_sum.npy")

            saved_files['samples'] = sample_record

    # Сохраняем метрики в CSV
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_csv_path = os.path.join(results_dir, f"metrics_{timestamp}.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    saved_files['metrics_csv'] = os.path.basename(metrics_csv_path)

    # Сохраняем важность признаков
    if hasattr(X_train, 'columns'):
        feature_names = list(X_train.columns)
    else:
        feature_names = [f'X{i+1}' for i in range(X_train.shape[1])]
    
    # Базовая важность признаков (из vanilla модели)
    if 'feature_importance' in results:
        fi = _prepare_feature_importance(results['feature_importance'], feature_names, normalize=False)
        fi_path = os.path.join(results_dir, f"feature_importance_{timestamp}.csv")
        fi.to_csv(fi_path, header=['importance'])
        saved_files['feature_importance'] = os.path.basename(fi_path)
    
    # SHAP важность признаков (основная)
    shap_files = {}
    shap_fi = _prepare_feature_importance(results['feature_importance_shap'], feature_names, normalize=True)
    results['feature_importance_shap'] = shap_fi.to_numpy(dtype=float)
    shap_fi_path = os.path.join(results_dir, f"feature_importance_shap_{timestamp}.csv")
    shap_fi.to_csv(shap_fi_path, header=['importance'])
    shap_files['feature_importance_shap'] = os.path.basename(shap_fi_path)

    shap_history_path = os.path.join(results_dir, f"shap_history_{timestamp}.json")
    with open(shap_history_path, 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(results['shap_history']), f, ensure_ascii=False, indent=2)
    shap_files['history'] = os.path.basename(shap_history_path)

    saved_files['shap'] = shap_files

    coeff_stats = {}
    coeff_tensor = results['model'].network.state_dict().get('coeffs')
    if coeff_tensor is not None:
        coeff_np = coeff_tensor.detach().cpu().numpy().astype(float)
        finite_mask = np.isfinite(coeff_np)
        coeff_clean = np.nan_to_num(coeff_np, nan=0.0, posinf=0.0, neginf=0.0)
        coeff_abs = np.abs(coeff_clean)
        coeff_stats = {
            'mean': float(np.nanmean(coeff_clean)),
            'std': float(np.nanstd(coeff_clean)),
            'min': float(np.nanmin(coeff_clean)),
            'max': float(np.nanmax(coeff_clean)),
            'abs_mean': float(np.nanmean(coeff_abs)),
            'finite_fraction': float(np.mean(finite_mask)),
            'nonzero': int(np.count_nonzero(coeff_clean)),
            'total': int(coeff_np.size)
        }

    summary = {
        'timestamp': timestamp,
        'tag': args.tag,
        'config_path': os.path.abspath(config_path),
        'config_sha256': config_sha256,
        'effective_config_sha256': effective_config_sha256,
        'run_meta': _to_serializable(run_meta),
        'model_state': os.path.basename(model_state_path) if model_state_path else None,
        'model_state_path': model_state_path,
        'train_size': int(X_train.shape[0]),  # Синтетические данные для базовой модели
        'shap_train_size': int(X_real_shap_array.shape[0]),  # Реальные данные для SHAP
        'test_size': int(X_real_test_array.shape[0]),  # Реальные данные для теста
        'vanilla_train_count': int(X_train.shape[0]),
        'shap_train_count': int(X_real_shap_array.shape[0]),
        'real_test_count': int(X_real_test_array.shape[0]),
        'normalize_sum': normalize_sum,
        'metrics': results['metrics'],
        'vanilla_metrics': results.get('vanilla_metrics'),
        'shap_metrics': results.get('shap_metrics_raw'),
        'band_metrics': band_metrics_norm,
        'metrics_source': results.get('metrics_source', 'shap'),
        'shap_config_enabled': True,
        'shap_applied': results.get('metrics_source', 'shap') == 'shap',
        'shap_rejected': results.get('metrics_source') == 'vanilla_fallback',
        'effective_ea_config': {
            'autonomous_error_shap': bool(shap_config.get('autonomous_error_shap', False)),
            'error_importance_mode': shap_config.get('error_importance_mode', 'baseline'),
            'error_importance_ema_beta': shap_config.get('error_importance_ema_beta'),
            'grad_importance_ema_beta': shap_config.get('grad_importance_ema_beta'),
            'error_target_rho': shap_config.get('error_target_rho'),
            'ea_alignment_loss': shap_config.get('ea_alignment_loss'),
            'ea_alignment_alpha': shap_config.get('ea_alignment_alpha'),
            'ea_warmup_fraction': shap_config.get('ea_warmup_fraction'),
            'ea_bypass_legacy_normalization': bool(shap_config.get('ea_bypass_legacy_normalization', False)),
            'ea_use_grad_balance': bool(shap_config.get('ea_use_grad_balance', False)),
            'ea_target_grad_ratio': shap_config.get('ea_target_grad_ratio'),
            'ea_scale_min': shap_config.get('ea_scale_min'),
            'ea_scale_max': shap_config.get('ea_scale_max'),
            'debug_grad_norms': bool(shap_config.get('debug_grad_norms', False)),
            'grad_norm_interval': shap_config.get('grad_norm_interval'),
            'fast_mode': bool(run_meta.get('fast_mode', False)),
            'inprocess_mode': bool(run_meta.get('inprocess_mode', False)),
        },
        'training_time_total': results.get('training_time'),
        'training_time_shap': results.get('training_time_shap'),
        'saved_files': saved_files,
        'diagnostics': {
            'prediction_stats': prediction_stats,
            'target_stats': target_stats,
            'coeff_stats': coeff_stats,
            'nonfinite_parameters': _to_serializable(results.get('nonfinite_report', {})),
            'regularization': _summarize_regularization_history(results.get('shap_history'), shap_config),
        },
        'dataset_settings': {
            'train_limit': dataset_config.get('train_limit'),
            'train_fraction': dataset_config.get('train_fraction'),
            'mix_with_real': dataset_config.get('mix_with_real', False),
            'mix_ratio': dataset_config.get('mix_ratio', 0.0),
            'synthetic_test_size': dataset_config.get('test_size', 0.25),
            'test_size': REAL_TEST_FRACTION,  # Фактическая доля реальных данных для финального теста
            'real_data_split': REAL_DATA_SPLIT,
            'random_state': dataset_config.get('random_state', 42),
            'shap_uses_real_data_only': True,
            'test_uses_real_data_only': True
        }
    }
    if metrics_denorm is not None:
        summary['metrics_denorm'] = metrics_denorm
    if band_metrics_denorm is not None:
        summary['band_metrics_denorm'] = band_metrics_denorm
    summary['shap_files'] = shap_files

    summary_path = os.path.join(results_dir, f"training_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(summary), f, ensure_ascii=False, indent=2)
    print(f"📄 Сводка обучения сохранена: {summary_path}")

    # Автоматическая генерация графиков
    if output_config.get('save_plots', False):
        print("\n📊 Генерация графиков...")
        try:
            import subprocess
            # Используем абсолютный путь к plot_results.py
            plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_results.py')
            plot_cmd = [
                sys.executable, plot_script,
                '--summary', summary_path,
                '--output-dir', results_dir
            ]
            result = subprocess.run(plot_cmd, capture_output=True, text=True, timeout=300, cwd=os.path.dirname(os.path.abspath(__file__)))
            if result.returncode == 0:
                print("✅ Графики успешно сгенерированы")
            else:
                error_msg = result.stderr if result.stderr else result.stdout
                print(f"⚠️  Ошибка при генерации графиков: {error_msg[:500]}")
                # Пробуем запустить вручную для диагностики
                print(f"   Попробуйте запустить вручную: python {plot_script} --summary {summary_path} --output-dir {results_dir}")
        except FileNotFoundError as e:
            print(f"⚠️  Файл plot_results.py не найден: {e}")
        except Exception as e:
            print(f"⚠️  Не удалось сгенерировать графики: {e}")
            import traceback
            print(f"   Детали ошибки: {traceback.format_exc()[:300]}")

    print("\n✅ Обучение завершено. Модель и метрики сохранены.")
    return model_state_path, summary_path


if __name__ == "__main__":
    train_and_save(parse_args())
