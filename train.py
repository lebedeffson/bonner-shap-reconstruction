#!/usr/bin/env python3
"""Обучение ANFIS модели и сохранение результатов"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

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


ENERGY_BANDS = [
    ("band_0_19", slice(0, 20)),
    ("band_20_39", slice(20, 40)),
    ("band_40_59", slice(40, 60)),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение ANFIS модели для восстановления спектра")
    parser.add_argument("--config", default="configs/config.yaml", help="Путь к YAML конфигурации")
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


def train_and_save(args):
    """Обучает модель и сохраняет артефакты"""

    print("=" * 80)
    print("🤖 ОБУЧЕНИЕ ANFIS МОДЕЛИ (train.py)")
    print("=" * 80)

    config_path = args.config
    print(f"\n⚙️  Конфигурация: {config_path}")
    config = load_config(config_path)
    dataset_config = config['dataset']
    normalize_sum = dataset_config.get('normalize_sum', False)

    if args.train_limit is not None:
        dataset_config['train_limit'] = args.train_limit
        print(f"   ➤ Переопределён dataset.train_limit = {args.train_limit}")

    if args.train_fraction is not None:
        dataset_config['train_fraction'] = args.train_fraction
        print(f"   ➤ Переопределён dataset.train_fraction = {args.train_fraction}")

    # Проверяем режим обучения ДО загрузки данных
    shap_config = config.get('shap_reg', {})
    integrated_training = False  # интегрированный режим отключаем; оставляем двухэтапный
    # Для интегрированного режима по умолчанию используем full_fast, иначе real_only
    training_mode = shap_config.get('training_mode', 'full_fast' if integrated_training else 'real_only')
    
    # Загружаем реальные данные для SHAP обучения и тестирования
    from src.utils.data_loader import load_validation_data
    from sklearn.model_selection import train_test_split
    
    real_data_path = dataset_config.get('validation_data')
    if not real_data_path or not os.path.exists(real_data_path):
        raise FileNotFoundError(f"Файл с реальными данными не найден: {real_data_path}")
    
    print("\n📂 Загрузка реальных данных...")
    X_real, y_real, SUM_real = load_validation_data(
        real_data_path,
        normalize_sum=normalize_sum,
        dataset_config=dataset_config
    )
    
    # Если режим real_only и интегрированное обучение - используем только реальные данные
    if training_mode == 'real_only' and integrated_training:
        print("   ▶️ Режим real_only: используем только реальные данные для обучения")
        # Используем реальные данные как тренировочные
        X_train = X_real
        y_train = y_real
        SUM_train = SUM_real
        X_test = X_real  # Для совместимости, но не используется
        y_test = y_real
        SUM_test = SUM_real
    else:
        # Стандартный режим: загружаем синтетические данные
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
            # Если синтетические данные не найдены, но режим не real_only - ошибка
            if training_mode != 'real_only':
                raise FileNotFoundError(f"Синтетические данные не найдены, но режим {training_mode} требует их. Ошибка: {e}")
            # Иначе используем только реальные данные
            print(f"   ⚠️  Синтетические данные не найдены, используем только реальные данные")
            X_train = X_real
            y_train = y_real
            SUM_train = SUM_real
            X_test = X_real
            y_test = y_real
            SUM_test = SUM_real
    
    # Разделяем реальные данные: 60% обучение, 20% валидация, 20% финальный тест
    random_state = dataset_config.get('random_state', 42)
    
    # Сначала отделяем 20% для финального теста
    X_temp, X_real_test, y_temp, y_real_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=random_state
    )
    
    # Оставшиеся 80% разделяем на обучение (60% от всего = 75% от остатка) и валидацию (20% от всего = 25% от остатка)
    X_real_shap, X_real_val, y_real_shap, y_real_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state
    )
    
    # Сохраняем SUM для теста
    if normalize_sum and SUM_real is not None:
        # Если SUM_real это numpy массив или pandas series
        if hasattr(SUM_real, 'iloc'):
             # Получаем индексы для разделения SUM
             _, test_indices = train_test_split(np.arange(len(X_real)), test_size=0.2, random_state=random_state)
             SUM_real_test = SUM_real.iloc[test_indices]
        else:
             _, test_indices = train_test_split(np.arange(len(X_real)), test_size=0.2, random_state=random_state)
             SUM_real_test = SUM_real[test_indices]
    else:
        SUM_real_test = None
    
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
    
    integrated_training = False  # форсируем двухэтапный режим
    
    if integrated_training:
        # ИНТЕГРИРОВАННОЕ ОБУЧЕНИЕ
        print("\n🔄 ИНТЕГРИРОВАННОЕ ОБУЧЕНИЕ с SHAP регуляризацией")
        print("=" * 80)
        
        manager = ANFISManager(config)
        
        # Создаем модель
        print("\n📦 Создание ANFIS модели...")
        model = manager.create_model(
            verbose=True,
            input_dim=X_train_array.shape[1],
            output_dim=y_train_array.shape[1]
        )
        
        # Определяем режим обучения и параметры
        training_mode = shap_config.get('training_mode', 'full_fast')  # По умолчанию full_fast для лучших результатов
        print(f"\n📊 Режим обучения: {training_mode}")
        
        if training_mode == 'real_only':
            # Только реальные данные, больше эпох
            train_config = shap_config.get('real_only', {})
            # Используем значения из основной секции shap_reg, если указаны, иначе из real_only
            epochs = shap_config.get('epochs', train_config.get('epochs', 100))
            batch_size = shap_config.get('batch_size', train_config.get('batch_size', 32))
            shap_n_samples = shap_config.get('shap_n_samples', train_config.get('shap_n_samples', 100))
            train_X = X_real_shap_array
            train_y = y_real_shap_array
            print(f"   ▶️ Используем {len(train_X)} реальных образцов")
        elif training_mode == 'full_fast':
            # Весь датасет, быстрее
            train_config = shap_config.get('full_fast', {})
            # Используем значения из основной секции shap_reg, если указаны, иначе из full_fast
            epochs = shap_config.get('epochs', train_config.get('epochs', 50))
            batch_size = shap_config.get('batch_size', train_config.get('batch_size', 128))
            shap_n_samples = shap_config.get('shap_n_samples', train_config.get('shap_n_samples', 100))
            train_samples = shap_config.get('train_samples', train_config.get('train_samples', None))
            
            # Используем синтетические данные для обучения
            if train_samples and train_samples < len(X_train_array):
                print(f"   ▶️ Подвыборка: {train_samples} из {len(X_train_array)} синтетических образцов")
                rng = np.random.default_rng(dataset_config.get('random_state', 42))
                indices = rng.choice(len(X_train_array), size=train_samples, replace=False)
                train_X = X_train_array[indices]
                train_y = y_train_array[indices]
            else:
                train_X = X_train_array
                train_y = y_train_array
                print(f"   ▶️ Используем все {len(train_X)} синтетических образцов")
        else:
            raise ValueError(f"Неизвестный режим обучения: {training_mode}")
        
        # Создаем интегрированный тренер
        print(f"\n🧭 Создание интегрированного SHAP тренера...")
        print(f"   ▶️ gamma = {shap_config.get('gamma', 0.3)}")
        print(f"   ▶️ epochs = {epochs}")
        print(f"   ▶️ batch_size = {batch_size}")
        print(f"   ▶️ shap_n_samples = {shap_n_samples}")
        
        shap_trainer = ShapIntegratedANFISTrainer(
            model,
            config,
            gamma=shap_config.get('gamma', 0.3),
            verbose=True
        )
        
        # Опциональная PSO инициализация
        use_pso_init = shap_config.get('use_pso_init', False)
        if use_pso_init:
            pso_epochs = shap_config.get('pso_epochs', 5)
            print(f"\n🔧 PSO инициализация ({pso_epochs} эпох)...")
            # PSO инициализация выполняется внутри fit_from_scratch если нужно
        
        # Интегрированное обучение
        print(f"\n🚀 Запуск интегрированного обучения...")
        shap_history = shap_trainer.fit_from_scratch(
            train_X,
            train_y,
            epochs=epochs,
            batch_size=batch_size,
            lr=shap_config.get('lr', 0.002),
            X_val=X_real_val_array,
            y_val=y_real_val_array
        )
        
        results = {
            'model': model,
            'training_time': shap_trainer.training_time,
            'training_time_shap': shap_trainer.training_time,
            'shap_history': shap_history
        }
        
    else:
        # ДВУХЭТАПНОЕ ОБУЧЕНИЕ (старый режим) - как в рабочей версии
        print("\n🛠️  ДВУХЭТАПНОЕ ОБУЧЕНИЕ")
        print("=" * 80)
        
        print("\n🛠️  Этап 1: Обучение базовой ANFIS модели на синтетических данных...")
        manager = ANFISManager(config)
        if hasattr(X_train, 'columns'):
            manager.set_feature_names(X_train.columns)
        results = manager.train_vanilla_model(X_train_array, y_train_array, X_real_val_array, y_real_val_array)
        
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
        
        shap_history = shap_trainer.fit(
            shap_X_train,
            shap_y_train,
            epochs=shap_config.get('epochs', 25),
            batch_size=shap_config.get('batch_size', 32),
            lr=shap_config.get('lr', 0.003)
        )
        
        results['training_time_shap'] = shap_trainer.training_time
        results['training_time'] += shap_trainer.training_time
        results['shap_history'] = shap_history
    
    # Тестирование на ВСЕХ реальных данных
    print("\n🧪 Финальное тестирование на ВСЕХ реальных данных...")
    # Преобразуем все реальные данные для финального теста
    X_real_test_array = np.array(X_real_test) if not isinstance(X_real_test, np.ndarray) else X_real_test
    y_real_test_array = np.array(y_real_test) if not isinstance(y_real_test, np.ndarray) else y_real_test
    X_real_test_array = np.nan_to_num(X_real_test_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_real_test_array = np.nan_to_num(y_real_test_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Предсказания и метрики
    shap_predictions = shap_trainer.predict(X_real_test_array)
    shap_predictions = manager._sanitize_predictions(
        shap_predictions,
        reference_shape=y_real_test_array.shape,
        context="shap"
    )
    
    shap_metrics = manager._calculate_metrics(y_real_test_array, shap_predictions)
    
    # Вычисляем важность признаков на данных для SHAP
    if integrated_training:
        # Для интегрированного обучения используем тренировочные данные
        shap_importance_data = train_X
    else:
        shap_importance_data = X_real_shap_array
    
    shap_importance = shap_trainer.get_global_shap_importance(shap_importance_data)
    
    # Проверка на NaN/Inf
    metrics_array = np.array(list(shap_metrics.values()), dtype=float)
    importance_array = np.array(shap_importance, dtype=float)
    if not np.isfinite(metrics_array).all() or not np.isfinite(importance_array).all():
        print("⚠️  SHAP-регуляризация дала некорректные значения (NaN/Inf).")
        raise ValueError("SHAP обучение завершилось с ошибкой: NaN/Inf в метриках")
    
    # Сохраняем результаты SHAP
    results['predictions'] = shap_predictions
    results['metrics'] = shap_metrics
    results['feature_importance_shap'] = shap_importance
    results['shap_history'] = shap_history
    results['training_time_shap'] = shap_trainer.training_time
    results['training_time'] += shap_trainer.training_time
    results['metrics_source'] = 'shap'
    
    # Обновляем тестовые данные на реальные
    y_test_array = y_real_test_array
    X_test_array = X_real_test_array
    SUM_test = SUM_real_test

    band_metrics_norm = _compute_band_metrics(y_test_array, np.asarray(results['predictions']), ENERGY_BANDS)

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
        metrics_denorm = {
            'mse': float(mean_squared_error(y_test_denorm, y_pred_denorm, multioutput='uniform_average')),
            'rmse': float(np.sqrt(mean_squared_error(y_test_denorm, y_pred_denorm, multioutput='uniform_average'))),
            'mae': float(mean_absolute_error(y_test_denorm, y_pred_denorm, multioutput='uniform_average')),
            'r2': float(r2_score(y_test_denorm, y_pred_denorm, multioutput='uniform_average'))
        }
        results['predictions_denorm'] = y_pred_denorm
        results['metrics_denorm'] = metrics_denorm

    band_metrics_denorm = _compute_band_metrics(
        y_test_denorm if y_test_denorm is not None else None,
        y_pred_denorm if y_pred_denorm is not None else None,
        ENERGY_BANDS
    )

    # Папка результатов и артефакты
    output_config = config.get('output', {})
    results_dir = output_config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)

    timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = f"{timestamp_base}_{args.tag}" if args.tag else timestamp_base

    model_state_path = os.path.join(results_dir, f"anfis_model_state_{timestamp}.pt")

    print(f"\n💾 Сохранение модели: {model_state_path}")
    torch.save(results['model'].network.state_dict(), model_state_path)

    saved_files = {}

    # Сохранение предсказаний и эталонов
    prediction_stats = {}
    target_stats = {}

    if output_config.get('save_predictions', False):
        predictions_array = np.asarray(results['predictions'], dtype=float)
        targets_array = np.asarray(y_test_array, dtype=float)

        predictions_array = np.nan_to_num(predictions_array, nan=0.0, posinf=0.0, neginf=0.0)
        targets_array = np.nan_to_num(targets_array, nan=0.0, posinf=0.0, neginf=0.0)

        predictions_path = os.path.join(results_dir, f"predictions_{timestamp}.npy")
        np.save(predictions_path, predictions_array)
        saved_files['predictions'] = os.path.basename(predictions_path)

        targets_test_path = os.path.join(results_dir, f"targets_test_{timestamp}.npy")
        np.save(targets_test_path, targets_array)
        saved_files['targets_test'] = os.path.basename(targets_test_path)

        prediction_stats = {
            'mean': float(np.nanmean(predictions_array)),
            'std': float(np.nanstd(predictions_array)),
            'min': float(np.nanmin(predictions_array)),
            'max': float(np.nanmax(predictions_array)),
            'zero_fraction': float(np.mean(np.isclose(predictions_array, 0.0)))
        }

        target_stats = {
            'mean': float(np.nanmean(targets_array)),
            'std': float(np.nanstd(targets_array)),
            'min': float(np.nanmin(targets_array)),
            'max': float(np.nanmax(targets_array))
        }

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
        fi = pd.Series(results['feature_importance'], index=feature_names)
        fi_path = os.path.join(results_dir, f"feature_importance_{timestamp}.csv")
        fi.to_csv(fi_path, header=['importance'])
        saved_files['feature_importance'] = os.path.basename(fi_path)
    
    # SHAP важность признаков (основная)
    shap_files = {}
    shap_fi = pd.Series(results['feature_importance_shap'], index=feature_names)
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
        'model_state': os.path.basename(model_state_path),
        'model_state_path': model_state_path,
        'train_size': int(X_train.shape[0]),  # Синтетические данные для базовой модели
        'shap_train_size': int(X_real_shap_array.shape[0]),  # Реальные данные для SHAP
        'test_size': int(X_real_test_array.shape[0]),  # Реальные данные для теста
        'normalize_sum': normalize_sum,
        'metrics': results['metrics'],
        'band_metrics': band_metrics_norm,
        'metrics_source': 'shap',
        'shap_config_enabled': True,
        'shap_applied': True,
        'training_time_total': results.get('training_time'),
        'training_time_shap': results.get('training_time_shap'),
        'saved_files': saved_files,
        'diagnostics': {
            'prediction_stats': prediction_stats,
            'target_stats': target_stats,
            'coeff_stats': coeff_stats,
            'nonfinite_parameters': _to_serializable(results.get('nonfinite_report', {}))
        },
        'dataset_settings': {
            'train_limit': dataset_config.get('train_limit'),
            'train_fraction': dataset_config.get('train_fraction'),
            'mix_with_real': dataset_config.get('mix_with_real', False),
            'mix_ratio': dataset_config.get('mix_ratio', 0.0),
            'test_size': 1.0,  # 100% реальных данных для финального теста
            'random_state': dataset_config.get('random_state', 42),
            'shap_uses_real_data_only': True,
            'test_uses_real_data_only': True
        }
    }
    if metrics_denorm:
        summary['metrics_denorm'] = metrics_denorm
    if band_metrics_denorm:
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
