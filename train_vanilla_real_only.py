#!/usr/bin/env python3
"""
Обучение чистой ANFIS модели ТОЛЬКО на реальных данных
Без SHAP регуляризации - только основное обучение
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config
from src.utils.data_loader import (
    load_validation_data,
    prepare_features_targets,
    denormalize_predictions
)


ENERGY_BANDS = [
    ("band_0_19", slice(0, 20)),
    ("band_20_39", slice(20, 40)),
    ("band_40_59", slice(40, 60)),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение чистой ANFIS модели ТОЛЬКО на реальных данных (без SHAP)")
    parser.add_argument("--config", default="configs/config_vanilla_real_only.yaml", help="Путь к YAML конфигурации")
    parser.add_argument("--tag", default="vanilla_real_only", help="Дополнительный суффикс к timestamp")
    return parser.parse_args()


def _to_serializable(obj):
    """Преобразование объектов в JSON-сериализуемый формат"""
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
    """Вычисление метрик по энергетическим полосам"""
    if y_true is None or y_pred is None:
        return {}
    if y_true.shape != y_pred.shape:
        return {}

    y_true = np.nan_to_num(np.asarray(y_true, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    metrics = {}
    for name, band_slice in bands:
        y_true_band = y_true[:, band_slice]
        y_pred_band = y_pred[:, band_slice]
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


def train_vanilla_real_only(args):
    """Обучение чистой ANFIS модели ТОЛЬКО на реальных данных"""
    
    print("=" * 80)
    print("🤖 ОБУЧЕНИЕ ЧИСТОЙ ANFIS МОДЕЛИ (ТОЛЬКО РЕАЛЬНЫЕ ДАННЫЕ, БЕЗ SHAP)")
    print("=" * 80)

    config_path = args.config
    print(f"\n⚙️  Конфигурация: {config_path}")
    config = load_config(config_path)
    dataset_config = config['dataset']
    model_config = config['model']
    normalize_sum = dataset_config.get('normalize_sum', False)

    # Создаем timestamp для результатов
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag if args.tag else "vanilla_real_only"
    run_id = f"{timestamp}_{tag}"

    # Загружаем ТОЛЬКО реальные данные
    print(f"\n📂 Загрузка РЕАЛЬНЫХ данных...")
    real_data_path = dataset_config.get('train_data') or dataset_config.get('validation_data')
    if not real_data_path or not os.path.exists(real_data_path):
        raise FileNotFoundError(f"Файл с реальными данными не найден: {real_data_path}")
    
    X_real, y_real, SUM_real = load_validation_data(real_data_path, normalize_sum=normalize_sum)
    print(f"   ✅ Загружено {len(X_real)} реальных образцов")
    print(f"   ✅ Размерность признаков: {X_real.shape[1]}")
    print(f"   ✅ Размерность целевых значений: {y_real.shape[1]}")

    # Разделяем на train/test
    print("\n🔀 Разделение данных на train/test...")
    test_size = dataset_config.get('test_size', 0.25)
    random_state = dataset_config.get('random_state', 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real,
        test_size=test_size,
        random_state=random_state
    )
    
    # Разделяем SUM тоже
    if normalize_sum and SUM_real is not None:
        if hasattr(X_train, 'index'):
            SUM_train = SUM_real.loc[X_train.index].values if hasattr(SUM_real, 'loc') else SUM_real[X_train.index]
            SUM_test = SUM_real.loc[X_test.index].values if hasattr(SUM_real, 'loc') else SUM_real[X_test.index]
        else:
            n_train = len(X_train)
            SUM_train = SUM_real[:n_train]
            SUM_test = SUM_real[n_train:]
    else:
        SUM_train = None
        SUM_test = None
    
    print(f"   ✅ Обучающая выборка: {len(X_train)} образцов ({100*(1-test_size):.0f}%)")
    print(f"   ✅ Тестовая выборка: {len(X_test)} образцов ({100*test_size:.0f}%)")

    # Преобразуем в массивы
    X_train_array = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    X_test_array = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
    y_test_array = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test

    # Очистка от NaN/Inf
    X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_array = np.nan_to_num(y_train_array, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_array = np.nan_to_num(X_test_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_test_array = np.nan_to_num(y_test_array, nan=0.0, posinf=0.0, neginf=0.0)

    # Обучение модели
    print("\n🛠️  Обучение ANFIS модели...")
    print("=" * 80)
    manager = ANFISManager(config)
    
    print(f"\n📊 Параметры модели:")
    print(f"   • num_rules: {model_config['num_rules']}")
    print(f"   • reg_lambda: {model_config['reg_lambda']}")
    print(f"   • PSO epochs: {model_config['optim_params']['epoch']}")
    print(f"   • PSO pop_size: {model_config['optim_params']['pop_size']}")
    print(f"   • SHAP регуляризация: ОТКЛЮЧЕНА")
    
    # Обучаем vanilla модель
    results = manager.train_vanilla_model(
        X_train_array, 
        y_train_array, 
        X_test_array, 
        y_test_array
    )
    
    model = results['model']
    training_time = results['training_time']
    test_metrics = results['metrics'].copy()
    
    print(f"\n✅ Обучение завершено за {training_time:.2f} сек")
    print(f"\n📊 Метрики на тестовой выборке (реальные данные):")
    print(f"   • MSE: {test_metrics['mse']:.6f}")
    print(f"   • RMSE: {test_metrics['rmse']:.6f}")
    print(f"   • MAE: {test_metrics['mae']:.6f}")
    print(f"   • R²: {test_metrics['r2']:.6f}")

    # Вычисляем метрики по бинам
    test_predictions = results['predictions']
    test_band_metrics = _compute_band_metrics(y_test_array, test_predictions, ENERGY_BANDS)

    # Сохранение результатов
    print("\n💾 Сохранение результатов...")
    results_dir = Path(config.get('output', {}).get('results_dir', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем модель
    model_path = results_dir / f"anfis_model_state_{run_id}.pt"
    if hasattr(model, 'network') and model.network is not None:
        torch.save(model.network.state_dict(), model_path)
        print(f"   ✅ Модель сохранена: {model_path}")

    # Сохраняем предсказания
    predictions_path = results_dir / f"predictions_{run_id}.npy"
    np.save(predictions_path, test_predictions)
    
    targets_path = results_dir / f"targets_test_{run_id}.npy"
    np.save(targets_path, y_test_array)

    # Денормализация предсказаний, если нужно
    predictions_denorm = None
    targets_denorm = None
    metrics_denorm = None
    if normalize_sum and SUM_test is not None:
        predictions_denorm = denormalize_predictions(test_predictions, SUM_test)
        targets_denorm = denormalize_predictions(y_test_array, SUM_test)
        
        predictions_denorm = np.nan_to_num(predictions_denorm, nan=0.0, posinf=0.0, neginf=0.0)
        targets_denorm = np.nan_to_num(targets_denorm, nan=0.0, posinf=0.0, neginf=0.0)
        
        predictions_denorm_path = results_dir / f"predictions_denorm_{run_id}.npy"
        targets_denorm_path = results_dir / f"targets_test_denorm_{run_id}.npy"
        
        np.save(predictions_denorm_path, predictions_denorm)
        np.save(targets_denorm_path, targets_denorm)
        
        metrics_denorm = {
            'mse': float(mean_squared_error(targets_denorm, predictions_denorm, multioutput='uniform_average')),
            'rmse': float(np.sqrt(mean_squared_error(targets_denorm, predictions_denorm, multioutput='uniform_average'))),
            'mae': float(mean_absolute_error(targets_denorm, predictions_denorm, multioutput='uniform_average')),
            'r2': float(r2_score(targets_denorm, predictions_denorm, multioutput='uniform_average'))
        }
        
        print(f"   ✅ Денормализованные предсказания сохранены")
        print(f"\n📊 Метрики на денормализованных данных:")
        print(f"   • MSE: {metrics_denorm['mse']:.6f}")
        print(f"   • RMSE: {metrics_denorm['rmse']:.6f}")
        print(f"   • MAE: {metrics_denorm['mae']:.6f}")
        print(f"   • R²: {metrics_denorm['r2']:.6f}")

    # Сохраняем сводку
    summary = {
        'timestamp': run_id,
        'tag': tag,
        'config_path': str(config_path),
        'model_state': f"anfis_model_state_{run_id}.pt",
        'model_state_path': str(model_path),
        'train_size': len(X_train_array),
        'test_size': len(X_test_array),
        'normalize_sum': normalize_sum,
        'metrics': test_metrics,
        'metrics_denorm': metrics_denorm,
        'band_metrics': test_band_metrics,
        'metrics_source': 'vanilla_real_only',
        'training_time': training_time,
        'model_config': {
            'num_rules': model_config['num_rules'],
            'reg_lambda': model_config['reg_lambda'],
            'pso_epochs': model_config['optim_params']['epoch'],
            'pso_pop_size': model_config['optim_params']['pop_size'],
        },
        'saved_files': {
            'predictions': f"predictions_{run_id}.npy",
            'targets_test': f"targets_test_{run_id}.npy",
        }
    }
    
    if predictions_denorm is not None:
        summary['saved_files']['predictions_denorm'] = f"predictions_denorm_{run_id}.npy"
        summary['saved_files']['targets_denorm'] = f"targets_test_denorm_{run_id}.npy"

    summary_path = results_dir / f"training_summary_{run_id}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(summary), f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Сводка сохранена: {summary_path}")

    print("\n" + "=" * 80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)
    print(f"\n📊 ИТОГОВЫЕ МЕТРИКИ:")
    print(f"   • Тестовая выборка R²: {test_metrics['r2']:.6f}")
    if metrics_denorm:
        print(f"   • Денормализованные данные R²: {metrics_denorm['r2']:.6f}")
    print(f"\n💾 Результаты сохранены в: {results_dir}")
    
    return summary


if __name__ == "__main__":
    args = parse_args()
    train_vanilla_real_only(args)

