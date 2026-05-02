#!/usr/bin/env python3
"""
Обучение чистой ANFIS модели ТОЛЬКО на реальных данных
Без SHAP регуляризации - только основное обучение
"""

import argparse
import json
import os
import sys
import hashlib
import subprocess
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
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


def _get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _array_sha256(*arrays):
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr))
        h.update(str(a.shape).encode("utf-8"))
        h.update(str(a.dtype).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


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


def train_vanilla_real_only(args):
    """Обучение чистой ANFIS модели ТОЛЬКО на реальных данных"""
    t_total_start = time.perf_counter()
    
    print("=" * 80)
    print("🤖 ОБУЧЕНИЕ ЧИСТОЙ ANFIS МОДЕЛИ (ТОЛЬКО РЕАЛЬНЫЕ ДАННЫЕ, БЕЗ SHAP)")
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
    
    X_real, y_real, SUM_real = load_validation_data(
        real_data_path,
        normalize_sum=normalize_sum,
        dataset_config=dataset_config
    )
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
    split_hash = _array_sha256(X_test_array, y_test_array)

    # Обучение модели
    print("\n🛠️  Обучение ANFIS модели...")
    print("=" * 80)
    manager = ANFISManager(config)
    if hasattr(X_train, 'columns'):
        manager.set_feature_names(X_train.columns)
    
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
    output_bands = _resolve_output_bands(y_test_array.shape[1])
    test_band_metrics = _compute_band_metrics(y_test_array, test_predictions, output_bands)

    # Сохранение результатов
    print("\n💾 Сохранение результатов...")
    results_dir = Path(config.get('output', {}).get('results_dir', 'results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}

    # Сохраняем модель
    model_path = results_dir / f"anfis_model_state_{run_id}.pt"
    if hasattr(model, 'network') and model.network is not None:
        torch.save(model.network.state_dict(), model_path)
        print(f"   ✅ Модель сохранена: {model_path}")

    # Сохраняем предсказания
    predictions_path = results_dir / f"predictions_{run_id}.npy"
    np.save(predictions_path, test_predictions)
    saved_files['predictions'] = f"predictions_{run_id}.npy"
    
    targets_path = results_dir / f"targets_test_{run_id}.npy"
    np.save(targets_path, y_test_array)
    saved_files['targets_test'] = f"targets_test_{run_id}.npy"

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
        
        saved_files['predictions_denorm'] = f"predictions_denorm_{run_id}.npy"
        saved_files['targets_denorm'] = f"targets_test_denorm_{run_id}.npy"
        
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

    # Сохранение подвыборки для графиков (как в train.py)
    output_config = config.get('output', {})
    if output_config.get('save_samples', False):
        sample_size = int(output_config.get('sample_size', 5))
        sample_size = max(sample_size, 0)
        if sample_size > 0:
            sample_size = min(sample_size, X_test_array.shape[0])
            rng = np.random.default_rng(dataset_config.get('random_state', 42))
            sample_indices = np.sort(rng.choice(X_test_array.shape[0], size=sample_size, replace=False))

            sample_prefix = results_dir / f"samples_{run_id}"
            np.save(f"{sample_prefix}_X.npy", np.asarray(X_test_array[sample_indices], dtype=float))
            np.save(f"{sample_prefix}_y.npy", np.asarray(y_test_array[sample_indices], dtype=float))
            np.save(f"{sample_prefix}_pred.npy", np.asarray(test_predictions[sample_indices], dtype=float))

            sample_record = {
                'indices': sample_indices.tolist(),
                'X': f"samples_{run_id}_X.npy",
                'y': f"samples_{run_id}_y.npy",
                'pred': f"samples_{run_id}_pred.npy"
            }
            if SUM_test is not None:
                sum_array = np.asarray(SUM_test)
                np.save(f"{sample_prefix}_sum.npy", np.asarray(sum_array[sample_indices], dtype=float))
                sample_record['sum'] = f"samples_{run_id}_sum.npy"

            saved_files['samples'] = sample_record

    # Сохраняем метрики в CSV
    metrics_df = pd.DataFrame([test_metrics])
    metrics_csv_path = results_dir / f"metrics_{run_id}.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    saved_files['metrics_csv'] = f"metrics_{run_id}.csv"

    # Сохраняем важность признаков
    feature_importance = np.asarray(results.get('feature_importance', []), dtype=float).reshape(-1)
    if feature_importance.size > 0:
        feature_names = list(X_train.columns) if hasattr(X_train, 'columns') else [f"X{i+1}" for i in range(feature_importance.size)]
        if len(feature_names) != feature_importance.size:
            feature_names = [f"X{i+1}" for i in range(feature_importance.size)]
        fi_df = pd.DataFrame({"importance": feature_importance}, index=feature_names)
        fi_path = results_dir / f"feature_importance_{run_id}.csv"
        fi_df.to_csv(fi_path)
        saved_files['feature_importance'] = f"feature_importance_{run_id}.csv"

    # Сохраняем сводку
    t_total_end = time.perf_counter()
    summary = {
        'timestamp': run_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'tag': tag,
        'model_mode': 'vanilla',
        'fallback_used': False,
        'config_path': str(config_path),
        'config_sha256': config_sha256,
        'effective_config_sha256': effective_config_sha256,
        'git_commit': _get_git_commit(),
        'seed': int(dataset_config.get('random_state', 42)),
        'dataset': Path(dataset_config.get('train_data', '')).stem,
        'split_hash': split_hash,
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
        'timing': {
            'data_loading_sec': None,
            'pso_sec': float(training_time),
            'vanilla_train_sec': float(training_time),
            'ea_train_sec': 0.0,
            'deletion_eval_sec': None,
            'total_sec': float(t_total_end - t_total_start),
        },
        'model_config': {
            'num_rules': model_config['num_rules'],
            'reg_lambda': model_config['reg_lambda'],
            'pso_epochs': model_config['optim_params']['epoch'],
            'pso_pop_size': model_config['optim_params']['pop_size'],
        },
        'saved_files': saved_files
    }

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
