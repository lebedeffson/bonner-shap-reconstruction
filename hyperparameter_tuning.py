#!/usr/bin/env python3
"""
Подбор гиперпараметров для ANFIS модели с SHAP регуляризацией
Использует валидационные данные для выбора лучших параметров
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np

# Принудительная запись вывода без буферизации
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
import yaml

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from src.models.anfis_manager import ANFISManager
from src.models.shap_trainer import ShapAwareANFISTrainer
from src.utils.config_loader import load_config
from src.utils.data_loader import (
    load_training_dataset,
    prepare_features_targets,
    load_validation_data,
    denormalize_predictions
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def create_hyperparameter_grid():
    """
    Создает расширенную сетку гиперпараметров для полного перебора
    """
    grid = {
        'num_rules': [5, 7, 10, 12, 15],  # Больше правил
        'reg_lambda': [0.1, 0.3, 0.5, 0.7],  # Больше вариантов регуляризации
        'pso_epoch': [10, 15, 20, 25],  # Больше эпох PSO
        'pso_pop_size': [20, 30, 40, 50],  # Больше размер популяции
        'optim': ['OriginalPSO', 'PSO', 'GA'],  # Разные оптимизаторы
        'mf_class': ['Gaussian', 'GBell', 'Triangular'],  # Все функции принадлежности
        'shap_gamma': [0.3, 0.5, 0.7, 0.9],  # Больше вариантов SHAP gamma
        'shap_epochs': [20, 25, 30, 35],  # Больше эпох SHAP
        'shap_lr': [0.001, 0.003, 0.005, 0.007]  # Больше вариантов learning rate
    }
    return grid


def train_single_config(config_dict, X_train_synth, y_train_synth, 
                       X_real_shap, y_real_shap, X_real_val, y_real_val,
                       normalize_sum, SUM_real_val):
    """
    Обучает модель с заданной конфигурацией и возвращает метрики на валидации
    """
    try:
        # Создаем временный конфиг
        temp_config = {
            'model': {
                'num_rules': config_dict['num_rules'],
                'mf_class': config_dict['mf_class'],
                'vanishing_strategy': 'blend',
                'optim': config_dict['optim'],  # Используем оптимизатор из конфига
                'reg_lambda': config_dict['reg_lambda'],
                'seed': 42,
                'n_workers': 4,
                'optim_params': {
                    'epoch': config_dict['pso_epoch'],
                    'pop_size': config_dict['pso_pop_size'],
                    'verbose': False
                }
            },
            'shap_reg': {
                'enabled': True,
                'gamma': config_dict['shap_gamma'],
                'epochs': config_dict['shap_epochs'],
                'batch_size': 32,
                'lr': config_dict['shap_lr'],
                'grad_clip': 1.0
            },
            'dataset': {
                'normalize_sum': normalize_sum
            }
        }
        
        # Обучение базовой модели
        manager = ANFISManager(temp_config)
        results = manager.train_vanilla_model(
            X_train_synth, y_train_synth, 
            X_real_val, y_real_val
        )
        
        # SHAP дообучение
        shap_trainer = ShapAwareANFISTrainer(
            results['model'],
            temp_config,
            gamma=config_dict['shap_gamma'],
            verbose=False
        )
        
        shap_trainer.fit(
            X_real_shap,
            y_real_shap,
            epochs=config_dict['shap_epochs'],
            batch_size=32,
            lr=config_dict['shap_lr']
        )
        
        # Валидация
        val_predictions = shap_trainer.predict(X_real_val)
        val_predictions = manager._sanitize_predictions(
            val_predictions,
            reference_shape=y_real_val.shape,
            context="validation"
        )
        
        val_metrics = manager._calculate_metrics(y_real_val, val_predictions)
        
        return {
            'success': True,
            'metrics': val_metrics,
            'r2': val_metrics.get('r2', -np.inf),
            'mse': val_metrics.get('mse', np.inf),
            'rmse': val_metrics.get('rmse', np.inf),
            'mae': val_metrics.get('mae', np.inf),
            'training_time': results.get('training_time', 0) + shap_trainer.training_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'r2': -np.inf,
            'mse': np.inf,
            'rmse': np.inf,
            'mae': np.inf
        }


def hyperparameter_tuning(args):
    """
    Выполняет подбор гиперпараметров
    """
    print("=" * 80, flush=True)
    print("🔍 ПОДБОР ГИПЕРПАРАМЕТРОВ", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()
    
    # Загружаем конфигурацию
    config = load_config(args.config)
    dataset_config = config['dataset']
    normalize_sum = dataset_config.get('normalize_sum', False)
    
    # Загружаем синтетические данные для базового обучения
    print("\n📂 Загрузка синтетических данных...")
    data_synth = load_training_dataset(dataset_config)
    X_train_synth, y_train_synth, _ = prepare_features_targets(
        data_synth, normalize_sum=normalize_sum
    )
    
    # Ограничиваем размер для ускорения
    if args.synth_limit:
        n_limit = min(args.synth_limit, len(X_train_synth))
        X_train_synth = X_train_synth.iloc[:n_limit] if hasattr(X_train_synth, 'iloc') else X_train_synth[:n_limit]
        y_train_synth = y_train_synth.iloc[:n_limit] if hasattr(y_train_synth, 'iloc') else y_train_synth[:n_limit]
        print(f"   ▶️ Используется {n_limit} синтетических образцов")
    
    # Загружаем реальные данные
    print("\n📂 Загрузка реальных данных...")
    real_data_path = dataset_config.get('validation_data')
    if not real_data_path or not os.path.exists(real_data_path):
        raise FileNotFoundError(f"Файл с реальными данными не найден: {real_data_path}")
    
    X_real, y_real, SUM_real = load_validation_data(real_data_path, normalize_sum=normalize_sum)
    
    # Разделяем: 80% для SHAP обучения, 20% для валидации
    random_state = dataset_config.get('random_state', 42)
    X_real_shap, X_real_val, y_real_shap, y_real_val = train_test_split(
        X_real, y_real, test_size=0.2, random_state=random_state
    )
    
    # Преобразуем в массивы
    X_real_shap = np.array(X_real_shap) if not isinstance(X_real_shap, np.ndarray) else X_real_shap
    y_real_shap = np.array(y_real_shap) if not isinstance(y_real_shap, np.ndarray) else y_real_shap
    X_real_val = np.array(X_real_val) if not isinstance(X_real_val, np.ndarray) else X_real_val
    y_real_val = np.array(y_real_val) if not isinstance(y_real_val, np.ndarray) else y_real_val
    
    X_real_shap = np.nan_to_num(X_real_shap, nan=0.0, posinf=0.0, neginf=0.0)
    y_real_shap = np.nan_to_num(y_real_shap, nan=0.0, posinf=0.0, neginf=0.0)
    X_real_val = np.nan_to_num(X_real_val, nan=0.0, posinf=0.0, neginf=0.0)
    y_real_val = np.nan_to_num(y_real_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    SUM_real_val = None
    if normalize_sum and SUM_real is not None:
        if hasattr(X_real_shap, 'index'):
            SUM_real_val = SUM_real.loc[X_real_val.index].values if hasattr(SUM_real, 'loc') else SUM_real[X_real_val.index]
        else:
            n_shap = len(X_real_shap)
            SUM_real_val = SUM_real[n_shap:]
    
    print(f"   ▶️ SHAP обучение: {len(X_real_shap)} образцов")
    print(f"   ▶️ Валидация: {len(X_real_val)} образцов")
    
    # Создаем сетку гиперпараметров
    grid = create_hyperparameter_grid()
    
    # Генерируем все комбинации
    param_names = list(grid.keys())
    param_values = list(grid.values())
    all_combinations = list(product(*param_values))
    
    total_combinations = len(all_combinations)
    if args.limit:
        all_combinations = all_combinations[:args.limit]
        total_combinations = len(all_combinations)
    
    print(f"\n🔍 Всего комбинаций для проверки: {total_combinations}", flush=True)
    sys.stdout.flush()
    
    # Результаты
    results = []
    
    # Перебираем комбинации
    for i, combination in enumerate(all_combinations, 1):
        config_dict = dict(zip(param_names, combination))
        
        print(f"\n{'='*80}", flush=True)
        print(f"Комбинация {i}/{total_combinations}", flush=True)
        print(f"Параметры: {json.dumps(config_dict, indent=2, ensure_ascii=False)}", flush=True)
        sys.stdout.flush()
        
        result = train_single_config(
            config_dict,
            X_train_synth, y_train_synth,
            X_real_shap, y_real_shap,
            X_real_val, y_real_val,
            normalize_sum, SUM_real_val
        )
        
        result['config'] = config_dict
        result['combination_number'] = i
        results.append(result)
        
        if result['success']:
            print(f"✅ R² = {result['r2']:.4f}, MSE = {result['mse']:.6f}, MAE = {result['mae']:.6f}", flush=True)
        else:
            print(f"❌ Ошибка: {result.get('error', 'Unknown')}", flush=True)
        
        sys.stdout.flush()
        
        # Сохраняем промежуточные результаты каждые 10 комбинаций
        if i % 10 == 0:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_csv_path = os.path.join(args.output_dir, f"hyperparameter_tuning_progress_{timestamp}.csv")
                temp_results_df = pd.DataFrame([
                    {
                        'optim': r['config'].get('optim', 'N/A'),
                        'num_rules': r['config']['num_rules'],
                        'reg_lambda': r['config']['reg_lambda'],
                        'pso_epoch': r['config']['pso_epoch'],
                        'pso_pop_size': r['config']['pso_pop_size'],
                        'mf_class': r['config']['mf_class'],
                        'shap_gamma': r['config']['shap_gamma'],
                        'shap_epochs': r['config']['shap_epochs'],
                        'shap_lr': r['config']['shap_lr'],
                        'r2': r['r2'],
                        'mse': r['mse'],
                        'rmse': r['rmse'],
                        'mae': r['mae'],
                        'training_time': r.get('training_time', 0),
                        'success': r['success'],
                        'error': r.get('error', '')
                    }
                    for r in results
                ])
                temp_results_df.to_csv(temp_csv_path, index=False)
                print(f"\n💾 Промежуточные результаты сохранены: {temp_csv_path} ({i}/{total_combinations})", flush=True)
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠️  Не удалось сохранить промежуточные результаты: {e}", flush=True)
                sys.stdout.flush()
    
    # Сохраняем результаты
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV с результатами
    results_df = pd.DataFrame([
        {
            'optim': r['config']['optim'],
            'num_rules': r['config']['num_rules'],
            'reg_lambda': r['config']['reg_lambda'],
            'pso_epoch': r['config']['pso_epoch'],
            'pso_pop_size': r['config']['pso_pop_size'],
            'mf_class': r['config']['mf_class'],
            'shap_gamma': r['config']['shap_gamma'],
            'shap_epochs': r['config']['shap_epochs'],
            'shap_lr': r['config']['shap_lr'],
            'r2': r['r2'],
            'mse': r['mse'],
            'rmse': r['rmse'],
            'mae': r['mae'],
            'training_time': r.get('training_time', 0),
            'success': r['success'],
            'error': r.get('error', '')
        }
        for r in results
    ])
    
    csv_path = os.path.join(args.output_dir, f"hyperparameter_tuning_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n📊 Результаты сохранены: {csv_path}")
    
    # JSON с полными результатами
    json_path = os.path.join(args.output_dir, f"hyperparameter_tuning_{timestamp}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📄 Полные результаты сохранены: {json_path}")
    
    # Находим лучшие комбинации
    successful_results = [r for r in results if r['success']]
    if successful_results:
        best_r2 = max(successful_results, key=lambda x: x['r2'])
        best_mse = min(successful_results, key=lambda x: x['mse'])
        
        print(f"\n🏆 ЛУЧШИЕ КОМБИНАЦИИ:")
        print(f"\nПо R² (R² = {best_r2['r2']:.4f}):")
        print(json.dumps(best_r2['config'], indent=2, ensure_ascii=False))
        
        print(f"\nПо MSE (MSE = {best_mse['mse']:.6f}):")
        print(json.dumps(best_mse['config'], indent=2, ensure_ascii=False))
        
        # Сохраняем лучшую конфигурацию
        best_config_path = os.path.join(args.output_dir, f"best_config_{timestamp}.yaml")
        best_config = config.copy()
        best_config['model'].update({
            'num_rules': best_r2['config']['num_rules'],
            'mf_class': best_r2['config']['mf_class'],
            'optim': best_r2['config']['optim'],
            'reg_lambda': best_r2['config']['reg_lambda'],
            'optim_params': {
                'epoch': best_r2['config']['pso_epoch'],
                'pop_size': best_r2['config']['pso_pop_size'],
                'verbose': False
            }
        })
        best_config['shap_reg'].update({
            'gamma': best_r2['config']['shap_gamma'],
            'epochs': best_r2['config']['shap_epochs'],
            'lr': best_r2['config']['shap_lr']
        })
        
        with open(best_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
        print(f"\n💾 Лучшая конфигурация сохранена: {best_config_path}")
    else:
        print("\n⚠️  Не удалось найти успешные комбинации!")
    
    print("\n✅ Подбор гиперпараметров завершен!")


def main():
    parser = argparse.ArgumentParser(
        description="Подбор гиперпараметров для ANFIS модели"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Базовый конфигурационный файл"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Директория для сохранения результатов"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Ограничить количество комбинаций для проверки"
    )
    parser.add_argument(
        "--synth-limit",
        type=int,
        default=50000,
        help="Ограничить количество синтетических данных для обучения (по умолчанию 50000)"
    )
    
    args = parser.parse_args()
    hyperparameter_tuning(args)


if __name__ == "__main__":
    main()

