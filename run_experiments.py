#!/usr/bin/env python3
"""
Большой цикл экспериментов с разными оптимизаторами и параметрами
SHAP регуляризация использует только реальные данные (375 спектров)
Тестирование на реальных данных
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from datetime import datetime
from itertools import product
import pandas as pd
import numpy as np

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from train import train_and_save, _to_serializable
from src.utils.config_loader import load_config


def create_experiment_configs():
    """
    Создает список всех комбинаций параметров для экспериментов
    """
    # Оптимизаторы (из документации xanfis)
    optimizers = ['OriginalPSO', 'PSO', 'GA']
    
    # Количество правил
    num_rules_list = [5, 7, 10, 12]
    
    # Коэффициент регуляризации
    reg_lambda_list = [0.1, 0.3, 0.5]
    
    # Параметры PSO
    pso_epochs_list = [10, 15, 20]
    pso_pop_size_list = [20, 30, 40]
    
    # SHAP параметры
    shap_gamma_list = [0.3, 0.5, 0.7]
    shap_epochs_list = [20, 25, 30]
    shap_lr_list = [0.001, 0.003, 0.005]
    
    # Функции принадлежности
    mf_class_list = ['Gaussian', 'GBell', 'Triangular']
    
    experiments = []
    
    for (optim, num_rules, reg_lambda, pso_epoch, pso_pop, 
         mf_class, shap_gamma, shap_epoch, shap_lr) in product(
        optimizers,
        num_rules_list,
        reg_lambda_list,
        pso_epochs_list,
        pso_pop_size_list,
        mf_class_list,
        shap_gamma_list,
        shap_epochs_list,
        shap_lr_list
    ):
        experiments.append({
            'optim': optim,
            'num_rules': num_rules,
            'reg_lambda': reg_lambda,
            'pso_epoch': pso_epoch,
            'pso_pop_size': pso_pop,
            'mf_class': mf_class,
            'shap_gamma': shap_gamma,
            'shap_epochs': shap_epoch,
            'shap_lr': shap_lr
        })
    
    return experiments


def create_config_from_experiment(base_config_path, experiment, output_dir):
    """
    Создает конфигурационный файл для конкретного эксперимента
    """
    config = load_config(base_config_path)
    
    # Обновляем параметры модели
    config['model']['optim'] = experiment['optim']
    config['model']['num_rules'] = experiment['num_rules']
    config['model']['reg_lambda'] = experiment['reg_lambda']
    config['model']['mf_class'] = experiment['mf_class']
    config['model']['optim_params']['epoch'] = experiment['pso_epoch']
    config['model']['optim_params']['pop_size'] = experiment['pso_pop_size']
    
    # Обновляем SHAP параметры
    config['shap_reg']['gamma'] = experiment['shap_gamma']
    config['shap_reg']['epochs'] = experiment['shap_epochs']
    config['shap_reg']['lr'] = experiment['shap_lr']
    config['shap_reg']['enabled'] = True
    # Важно: train_samples будет переопределен в train.py для использования только реальных данных
    
    # Сохраняем конфиг
    exp_id = (
        f"{experiment['optim']}_r{experiment['num_rules']}_"
        f"reg{experiment['reg_lambda']}_pso{experiment['pso_epoch']}x{experiment['pso_pop']}_"
        f"{experiment['mf_class']}_shap{experiment['shap_gamma']}x{experiment['shap_epochs']}x{experiment['shap_lr']}"
    )
    config_path = os.path.join(output_dir, f"config_{exp_id}.yaml")
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return config_path, exp_id


def run_experiments(args):
    """
    Запускает все эксперименты
    """
    print("=" * 80)
    print("🔬 БОЛЬШОЙ ЦИКЛ ЭКСПЕРИМЕНТОВ")
    print("=" * 80)
    
    base_config_path = args.base_config
    experiments = create_experiment_configs()
    
    # Ограничение количества экспериментов для тестирования
    if args.limit is not None and args.limit > 0:
        experiments = experiments[:args.limit]
        print(f"\n📊 Ограничено до {len(experiments)} экспериментов (для тестирования)")
    else:
        print(f"\n📊 Всего экспериментов: {len(experiments)}")
    
    # Создаем директорию для конфигов экспериментов
    exp_configs_dir = os.path.join(args.output_dir, "experiment_configs")
    os.makedirs(exp_configs_dir, exist_ok=True)
    
    # Результаты всех экспериментов
    all_results = []
    
    # Загружаем базовый конфиг для проверки путей к данным
    base_config = load_config(base_config_path)
    dataset_config = base_config['dataset']
    
    # Проверяем наличие реальных данных
    real_data_path = dataset_config.get('validation_data')
    if not real_data_path or not os.path.exists(real_data_path):
        raise FileNotFoundError(f"Файл с реальными данными не найден: {real_data_path}")
    
    print(f"\n📂 Реальные данные будут использованы для SHAP и тестирования: {real_data_path}")
    
    # Запускаем эксперименты
    for i, experiment in enumerate(experiments, 1):
        print("\n" + "=" * 80)
        print(f"ЭКСПЕРИМЕНТ {i}/{len(experiments)}")
        print("=" * 80)
        print(f"Параметры: {json.dumps(experiment, indent=2, ensure_ascii=False)}")
        
        try:
            # Создаем конфиг для эксперимента
            config_path, exp_id = create_config_from_experiment(
                base_config_path, experiment, exp_configs_dir
            )
            
            # Создаем аргументы для train.py
            class Args:
                def __init__(self):
                    self.config = config_path
                    self.tag = f"exp_{i:04d}_{exp_id}"
                    self.train_limit = None
                    self.train_fraction = None
            
            train_args = Args()
            
            # Запускаем обучение
            # Модифицируем train.py чтобы он использовал реальные данные для SHAP
            # Это будет сделано через модификацию train.py
            
            model_path, summary_path = train_and_save(train_args)
            
            # Загружаем результаты
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            # Добавляем параметры эксперимента
            summary['experiment'] = experiment
            summary['experiment_id'] = exp_id
            summary['experiment_number'] = i
            
            all_results.append(summary)
            
            print(f"✅ Эксперимент {i} завершен успешно")
            
        except Exception as e:
            print(f"❌ Ошибка в эксперименте {i}: {e}")
            import traceback
            traceback.print_exc()
            
            # Сохраняем информацию об ошибке
            error_result = {
                'experiment': experiment,
                'experiment_id': exp_id if 'exp_id' in locals() else f"exp_{i:04d}",
                'experiment_number': i,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(error_result)
    
    # Сохраняем сводку всех экспериментов
    results_df = pd.DataFrame([
        {
            'exp_num': r.get('experiment_number', 0),
            'exp_id': r.get('experiment_id', 'unknown'),
            'optim': r.get('experiment', {}).get('optim', 'unknown'),
            'num_rules': r.get('experiment', {}).get('num_rules', 0),
            'reg_lambda': r.get('experiment', {}).get('reg_lambda', 0),
            'mf_class': r.get('experiment', {}).get('mf_class', 'unknown'),
            'shap_gamma': r.get('experiment', {}).get('shap_gamma', 0),
            'r2': r.get('metrics', {}).get('r2', np.nan),
            'mse': r.get('metrics', {}).get('mse', np.nan),
            'rmse': r.get('metrics', {}).get('rmse', np.nan),
            'mae': r.get('metrics', {}).get('mae', np.nan),
            'training_time': r.get('training_time_total', np.nan),
            'error': r.get('error', None)
        }
        for r in all_results
    ])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv_path = os.path.join(args.output_dir, f"all_experiments_{timestamp}.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n📊 Сводка всех экспериментов сохранена: {results_csv_path}")
    
    # Сохраняем полные результаты в JSON
    results_json_path = os.path.join(args.output_dir, f"all_experiments_{timestamp}.json")
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(all_results), f, ensure_ascii=False, indent=2)
    print(f"📄 Полные результаты сохранены: {results_json_path}")
    
    # Топ-10 экспериментов по R²
    if 'r2' in results_df.columns:
        top_results = results_df.nlargest(10, 'r2')
        print("\n🏆 ТОП-10 экспериментов по R²:")
        print(top_results[['exp_num', 'exp_id', 'r2', 'mse', 'mae']].to_string(index=False))
    
    print("\n✅ Все эксперименты завершены!")


def main():
    parser = argparse.ArgumentParser(
        description="Запуск большого цикла экспериментов с разными параметрами"
    )
    parser.add_argument(
        "--base-config",
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
        help="Ограничить количество экспериментов (для тестирования)"
    )
    
    args = parser.parse_args()
    
    # Создаем директорию результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_experiments(args)


if __name__ == "__main__":
    main()

