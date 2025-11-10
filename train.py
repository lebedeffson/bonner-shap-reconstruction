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
from src.models.shap_trainer import ShapAwareANFISTrainer
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

    print("\n📂 Загрузка обучающих данных...")
    data = load_training_dataset(dataset_config)

    X, y, SUM_train = prepare_features_targets(data, normalize_sum=normalize_sum)

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

    print("\n🛠️  Обучение модели...")
    manager = ANFISManager(config)
    results = manager.train_vanilla_model(X_train, y_train, X_test, y_test)
    vanilla_metrics = results['metrics'].copy()
    vanilla_time = results['training_time']
    results['metrics_vanilla'] = vanilla_metrics.copy()
    results['training_time_vanilla'] = vanilla_time
    results['training_time'] = vanilla_time

    # Дополнительное обучение с SHAP-регуляризацией при необходимости
    shap_config = config.get('shap_reg', {})
    shap_results = None
    y_test_array = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
    X_test_array = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
    X_train_array = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

    y_test_array = np.nan_to_num(y_test_array, nan=0.0, posinf=0.0, neginf=0.0)
    y_train_array = np.nan_to_num(y_train_array, nan=0.0, posinf=0.0, neginf=0.0)

    if shap_config.get('enabled', False):
        print("\n🧭 Запуск SHAP-регуляризации...")
        vanilla_state = {
            name: param.detach().cpu().clone()
            for name, param in results['model'].network.state_dict().items()
        }

        shap_subset = shap_config.get('train_samples')
        if shap_subset is not None:
            shap_subset = int(shap_subset)
            if shap_subset <= 0:
                shap_subset = None

        if shap_subset is not None and shap_subset < len(X_train_array):
            rng = np.random.default_rng(dataset_config.get('random_state', 42))
            subset_idx = rng.choice(len(X_train_array), size=shap_subset, replace=False)
            shap_X_train = X_train_array[subset_idx]
            shap_y_train = y_train_array[subset_idx]
            print(f"   ▶️ SHAP будет обучаться на подвыборке {shap_subset} образцов")
        else:
            shap_X_train = X_train_array
            shap_y_train = y_train_array

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
            lr=shap_config.get('lr', 0.005)
        )

        shap_predictions = shap_trainer.predict(X_test_array)
        shap_predictions = manager._sanitize_predictions(
            shap_predictions,
            reference_shape=y_test_array.shape,
            context="shap"
        )

        shap_metrics = manager._calculate_metrics(y_test_array, shap_predictions)
        shap_importance = shap_trainer.get_global_shap_importance(X_train_array)

        metrics_array = np.array(list(shap_metrics.values()), dtype=float)
        importance_array = np.array(shap_importance, dtype=float)
        if not np.isfinite(metrics_array).all() or not np.isfinite(importance_array).all():
            print("⚠️  SHAP-регуляризация дала некорректные значения (NaN/Inf). Использую результаты Vanilla.")
            try:
                device = next(results['model'].network.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
            restored_state = {
                name: tensor.to(device)
                for name, tensor in vanilla_state.items()
            }
            results['model'].network.load_state_dict(restored_state, strict=False)
            shap_results = None
        else:
            shap_results = {
                'history': shap_history,
                'metrics': shap_metrics,
                'predictions': shap_predictions,
                'feature_importance': shap_importance,
                'training_time': shap_trainer.training_time
            }

            # Обновляем основные результаты на SHAP-модель
            results['predictions'] = shap_predictions
            results['metrics'] = shap_metrics
            results['feature_importance_shap'] = shap_importance
            results['shap_history'] = shap_history
            results['training_time_shap'] = shap_trainer.training_time
            results['training_time'] += shap_trainer.training_time
            results['metrics_source'] = 'shap'
    else:
        results['metrics_source'] = 'vanilla'

    band_metrics_norm = _compute_band_metrics(y_test_array, np.asarray(results['predictions']), ENERGY_BANDS)

    # Денормализация метрик
    metrics_denorm = None
    y_test_denorm = None
    y_pred_denorm = None
    if normalize_sum and SUM_test is not None:
        print("\n🔄 Денормализация предсказаний...")
        y_pred_denorm = denormalize_predictions(results['predictions'], SUM_test)
        y_test_denorm = denormalize_predictions(y_test.values, SUM_test)
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
        sample_size = int(output_config.get('sample_size', 5))
        sample_size = max(sample_size, 0)
        if sample_size > 0:
            sample_size = min(sample_size, X_test_array.shape[0])
            rng = np.random.default_rng(dataset_config.get('random_state', 42))
            sample_indices = np.sort(rng.choice(X_test_array.shape[0], size=sample_size, replace=False))

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
    feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'Q{i+1}' for i in range(X_train.shape[1])]
    fi = pd.Series(results['feature_importance'], index=feature_names)
    fi_path = os.path.join(results_dir, f"feature_importance_{timestamp}.csv")
    fi.to_csv(fi_path, header=['importance'])
    saved_files['feature_importance'] = os.path.basename(fi_path)

    shap_files = {}
    if shap_results is not None:
        shap_fi = pd.Series(shap_results['feature_importance'], index=feature_names)
        shap_fi_path = os.path.join(results_dir, f"feature_importance_shap_{timestamp}.csv")
        shap_fi.to_csv(shap_fi_path, header=['importance'])
        shap_files['feature_importance_shap'] = os.path.basename(shap_fi_path)

        shap_history_path = os.path.join(results_dir, f"shap_history_{timestamp}.json")
        with open(shap_history_path, 'w', encoding='utf-8') as f:
            json.dump(_to_serializable(shap_results['history']), f, ensure_ascii=False, indent=2)
        shap_files['history'] = os.path.basename(shap_history_path)

    if shap_files:
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
        'train_size': int(X_train.shape[0]),
        'test_size': int(X_test.shape[0]),
        'normalize_sum': normalize_sum,
        'metrics': results['metrics'],
        'metrics_vanilla': vanilla_metrics,
        'band_metrics': band_metrics_norm,
        'metrics_source': results.get('metrics_source', 'vanilla'),
        'shap_config_enabled': shap_config.get('enabled', False),
        'shap_applied': shap_results is not None,
        'training_time_total': results.get('training_time'),
        'training_time_vanilla': results.get('training_time_vanilla'),
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
            'test_size': dataset_config.get('test_size', 0.25),
            'random_state': dataset_config.get('random_state', 42)
        }
    }
    if metrics_denorm:
        summary['metrics_denorm'] = metrics_denorm
    if band_metrics_denorm:
        summary['band_metrics_denorm'] = band_metrics_denorm
    if shap_results is not None:
        summary['metrics_shap'] = shap_results['metrics']
        summary['shap_files'] = shap_files

    summary_path = os.path.join(results_dir, f"training_summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_to_serializable(summary), f, ensure_ascii=False, indent=2)
    print(f"📄 Сводка обучения сохранена: {summary_path}")

    print("\n✅ Обучение завершено. Модель и метрики сохранены.")
    return model_state_path, summary_path


if __name__ == "__main__":
    train_and_save(parse_args())
