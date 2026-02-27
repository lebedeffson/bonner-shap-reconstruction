#!/usr/bin/env python3
"""
Простой скрипт для генерации графиков по уже обученной модели
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config
from src.utils.data_loader import load_validation_data, resolve_feature_count, resolve_target_count
from constants import Ebins_float_IAEA_Comp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_model(model_path, config_path):
    """Загружает обученную модель"""
    config = load_config(config_path)
    manager = ANFISManager(config)
    dataset_config = config.get('dataset', {})
    input_dim = resolve_feature_count(dataset_config) or 10
    output_dim = resolve_target_count(dataset_config) or 60
    
    # Создаем модель
    model = manager.create_model(
        input_dim=input_dim,
        output_dim=output_dim,
        verbose=False
    )
    
    # Загружаем веса
    state_dict = torch.load(model_path, map_location='cpu')
    model.network.load_state_dict(state_dict)
    model.network.eval()
    
    return model, config


def generate_predictions(model, X_test, device='cpu'):
    """Генерирует предсказания модели"""
    model.network.eval()
    with torch.no_grad():
        # Преобразуем в numpy array если нужно
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        X_test = np.asarray(X_test, dtype=np.float32)
        X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        predictions = model.network(X_tensor).cpu().numpy()
    return predictions


def plot_spectra_comparison(y_true, y_pred, output_path, n_samples=5, seed=42):
    """Сравнение истинных и предсказанных спектров"""
    np.random.seed(seed)
    n = len(y_true)
    indices = np.random.choice(n, min(n_samples, n), replace=False)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    # Используем правильные размерности для бинов
    n_bins = y_true.shape[1] if y_true.ndim > 1 else len(y_true[0])
    if len(Ebins_float_IAEA_Comp) == n_bins + 1:
        # Ebins содержит границы бинов (n_bins+1 элементов)
        x_bins = Ebins_float_IAEA_Comp[:-1]
    elif len(Ebins_float_IAEA_Comp) == n_bins:
        # Ebins содержит центры бинов (n_bins элементов)
        x_bins = Ebins_float_IAEA_Comp
    else:
        # Создаем логарифмические бины если размерности не совпадают
        x_bins = np.logspace(-3, 8, n_bins)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        ax.step(x_bins, y_true[idx], where='post', 
                label='Истинный', linewidth=2, alpha=0.7)
        ax.step(x_bins, y_pred[idx], where='post', 
                label='Предсказанный', linewidth=2, alpha=0.7, linestyle='--')
        ax.set_xscale('log')
        ax.set_xlabel('Энергия (эВ)')
        ax.set_ylabel('Плотность потока')
        ax.set_title(f'Спектр #{idx+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Сохранено: {output_path}")


def plot_scatter(y_true, y_pred, output_path):
    """Scatter plot предсказаний vs истинных значений"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Flatten для scatter plot
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Убираем нули для лучшей визуализации
    mask = (y_true_flat > 0) & (y_pred_flat > 0)
    y_true_flat = y_true_flat[mask]
    y_pred_flat = y_pred_flat[mask]
    
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=1)
    
    # Линия идеального предсказания
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальное предсказание')
    
    # Метрики
    r2 = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    ax.set_xlabel('Истинные значения')
    ax.set_ylabel('Предсказанные значения')
    ax.set_title(f'Scatter Plot (R²={r2:.3f}, RMSE={rmse:.4f}, MAE={mae:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Сохранено: {output_path}")


def plot_error_distribution(y_true, y_pred, output_path):
    """Распределение ошибок"""
    errors = (y_pred - y_true).flatten()
    errors = errors[np.isfinite(errors)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Гистограмма ошибок
    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='r', linestyle='--', linewidth=2, label='Нулевая ошибка')
    axes[0].set_xlabel('Ошибка (предсказание - истина)')
    axes[0].set_ylabel('Частота')
    axes[0].set_title('Распределение ошибок')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Ошибка по энергетическим бинам
    errors_by_bin = np.mean(np.abs(y_pred - y_true), axis=0)
    n_bins = len(errors_by_bin)
    if len(Ebins_float_IAEA_Comp) == n_bins + 1:
        x_bins = Ebins_float_IAEA_Comp[:-1]
    elif len(Ebins_float_IAEA_Comp) == n_bins:
        x_bins = Ebins_float_IAEA_Comp
    else:
        x_bins = np.logspace(-3, 8, n_bins)
    axes[1].step(x_bins, errors_by_bin, where='post', linewidth=2)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Энергия (эВ)')
    axes[1].set_ylabel('Средняя абсолютная ошибка')
    axes[1].set_title('Ошибка по энергетическим бинам')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Сохранено: {output_path}")


def plot_metrics_table(y_true, y_pred, output_path):
    """Таблица метрик"""
    r2 = r2_score(y_true.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ['R²', f'{r2:.4f}'],
        ['MSE', f'{mse:.6f}'],
        ['RMSE', f'{rmse:.6f}'],
        ['MAE', f'{mae:.6f}']
    ]
    
    table = ax.table(cellText=data, colLabels=['Метрика', 'Значение'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 2)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Сохранено: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Генерация графиков по обученной модели")
    parser.add_argument("--model", required=True, help="Путь к файлу модели (.pt)")
    parser.add_argument("--config", default="configs/config.yaml", help="Путь к конфигурации")
    parser.add_argument("--output-dir", default="results", help="Директория для сохранения графиков")
    parser.add_argument("--tag", help="Тег для имен файлов (если не указан, берется из имени модели)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Устройство для вычислений")
    
    args = parser.parse_args()
    
    # Определяем тег
    if args.tag:
        tag = args.tag
    else:
        # Извлекаем тег из имени файла
        model_name = Path(args.model).stem
        parts = model_name.split('_')
        if len(parts) >= 4:
            tag = '_'.join(parts[3:])  # Берем все после timestamp
        else:
            tag = "trained_model"
    
    print("=" * 80)
    print(f"📊 ГЕНЕРАЦИЯ ГРАФИКОВ ДЛЯ МОДЕЛИ: {tag}")
    print("=" * 80)
    
    # Загружаем модель
    print("\n📦 Загрузка модели...")
    model, config = load_model(args.model, args.config)
    print(f"   ✅ Модель загружена")
    
    # Перемещаем на GPU если нужно
    if args.device == 'cuda' and torch.cuda.is_available():
        model.network = model.network.cuda()
        print(f"   ✅ Модель перемещена на GPU")
    
    # Загружаем данные
    print("\n📂 Загрузка данных...")
    dataset_config = config['dataset']
    X_test, y_test, SUM_test = load_validation_data(
        dataset_config.get('validation_data'),
        normalize_sum=dataset_config.get('normalize_sum', True),
        dataset_config=dataset_config
    )
    # Преобразуем в numpy arrays если нужно
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    print(f"   ✅ Загружено {len(X_test)} образцов")
    
    # Генерируем предсказания
    print("\n🔮 Генерация предсказаний...")
    y_pred = generate_predictions(model, X_test, device=args.device)
    print(f"   ✅ Предсказания сгенерированы")
    
    # Создаем директорию для результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Генерируем графики
    print("\n📈 Генерация графиков...")
    
    # 1. Сравнение спектров
    plot_spectra_comparison(
        y_test, y_pred, 
        output_dir / f"spectra_comparison_{tag}.png",
        n_samples=5
    )
    
    # 2. Scatter plot
    plot_scatter(
        y_test, y_pred,
        output_dir / f"scatter_{tag}.png"
    )
    
    # 3. Распределение ошибок
    plot_error_distribution(
        y_test, y_pred,
        output_dir / f"errors_{tag}.png"
    )
    
    # 4. Таблица метрик
    plot_metrics_table(
        y_test, y_pred,
        output_dir / f"metrics_{tag}.png"
    )
    
    # Сохраняем предсказания
    np.save(output_dir / f"predictions_{tag}.npy", y_pred)
    print(f"   ✅ Предсказания сохранены")
    
    print("\n" + "=" * 80)
    print("✅ ВСЕ ГРАФИКИ СОЗДАНЫ УСПЕШНО!")
    print("=" * 80)


if __name__ == "__main__":
    main()
