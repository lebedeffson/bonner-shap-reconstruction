#!/usr/bin/env python3
"""
Визуализация интерпретируемости модели с SHAP регуляризацией
Показывает важность признаков и улучшение интерпретируемости
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config
from src.utils.data_loader import (
    load_validation_data,
    resolve_feature_count,
    resolve_target_count
)

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model_and_data(config_path, model_path):
    """Загружает модель и данные"""
    config = load_config(config_path)
    manager = ANFISManager(config)
    dataset_config = config.get('dataset', {})
    
    # Загружаем модель
    import torch
    input_dim = resolve_feature_count(dataset_config) or 10
    output_dim = resolve_target_count(dataset_config) or 60
    model = manager.create_model(
        input_dim=input_dim,
        output_dim=output_dim
    )
    model.network.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.network.eval()
    
    # Загружаем данные
    X_real, y_real, _ = load_validation_data(
        dataset_config.get('validation_data'),
        normalize_sum=dataset_config.get('normalize_sum', True),
        dataset_config=dataset_config
    )
    
    return model, X_real, y_real, config


def compute_feature_importance_shap(model, X_sample, baseline=None):
    """Вычисляет важность признаков через SHAP-подобный подход"""
    import torch
    
    model.network.eval()
    X_tensor = torch.tensor(X_sample.values if hasattr(X_sample, 'values') else X_sample, 
                           dtype=torch.float32)
    
    if baseline is None:
        baseline = torch.mean(X_tensor, dim=0)
    
    X_tensor.requires_grad_(True)
    predictions = model.network(X_tensor)
    
    # Вычисляем градиенты
    output_dim = predictions.shape[1] if predictions.ndim > 1 else 1
    grad_outputs = torch.ones_like(predictions) / output_dim
    
    grad_input = torch.autograd.grad(
        outputs=predictions,
        inputs=X_tensor,
        grad_outputs=grad_outputs,
        create_graph=False,
        retain_graph=False,
        only_inputs=True
    )[0]
    
    # Важность через градиенты
    importance = torch.abs(grad_input) * torch.abs(X_tensor)
    importance_per_feature = torch.mean(importance, dim=0).detach().numpy()
    
    # Нормализуем
    importance_normalized = importance_per_feature / (np.sum(importance_per_feature) + 1e-10)
    
    return importance_normalized


def compute_gini_coefficient(importance):
    """Вычисляет коэффициент Джини для измерения неравномерности"""
    sorted_importance = np.sort(importance)
    n = len(sorted_importance)
    indices = np.arange(1, n + 1)
    sum_weighted = np.sum(indices * sorted_importance)
    sum_total = np.sum(sorted_importance)
    gini = 1.0 - 2.0 * sum_weighted / (n * sum_total + 1e-10)
    return max(0.0, min(1.0, gini))


def compute_entropy(importance):
    """Вычисляет энтропию Шеннона"""
    importance_clean = importance[importance > 1e-10]
    if len(importance_clean) == 0:
        return 0.0
    entropy = -np.sum(importance_clean * np.log(importance_clean))
    max_entropy = np.log(len(importance))
    return entropy / (max_entropy + 1e-10)


def visualize_feature_importance(importance_dict, output_path, feature_names=None):
    """Визуализирует важность признаков"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('📊 Интерпретируемость модели: Важность признаков', fontsize=16, fontweight='bold')
    
    # 1. Bar plot важности признаков
    ax1 = axes[0, 0]
    model_names = list(importance_dict.keys())
    if not feature_names:
        feature_names = [f'X{i+1}' for i in range(len(next(iter(importance_dict.values()))))]
    
    x = np.arange(len(feature_names))
    width = 0.35
    
    for i, model_name in enumerate(model_names):
        importance = importance_dict[model_name]
        offset = (i - len(model_names)/2 + 0.5) * width / len(model_names)
        ax1.bar(x + offset, importance, width/len(model_names), label=model_name, alpha=0.8)
    
    ax1.set_xlabel('Признаки', fontsize=12)
    ax1.set_ylabel('Важность', fontsize=12)
    ax1.set_title('Важность признаков по моделям', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(feature_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gini coefficient сравнение
    ax2 = axes[0, 1]
    gini_values = [compute_gini_coefficient(importance_dict[name]) for name in model_names]
    colors = ['green' if g > 0.2 else 'orange' for g in gini_values]
    bars = ax2.bar(model_names, gini_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Коэффициент Джини', fontsize=12)
    ax2.set_title('Неравномерность распределения важности\n(выше = лучше интерпретируемость)', 
                  fontsize=14, fontweight='bold')
    ax2.axhline(y=0.3, color='red', linestyle='--', label='Целевое значение (0.3)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, val in zip(bars, gini_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Энтропия сравнение
    ax3 = axes[1, 0]
    entropy_values = [compute_entropy(importance_dict[name]) for name in model_names]
    colors = ['green' if e < 0.5 else 'orange' for e in entropy_values]
    bars = ax3.bar(model_names, entropy_values, color=colors, alpha=0.7)
    ax3.set_ylabel('Нормализованная энтропия', fontsize=12)
    ax3.set_title('Разреженность важности\n(ниже = лучше интерпретируемость)', 
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Добавляем значения на столбцы
    for bar, val in zip(bars, entropy_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Топ-3 важных признака
    ax4 = axes[1, 1]
    top_features_data = {}
    for model_name in model_names:
        importance = importance_dict[model_name]
        top_indices = np.argsort(importance)[-3:][::-1]
        top_features_data[model_name] = {
            'features': [feature_names[i] for i in top_indices],
            'importance': [importance[i] for i in top_indices]
        }
    
    # Создаем таблицу
    table_data = []
    for model_name in model_names:
        row = [model_name]
        for feat, imp in zip(top_features_data[model_name]['features'], 
                            top_features_data[model_name]['importance']):
            row.append(f"{feat}: {imp:.3f}")
        table_data.append(row)
    
    ax4.axis('tight')
    ax4.axis('off')
    table = ax4.table(cellText=table_data,
                     colLabels=['Модель', '1-й признак', '2-й признак', '3-й признак'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.3, 0.23, 0.23, 0.23])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax4.set_title('Топ-3 важных признака по моделям', fontsize=14, fontweight='bold')
    
    # Выделяем заголовок
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ График сохранен: {output_path}")
    plt.close()


def create_interpretability_report(results_dir, output_path, feature_names=None):
    """Создает отчет об интерпретируемости"""
    # Ищем файлы с важностью признаков
    importance_files = list(Path(results_dir).glob("*feature_importance*.csv"))
    
    if not importance_files:
        print("⚠️  Файлы с важностью признаков не найдены")
        return
    
    importance_data = {}
    for file in importance_files:
        df = pd.read_csv(file)
        model_name = file.stem.replace('feature_importance_', '').replace('_shap', ' (SHAP)')
        if 'shap' in file.stem:
            model_name = model_name.replace(' (SHAP)', '') + ' (SHAP)'
        else:
            model_name = model_name + ' (Vanilla)'
        
        importance_data[model_name] = df['importance'].values
    
    # Создаем визуализацию
    visualize_feature_importance(importance_data, output_path, feature_names=feature_names)
    
    # Создаем текстовый отчет
    report_path = output_path.replace('.png', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("📊 ОТЧЕТ ОБ ИНТЕРПРЕТИРУЕМОСТИ МОДЕЛИ\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, importance in importance_data.items():
            f.write(f"\n{model_name}:\n")
            f.write("-" * 80 + "\n")
            
            # Gini coefficient
            gini = compute_gini_coefficient(importance)
            f.write(f"  Коэффициент Джини: {gini:.4f}\n")
            if gini > 0.3:
                f.write(f"    ✅ Хорошая интерпретируемость (неравномерное распределение)\n")
            elif gini > 0.2:
                f.write(f"    ⚠️  Средняя интерпретируемость\n")
            else:
                f.write(f"    ❌ Плохая интерпретируемость (равномерное распределение)\n")
            
            # Энтропия
            entropy = compute_entropy(importance)
            f.write(f"  Нормализованная энтропия: {entropy:.4f}\n")
            if entropy < 0.4:
                f.write(f"    ✅ Хорошая разреженность (несколько признаков важны)\n")
            elif entropy < 0.6:
                f.write(f"    ⚠️  Средняя разреженность\n")
            else:
                f.write(f"    ❌ Плохая разреженность (все признаки одинаково важны)\n")
            
            # Топ-3 признака
            top_indices = np.argsort(importance)[-3:][::-1]
            f.write(f"  Топ-3 важных признака:\n")
            for i, idx in enumerate(top_indices, 1):
                if feature_names and idx < len(feature_names):
                    feat_name = feature_names[idx]
                else:
                    feat_name = f'X{idx+1}'
                feat_importance = importance[idx]
                f.write(f"    {i}. {feat_name}: {feat_importance:.4f} ({feat_importance*100:.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("💡 ИНТЕРПРЕТАЦИЯ МЕТРИК:\n")
        f.write("=" * 80 + "\n")
        f.write("  • Коэффициент Джини: измеряет неравномерность распределения важности\n")
        f.write("    - Высокий Gini (>0.3) = несколько признаков очень важны = хорошая интерпретируемость\n")
        f.write("    - Низкий Gini (<0.2) = все признаки одинаково важны = плохая интерпретируемость\n")
        f.write("\n")
        f.write("  • Энтропия Шеннона: измеряет разреженность важности\n")
        f.write("    - Низкая энтропия (<0.4) = несколько признаков доминируют = хорошая интерпретируемость\n")
        f.write("    - Высокая энтропия (>0.6) = все признаки одинаково важны = плохая интерпретируемость\n")
    
    print(f"✅ Отчет сохранен: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Визуализация интерпретируемости модели")
    parser.add_argument("--results-dir", default="results", help="Директория с результатами")
    parser.add_argument("--output", default="results/interpretability_comparison.png", 
                       help="Путь для сохранения графика")
    parser.add_argument("--config", help="Путь к YAML конфигурации (для имен признаков)")
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 ВИЗУАЛИЗАЦИЯ ИНТЕРПРЕТИРУЕМОСТИ")
    print("=" * 80)
    
    feature_names = None
    if args.config:
        config = load_config(args.config)
        dataset_config = config.get('dataset', {})
        if dataset_config.get('feature_columns'):
            feature_names = list(dataset_config.get('feature_columns'))
        elif dataset_config.get('feature_prefix') is not None and dataset_config.get('feature_count') is not None:
            start = int(dataset_config.get('feature_index_start', 1))
            count = int(dataset_config.get('feature_count'))
            prefix = dataset_config.get('feature_prefix')
            feature_names = [f"{prefix}{i}" for i in range(start, start + count)]
    create_interpretability_report(args.results_dir, args.output, feature_names=feature_names)
    
    print("\n✅ Готово!")


if __name__ == "__main__":
    main()
