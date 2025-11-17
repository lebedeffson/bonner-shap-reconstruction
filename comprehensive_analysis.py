#!/usr/bin/env python3
"""
Комплексный анализ всех результатов обучения
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys


def load_all_results(results_dir="results"):
    """Загружает все результаты обучения"""
    results_dir = Path(results_dir)
    result_files = sorted(results_dir.glob("training_summary_*.json"), reverse=True)
    
    all_results = []
    for f in result_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                data['_file'] = f.name
                data['_timestamp'] = f.stat().st_mtime
                all_results.append(data)
        except Exception as e:
            print(f"⚠️  Ошибка при загрузке {f.name}: {e}")
    
    return all_results


def compute_interpretability_metrics(results_dir, tag):
    """Вычисляет метрики интерпретируемости"""
    # Ищем файлы важности признаков с разными паттернами
    importance_files = (
        list(Path(results_dir).glob(f"*{tag}*feature_importance*.csv")) +
        list(Path(results_dir).glob(f"feature_importance*{tag}*.csv")) +
        list(Path(results_dir).glob(f"feature_importance_shap*{tag}*.csv"))
    )
    
    if not importance_files:
        return None
    
    try:
        df = pd.read_csv(importance_files[0], index_col=0)
        
        # Определяем колонку с важностью
        if 'importance' in df.columns:
            importance_col = 'importance'
        elif len(df.columns) == 1:
            importance_col = df.columns[0]
        else:
            # Ищем колонку с числовыми значениями
            for col in df.columns:
                if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                    importance_col = col
                    break
            else:
                return None
        
        importance = df[importance_col].values
        
        # Нормализуем
        importance = np.abs(importance)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        
        # Gini coefficient
        sorted_imp = np.sort(importance)
        n = len(sorted_imp)
        indices = np.arange(1, n + 1)
        gini = 1.0 - 2.0 * np.sum(indices * sorted_imp) / (n * np.sum(sorted_imp))
        gini = max(0.0, min(1.0, gini))
        
        # Энтропия
        entropy = -np.sum(importance[importance > 1e-10] * np.log(importance[importance > 1e-10] + 1e-10))
        max_entropy = np.log(len(importance))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        # Топ-3 признака
        top3_indices = np.argsort(importance)[-3:][::-1]
        top3_features = [f'Q{i+1}' for i in top3_indices]
        top3_values = importance[top3_indices]
        
        return {
            'gini': gini,
            'entropy': normalized_entropy,
            'top3_features': top3_features,
            'top3_values': top3_values.tolist()
        }
    except Exception as e:
        print(f"⚠️  Ошибка при вычислении метрик интерпретируемости для {tag}: {e}")
        return None


def analyze_all_results():
    """Комплексный анализ всех результатов"""
    print("=" * 100)
    print("📊 КОМПЛЕКСНЫЙ АНАЛИЗ ВСЕХ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
    print("=" * 100)
    print(f"Время анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results_dir = Path("results")
    all_results = load_all_results(results_dir)
    
    if not all_results:
        print("⚠️  Результаты не найдены")
        return
    
    print(f"Найдено результатов: {len(all_results)}\n")
    
    # Создаем таблицу сравнения
    comparison_data = []
    
    for result in all_results:
        tag = result.get('tag', 'unknown')
        metrics = result.get('metrics', {})
        interpretability = compute_interpretability_metrics(results_dir, tag)
        
        comparison_data.append({
            'tag': tag,
            'timestamp': result.get('timestamp', ''),
            'r2': metrics.get('r2', 0),
            'mse': metrics.get('mse', 0),
            'rmse': metrics.get('rmse', 0),
            'mae': metrics.get('mae', 0),
            'training_time': result.get('training_time_total', 0),
            'shap_time': result.get('training_time_shap', 0),
            'gini': interpretability['gini'] if interpretability else 0,
            'entropy': interpretability['entropy'] if interpretability else 1,
            'top3_features': ', '.join(interpretability['top3_features']) if interpretability else 'N/A',
            'shap_enabled': result.get('shap_config_enabled', False),
            'shap_applied': result.get('shap_applied', False),
            'train_size': result.get('train_size', 0),
            'shap_train_size': result.get('shap_train_size', 0),
        })
    
    df = pd.DataFrame(comparison_data)
    
    # 1. ОСНОВНЫЕ МЕТРИКИ
    print("=" * 100)
    print("1️⃣ СРАВНЕНИЕ МЕТРИК ТОЧНОСТИ")
    print("=" * 100)
    
    print("\n📊 Таблица сравнения:")
    print(df[['tag', 'r2', 'mse', 'rmse', 'mae', 'training_time']].to_string(index=False))
    
    # Лучшие результаты
    best_r2_idx = df['r2'].idxmax()
    best_mse_idx = df['mse'].idxmin()
    best_time_idx = df['training_time'].idxmin()
    
    print("\n🏆 ЛУЧШИЕ РЕЗУЛЬТАТЫ:")
    print(f"   • Лучший R²: {df.loc[best_r2_idx, 'tag']} (R² = {df.loc[best_r2_idx, 'r2']:.6f})")
    print(f"   • Лучший MSE: {df.loc[best_mse_idx, 'tag']} (MSE = {df.loc[best_mse_idx, 'mse']:.6f})")
    print(f"   • Самое быстрое обучение: {df.loc[best_time_idx, 'tag']} ({df.loc[best_time_idx, 'training_time']:.1f} сек)")
    
    # 2. ИНТЕРПРЕТИРУЕМОСТЬ
    print("\n" + "=" * 100)
    print("2️⃣ СРАВНЕНИЕ ИНТЕРПРЕТИРУЕМОСТИ")
    print("=" * 100)
    
    interpretability_df = df[df['gini'] > 0][['tag', 'gini', 'entropy', 'top3_features']]
    
    if not interpretability_df.empty:
        print("\n📊 Таблица интерпретируемости:")
        print(interpretability_df.to_string(index=False))
        
        best_gini_idx = interpretability_df['gini'].idxmax()
        best_entropy_idx = interpretability_df['entropy'].idxmin()
        
        print("\n🏆 ЛУЧШАЯ ИНТЕРПРЕТИРУЕМОСТЬ:")
        print(f"   • Лучший Gini: {interpretability_df.loc[best_gini_idx, 'tag']} (Gini = {interpretability_df.loc[best_gini_idx, 'gini']:.4f})")
        print(f"   • Лучшая разреженность: {interpretability_df.loc[best_entropy_idx, 'tag']} (Entropy = {interpretability_df.loc[best_entropy_idx, 'entropy']:.4f})")
    
    # 3. ДЕТАЛЬНЫЙ АНАЛИЗ КАЖДОГО РЕЗУЛЬТАТА
    print("\n" + "=" * 100)
    print("3️⃣ ДЕТАЛЬНЫЙ АНАЛИЗ КАЖДОГО РЕЗУЛЬТАТА")
    print("=" * 100)
    
    for i, result in enumerate(all_results, 1):
        tag = result.get('tag', 'unknown')
        metrics = result.get('metrics', {})
        interpretability = compute_interpretability_metrics(results_dir, tag)
        
        print(f"\n{'─' * 100}")
        print(f"📋 РЕЗУЛЬТАТ #{i}: {tag}")
        print(f"{'─' * 100}")
        
        print(f"\n📊 Метрики точности:")
        print(f"   • R²: {metrics.get('r2', 0):.6f}")
        print(f"   • MSE: {metrics.get('mse', 0):.6f}")
        print(f"   • RMSE: {metrics.get('rmse', 0):.6f}")
        print(f"   • MAE: {metrics.get('mae', 0):.6f}")
        
        # Оценка качества
        r2 = metrics.get('r2', 0)
        if r2 > 0.6:
            print("   🎉 ОТЛИЧНЫЙ РЕЗУЛЬТАТ!")
        elif r2 > 0.4:
            print("   ✅ ХОРОШИЙ РЕЗУЛЬТАТ!")
        elif r2 > 0.0:
            print("   ⚠️  ПРИЕМЛЕМЫЙ РЕЗУЛЬТАТ")
        else:
            print("   ❌ ТРЕБУЕТСЯ УЛУЧШЕНИЕ")
        
        if interpretability:
            print(f"\n📈 Интерпретируемость:")
            print(f"   • Коэффициент Джини: {interpretability['gini']:.4f}")
            print(f"   • Нормализованная энтропия: {interpretability['entropy']:.4f}")
            print(f"   • Топ-3 признака: {', '.join(interpretability['top3_features'])}")
            
            if interpretability['gini'] > 0.3 and interpretability['entropy'] < 0.4:
                print("   ✅ ОТЛИЧНАЯ интерпретируемость!")
            elif interpretability['gini'] > 0.2 or interpretability['entropy'] < 0.6:
                print("   ✅ ХОРОШАЯ интерпретируемость")
            else:
                print("   ⚠️  ТРЕБУЕТСЯ УЛУЧШЕНИЕ интерпретируемости")
        
        print(f"\n⏱️  Время обучения:")
        total_time = result.get('training_time_total', 0)
        shap_time = result.get('training_time_shap', 0)
        print(f"   • Общее время: {total_time:.2f} сек ({total_time/60:.2f} мин)")
        if shap_time > 0:
            print(f"   • SHAP регуляризация: {shap_time:.2f} сек ({shap_time/60:.2f} мин)")
        
        print(f"\n⚙️  Настройки:")
        print(f"   • SHAP включен: {result.get('shap_config_enabled', False)}")
        print(f"   • SHAP применен: {result.get('shap_applied', False)}")
        print(f"   • Размер обучающей выборки: {result.get('train_size', 0):,}")
        print(f"   • Размер SHAP выборки: {result.get('shap_train_size', 0):,}")
        
        # SHAP история
        if result.get('shap_files', {}).get('history'):
            history_file = results_dir / result['shap_files']['history']
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        shap_history = json.load(f)
                    
                    print(f"\n📉 Динамика SHAP:")
                    if 'total_loss' in shap_history and shap_history['total_loss']:
                        total_losses = shap_history['total_loss']
                        print(f"   • Начальный loss: {total_losses[0]:.6f}")
                        print(f"   • Финальный loss: {total_losses[-1]:.6f}")
                        improvement = (total_losses[0] - total_losses[-1]) / total_losses[0] * 100 if total_losses[0] > 0 else 0
                        print(f"   • Улучшение: {improvement:.2f}%")
                    
                    if 'main_loss' in shap_history and shap_history['main_loss']:
                        main_losses = shap_history['main_loss']
                        print(f"   • Main loss (начальный): {main_losses[0]:.6f}")
                        print(f"   • Main loss (финальный): {main_losses[-1]:.6f}")
                except Exception as e:
                    print(f"   ⚠️  Не удалось загрузить SHAP историю: {e}")
    
    # 4. ИТОГОВЫЕ РЕКОМЕНДАЦИИ
    print("\n" + "=" * 100)
    print("4️⃣ ИТОГОВЫЕ РЕКОМЕНДАЦИИ")
    print("=" * 100)
    
    # Находим лучший результат по комбинации метрик
    df['combined_score'] = (
        df['r2'] * 0.5 +  # Точность важнее
        (df['gini'] / 1.0) * 0.3 +  # Интерпретируемость
        (1.0 - df['entropy']) * 0.2  # Разреженность
    )
    
    best_combined_idx = df['combined_score'].idxmax()
    best_tag = df.loc[best_combined_idx, 'tag']
    
    print(f"\n🏆 РЕКОМЕНДУЕМАЯ МОДЕЛЬ: {best_tag}")
    print(f"   • Комбинированный score: {df.loc[best_combined_idx, 'combined_score']:.4f}")
    print(f"   • R²: {df.loc[best_combined_idx, 'r2']:.6f}")
    print(f"   • Gini: {df.loc[best_combined_idx, 'gini']:.4f}")
    print(f"   • Entropy: {df.loc[best_combined_idx, 'entropy']:.4f}")
    
    # Сравнение с другими
    print(f"\n📊 Сравнение с другими моделями:")
    for idx, row in df.iterrows():
        if row['tag'] != best_tag:
            r2_diff = row['r2'] - df.loc[best_combined_idx, 'r2']
            print(f"   • {row['tag']}: R² {r2_diff:+.6f}, Gini {row['gini'] - df.loc[best_combined_idx, 'gini']:+.4f}")
    
    print("\n" + "=" * 100)
    print("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 100)
    
    # Сохраняем таблицу сравнения
    output_file = results_dir / f"comparison_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Таблица сравнения сохранена: {output_file}")


def cleanup_old_results(keep_tags=None):
    """Удаляет старые результаты"""
    if keep_tags is None:
        keep_tags = []
    
    results_dir = Path('results')
    if not results_dir.exists():
        return
    
    print("\n" + "=" * 100)
    print("🧹 ОЧИСТКА СТАРЫХ РЕЗУЛЬТАТОВ")
    print("=" * 100)
    
    # Находим все файлы результатов
    all_files = {
        'summaries': list(results_dir.glob('training_summary_*.json')),
        'models': list(results_dir.glob('anfis_model_state_*.pt')),
        'predictions': list(results_dir.glob('predictions_*.npy')),
        'targets': list(results_dir.glob('targets_*.npy')),
        'shap_history': list(results_dir.glob('shap_history_*.json')),
        'feature_importance': list(results_dir.glob('*feature_importance*.csv')),
        'metrics_csv': list(results_dir.glob('metrics_*.csv')),
        'samples': list(results_dir.glob('samples_*.npy')),
    }
    
    # Определяем файлы для сохранения
    files_to_keep = set()
    
    # Сохраняем файлы с указанными тегами
    for tag in keep_tags:
        for file_type, files in all_files.items():
            for f in files:
                if tag in f.name:
                    files_to_keep.add(f)
    
    # Сохраняем последние файлы каждого типа (если нет тегов)
    if not keep_tags:
        for file_type, files in all_files.items():
            if files:
                sorted_files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
                # Оставляем последние 3 файла каждого типа
                for f in sorted_files[:3]:
                    files_to_keep.add(f)
    
    # Удаляем старые файлы
    deleted_count = 0
    total_size = 0
    
    for file_type, files in all_files.items():
        for f in files:
            if f not in files_to_keep:
                try:
                    size = f.stat().st_size
                    f.unlink()
                    deleted_count += 1
                    total_size += size
                    print(f"   🗑️  Удален: {f.name}")
                except Exception as e:
                    print(f"   ⚠️  Не удалось удалить {f.name}: {e}")
    
    print(f"\n✅ Удалено файлов: {deleted_count}")
    print(f"   Освобождено места: {total_size / 1024 / 1024:.2f} MB")
    
    # Очистка старых спектров
    spectra_dirs = [
        results_dir / 'spectra' / 'normalized',
        results_dir / 'spectra' / 'saved'
    ]
    
    for spectra_dir in spectra_dirs:
        if spectra_dir.exists():
            spectrum_files = list(spectra_dir.glob('*.png'))
            if spectrum_files:
                sorted_spectra = sorted(spectrum_files, key=lambda p: p.stat().st_mtime, reverse=True)
                # Оставляем последние 10 файлов
                for f in sorted_spectra[10:]:
                    try:
                        f.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"   ⚠️  Не удалось удалить {f.name}: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Комплексный анализ всех результатов")
    parser.add_argument("--cleanup", action="store_true", help="Удалить старые результаты")
    parser.add_argument("--keep-tags", nargs="+", help="Теги для сохранения при очистке")
    args = parser.parse_args()
    
    # Анализируем результаты
    analyze_all_results()
    
    # Очищаем старые результаты, если нужно
    if args.cleanup:
        cleanup_old_results(keep_tags=args.keep_tags)


if __name__ == "__main__":
    main()

