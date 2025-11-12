"""
Утилиты для загрузки и предобработки данных
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(data_path, drop_index=True):
    """
    Загрузка данных из CSV файла
    
    Args:
        data_path: Путь к CSV файлу
        drop_index: Удалять ли индексный столбец
        
    Returns:
        pd.DataFrame: Загруженные данные
    """
    print(f"📂 Загрузка данных из {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл не найден: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл не найден: {e}")
    except Exception as e:
        raise Exception(f"Ошибка при чтении CSV файла: {e}")
    data.dropna(inplace=True)
    
    if drop_index and data.columns[0] == 'Unnamed: 0':
        data.drop(columns=data.columns[0], axis=1, inplace=True)
    
    print(f"✅ Загружено {len(data)} образцов, {len(data.columns)} столбцов")
    return data


def load_training_dataset(dataset_config):
    """
    Загрузка обучающего датасета с учетом смешивания

    Args:
        dataset_config: Раздел конфигурации dataset

    Returns:
        pd.DataFrame: комбинированные данные для обучения
    """
    train_data = load_data(dataset_config['train_data'])

    random_state = dataset_config.get('random_state', 42)

    # Ограничение размера обучающего датасета для отладки/экспериментов
    train_limit = dataset_config.get('train_limit')
    if train_limit is not None:
        train_limit = int(train_limit)
        if train_limit <= 0:
            raise ValueError("dataset.train_limit должен быть положительным числом")
        if train_limit < len(train_data):
            strategy = dataset_config.get('train_sample_strategy', 'head')
            if strategy == 'random':
                train_data = train_data.sample(n=train_limit, random_state=random_state).reset_index(drop=True)
            else:
                train_data = train_data.iloc[:train_limit].reset_index(drop=True)
            print(f"🔬 Использую подвыборку train_limit={train_limit} (стратегия: {strategy})")

    train_fraction = dataset_config.get('train_fraction')
    if train_fraction is not None:
        if not 0 < train_fraction <= 1:
            raise ValueError("dataset.train_fraction должен быть в диапазоне (0, 1]")
        n_fraction = max(int(len(train_data) * float(train_fraction)), 1)
        train_data = train_data.sample(n=n_fraction, random_state=random_state).reset_index(drop=True)
        print(f"🔬 Использую долю train_fraction={train_fraction:.3f} → {n_fraction} образцов")

    mix_with_real = dataset_config.get('mix_with_real', False)
    mix_ratio = dataset_config.get('mix_ratio', 0.0)

    if mix_with_real and mix_ratio > 0:
        real_path = dataset_config.get('validation_data')
        if real_path and os.path.exists(real_path):
            print("\n🔄 Смешивание с реальными данными...")
            real_data = load_data(real_path)
            if len(real_data) == 0:
                print("⚠️  Реальные данные пустые - смешивание пропущено")
                return train_data

            n_total = len(train_data)
            n_real = min(max(int(n_total * mix_ratio), 1), len(real_data))
            n_generated = max(n_total - n_real, 0)

            generated_sample = train_data.sample(n=n_generated, random_state=random_state)
            real_sample = real_data.sample(n=n_real, random_state=random_state)

            combined = pd.concat([generated_sample, real_sample], ignore_index=True)
            combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

            print(f"   ▶️ Использовано {n_generated} сгенерированных и {n_real} реальных спектров")
            return combined
        else:
            print("⚠️  Путь к реальным данным не указан - смешивание пропущено")

    return train_data


def prepare_features_targets(data, normalize_sum=False):
    """
    Подготовка признаков и целевых переменных
    
    Args:
        data: DataFrame с данными
        normalize_sum: Применять ли нормализацию на SUM
        
    Returns:
        tuple: (X, y, SUM) где SUM - суммы показаний (если normalize_sum=True)
    """
    # Признаки: показания детекторов Q1-Q10
    X = data.filter(regex='Q', axis=1).copy()
    
    # Целевая переменная: спектр (первые 60 столбцов)
    y = data.iloc[:, 0:60].copy()
    
    feature_names = X.columns.tolist()
    
    print(f"📊 Признаки: {len(feature_names)} ({', '.join(feature_names)})")
    print(f"📊 Целевые переменные: {y.shape[1]} бинов спектра")
    
    SUM = None
    
    if normalize_sum:
        # Вычисляем SUM для каждого образца
        # Сохраняем как Series с теми же индексами, что и X
        SUM = X.sum(axis=1)

        # Избегаем деления на ноль
        zero_mask = SUM == 0
        if zero_mask.any():
            eps = np.finfo(float).eps
            print(f"⚠️  Найдено {zero_mask.sum()} образцов с SUM=0. Заменяю на {eps:.2e}")
            SUM = SUM.mask(zero_mask, eps)
        
        # Нормализуем входы
        X_normalized = X.div(SUM, axis=0)
        
        # Нормализуем выходы
        y_normalized = y.div(SUM, axis=0)
        
        print(f"✅ Применена нормализация на SUM")
        print(f"   Средний SUM: {SUM.mean():.4f}, Мин: {SUM.min():.4f}, Макс: {SUM.max():.4f}")
        
        return X_normalized, y_normalized, SUM
    
    return X, y, SUM


def split_data(X, y, test_size=0.25, random_state=42):
    """
    Разделение данных на train/test
    
    Args:
        X: Признаки
        y: Целевые переменные
        test_size: Доля тестовой выборки
        random_state: Seed для воспроизводимости
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"✅ Данные разделены:")
    print(f"   Train: {X_train.shape[0]} образцов")
    print(f"   Test: {X_test.shape[0]} образцов")
    
    return X_train, X_test, y_train, y_test


def denormalize_predictions(y_pred_normalized, SUM):
    """
    Денормализация предсказаний (умножение на SUM)
    
    Args:
        y_pred_normalized: Нормализованные предсказания
        SUM: Суммы показаний
        
    Returns:
        np.array: Денормализованные предсказания
    """
    if SUM is None:
        return y_pred_normalized
    
    y_pred = np.array(y_pred_normalized)
    SUM_array = np.array(SUM)
    
    # Убеждаемся, что формы совместимы
    if y_pred.ndim == 1:
        # Одномерный массив - умножаем каждый элемент на соответствующий SUM
        return y_pred * SUM_array
    elif y_pred.ndim == 2:
        # Двумерный массив (samples, features)
        # Умножаем каждую строку на соответствующий SUM
        return y_pred * SUM_array[:, np.newaxis]
    else:
        raise ValueError(f"Неожиданная размерность предсказаний: {y_pred.ndim}")


def load_validation_data(data_path, normalize_sum=False):
    """
    Загрузка валидационных данных (реальные спектры)
    
    Args:
        data_path: Путь к файлу с валидационными данными
        normalize_sum: Применять ли нормализацию на SUM
        
    Returns:
        tuple: (X_val, y_val, SUM_val)
    """
    data = load_data(data_path, drop_index=True)
    X_val, y_val, SUM_val = prepare_features_targets(data, normalize_sum=normalize_sum)
    
    print(f"✅ Валидационные данные: {len(X_val)} образцов")
    return X_val, y_val, SUM_val

