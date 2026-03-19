"""
Утилиты для загрузки и предобработки данных
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger


def load_data(data_path, drop_index=True):
    """
    Загрузка данных из CSV файла
    
    Args:
        data_path: Путь к CSV файлу
        drop_index: Удалять ли индексный столбец
        
    Returns:
        pd.DataFrame: Загруженные данные
    """
    logger = get_logger("anfis_shap.data_loader")
    logger.info(f"Загрузка данных из {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл не найден: {data_path}")
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Файл не найден: {e}")
    except pd.errors.EmptyDataError:
        raise
    except Exception as e:
        raise Exception(f"Ошибка при чтении CSV файла: {e}")

    if data.empty:
        raise ValueError(f"CSV файл пуст: {data_path}")

    data.dropna(inplace=True)
    
    if drop_index and data.columns[0] == 'Unnamed: 0':
        data.drop(columns=[data.columns[0]], inplace=True)
    
    logger.info(f"Загружено {len(data)} образцов, {len(data.columns)} столбцов")
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
            logger = get_logger("anfis_shap.data_loader")
            logger.info(f"Использую подвыборку train_limit={train_limit} (стратегия: {strategy})")

    train_fraction = dataset_config.get('train_fraction')
    if train_fraction is not None:
        if not 0 < train_fraction <= 1:
            raise ValueError("dataset.train_fraction должен быть в диапазоне (0, 1]")
        n_fraction = max(int(len(train_data) * float(train_fraction)), 1)
        train_data = train_data.sample(n=n_fraction, random_state=random_state).reset_index(drop=True)
        logger = get_logger("anfis_shap.data_loader")
        logger.info(f"Использую долю train_fraction={train_fraction:.3f} → {n_fraction} образцов")

    mix_with_real = dataset_config.get('mix_with_real', False)
    mix_ratio = dataset_config.get('mix_ratio', 0.0)

    if mix_with_real and mix_ratio > 0:
        real_path = dataset_config.get('validation_data')
        logger = get_logger("anfis_shap.data_loader")
        if real_path and os.path.exists(real_path):
            logger.info("Смешивание с реальными данными...")
            real_data = load_data(real_path)
            if len(real_data) == 0:
                logger.warning("Реальные данные пустые - смешивание пропущено")
                return train_data

            n_total = len(train_data)
            n_real = min(max(int(n_total * mix_ratio), 1), len(real_data))
            n_generated = max(n_total - n_real, 0)

            generated_sample = train_data.sample(n=n_generated, random_state=random_state)
            real_sample = real_data.sample(n=n_real, random_state=random_state)

            combined = pd.concat([generated_sample, real_sample], ignore_index=True)
            combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)

            logger.info(f"Использовано {n_generated} сгенерированных и {n_real} реальных спектров")
            return combined
        else:
            logger.warning("Путь к реальным данным не указан - смешивание пропущено")

    return train_data


def _resolve_columns_by_indices(columns, indices):
    return [columns[int(idx)] for idx in indices]


def resolve_feature_columns(data, dataset_config=None):
    """
    Определение колонок признаков из данных и конфигурации.
    Поддерживаются:
      - feature_columns: явный список
      - feature_indices: индексы колонок
      - feature_prefix + feature_count (+ feature_index_start)
      - feature_regex
    По умолчанию: regex 'Q'
    """
    if dataset_config is None:
        return data.filter(regex='Q', axis=1).columns.tolist()

    feature_columns = dataset_config.get('feature_columns')
    if feature_columns:
        return list(feature_columns)

    feature_indices = dataset_config.get('feature_indices')
    if feature_indices:
        return _resolve_columns_by_indices(data.columns, feature_indices)

    feature_prefix = dataset_config.get('feature_prefix')
    feature_count = dataset_config.get('feature_count')
    if feature_prefix is not None and feature_count is not None:
        start = int(dataset_config.get('feature_index_start', 1))
        count = int(feature_count)
        return [f"{feature_prefix}{i}" for i in range(start, start + count)]

    feature_regex = dataset_config.get('feature_regex')
    if feature_regex:
        return data.filter(regex=feature_regex, axis=1).columns.tolist()

    return data.filter(regex='Q', axis=1).columns.tolist()


def resolve_target_columns(data, dataset_config=None):
    """
    Определение колонок целей из данных и конфигурации.
    Поддерживаются:
      - target_columns: явный список
      - target_indices: индексы колонок
      - target_prefix + target_count (+ target_index_start)
      - target_range: [start, end) по позициям колонок
      - target_regex
    По умолчанию: первые 60 колонок.
    """
    if dataset_config is None:
        feature_columns = set(resolve_feature_columns(data, dataset_config))
        target_columns = [col for col in data.columns if col not in feature_columns]
        return target_columns[:60]

    target_columns = dataset_config.get('target_columns')
    if target_columns:
        return list(target_columns)

    target_indices = dataset_config.get('target_indices')
    if target_indices:
        return _resolve_columns_by_indices(data.columns, target_indices)

    target_prefix = dataset_config.get('target_prefix')
    target_count = dataset_config.get('target_count')
    if target_prefix is not None and target_count is not None:
        start = int(dataset_config.get('target_index_start', 1))
        count = int(target_count)
        return [f"{target_prefix}{i}" for i in range(start, start + count)]

    target_range = dataset_config.get('target_range')
    if target_range:
        start, end = int(target_range[0]), int(target_range[1])
        return data.columns[start:end].tolist()

    target_regex = dataset_config.get('target_regex')
    if target_regex:
        return data.filter(regex=target_regex, axis=1).columns.tolist()

    return data.columns[:60].tolist()


def resolve_feature_count(dataset_config=None):
    if dataset_config is None:
        return None
    if dataset_config.get('feature_columns'):
        return len(dataset_config.get('feature_columns'))
    if dataset_config.get('feature_indices'):
        return len(dataset_config.get('feature_indices'))
    if dataset_config.get('feature_count') is not None:
        return int(dataset_config.get('feature_count'))
    return None


def resolve_target_count(dataset_config=None):
    if dataset_config is None:
        return None
    if dataset_config.get('target_columns'):
        return len(dataset_config.get('target_columns'))
    if dataset_config.get('target_indices'):
        return len(dataset_config.get('target_indices'))
    if dataset_config.get('target_count') is not None:
        return int(dataset_config.get('target_count'))
    target_range = dataset_config.get('target_range')
    if target_range:
        return int(target_range[1]) - int(target_range[0])
    return None


def resolve_feature_columns_from_config(dataset_config=None):
    if dataset_config is None:
        return None
    feature_columns = dataset_config.get('feature_columns')
    if feature_columns:
        return list(feature_columns)
    feature_prefix = dataset_config.get('feature_prefix')
    feature_count = dataset_config.get('feature_count')
    if feature_prefix is not None and feature_count is not None:
        start = int(dataset_config.get('feature_index_start', 1))
        count = int(feature_count)
        return [f"{feature_prefix}{i}" for i in range(start, start + count)]
    if feature_count is not None:
        count = int(feature_count)
        return [f"X{i+1}" for i in range(count)]
    return None


def resolve_target_columns_from_config(dataset_config=None):
    if dataset_config is None:
        return None
    target_columns = dataset_config.get('target_columns')
    if target_columns:
        return list(target_columns)
    target_prefix = dataset_config.get('target_prefix')
    target_count = dataset_config.get('target_count')
    if target_prefix is not None and target_count is not None:
        start = int(dataset_config.get('target_index_start', 1))
        count = int(target_count)
        return [f"{target_prefix}{i}" for i in range(start, start + count)]
    if target_count is not None:
        count = int(target_count)
        return [f"E{i+1}" for i in range(count)]
    return None


def prepare_features_targets(data, normalize_sum=False, dataset_config=None):
    """
    Подготовка признаков и целевых переменных
    
    Args:
        data: DataFrame с данными
        normalize_sum: Применять ли нормализацию на SUM
        
    Returns:
        tuple: (X, y, SUM) где SUM - суммы показаний (если normalize_sum=True)
    """
    if data is None or len(data.columns) == 0:
        raise KeyError("Пустой DataFrame: невозможно определить признаки и цели")

    feature_columns = resolve_feature_columns(data, dataset_config)
    target_columns = resolve_target_columns(data, dataset_config)

    if len(feature_columns) == 0:
        raise KeyError("Не удалось определить ни одной колонки признаков")
    if len(target_columns) == 0:
        raise KeyError("Не удалось определить ни одной колонки целей")

    # Проверяем колонки
    missing_features = [col for col in feature_columns if col not in data.columns]
    missing_targets = [col for col in target_columns if col not in data.columns]
    if missing_features:
        raise ValueError(f"Не найдены колонки признаков: {missing_features}")
    if missing_targets:
        raise ValueError(f"Не найдены колонки целей: {missing_targets}")

    overlap = set(feature_columns) & set(target_columns)
    if overlap:
        raise ValueError(f"Колонки признаков и целей пересекаются: {sorted(overlap)}")

    # Признаки и целевая переменная по конфигурации
    X = data[feature_columns].copy()
    y = data[target_columns].copy()
    
    feature_names = X.columns.tolist()
    
    logger = get_logger("anfis_shap.data_loader")
    logger.info(f"Признаки: {len(feature_names)} ({', '.join(feature_names)})")
    logger.info(f"Целевые переменные: {y.shape[1]} бинов спектра")
    
    SUM = None
    
    if normalize_sum:
        # Вычисляем SUM для каждого образца
        # Сохраняем как Series с теми же индексами, что и X
        SUM = X.sum(axis=1)

        # Избегаем деления на ноль
        zero_mask = SUM == 0
        if zero_mask.any():
            eps = np.finfo(float).eps
            logger = get_logger("anfis_shap.data_loader")
            logger.warning(f"Найдено {zero_mask.sum()} образцов с SUM=0. Заменяю на {eps:.2e}")
            SUM = SUM.mask(zero_mask, eps)
        
        # Нормализуем входы
        X_normalized = X.div(SUM, axis=0)
        
        # Нормализуем выходы
        y_normalized = y.div(SUM, axis=0)
        
        logger = get_logger("anfis_shap.data_loader")
        logger.info("Применена нормализация на SUM")
        logger.info(f"Средний SUM: {SUM.mean():.4f}, Мин: {SUM.min():.4f}, Макс: {SUM.max():.4f}")
        
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
    
    logger = get_logger("anfis_shap.data_loader")
    logger.info("Данные разделены:")
    logger.info(f"Train: {X_train.shape[0]} образцов")
    logger.info(f"Test: {X_test.shape[0]} образцов")
    
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
    if SUM_array.ndim > 1:
        SUM_array = SUM_array.reshape(-1)
    
    # Убеждаемся, что формы совместимы
    if y_pred.ndim == 1:
        if SUM_array.size not in (1, y_pred.shape[0]):
            raise ValueError(
                f"Несовместимые формы для денормализации: y_pred={y_pred.shape}, SUM={SUM_array.shape}"
            )
        if SUM_array.size == 1:
            return y_pred * SUM_array.item()
        return y_pred * SUM_array
    elif y_pred.ndim == 2:
        if SUM_array.size != y_pred.shape[0]:
            raise ValueError(
                f"Несовместимые формы для денормализации: y_pred={y_pred.shape}, SUM={SUM_array.shape}"
            )
        return y_pred * SUM_array[:, np.newaxis]
    else:
        raise ValueError(f"Неожиданная размерность предсказаний: {y_pred.ndim}")


def load_validation_data(data_path, normalize_sum=False, dataset_config=None):
    """
    Загрузка валидационных данных (реальные спектры)
    
    Args:
        data_path: Путь к файлу с валидационными данными
        normalize_sum: Применять ли нормализацию на SUM
        
    Returns:
        tuple: (X_val, y_val, SUM_val)
    """
    data = load_data(data_path, drop_index=True)
    X_val, y_val, SUM_val = prepare_features_targets(
        data, normalize_sum=normalize_sum, dataset_config=dataset_config
    )
    
    logger = get_logger("anfis_shap.data_loader")
    logger.info(f"Валидационные данные: {len(X_val)} образцов")
    return X_val, y_val, SUM_val
