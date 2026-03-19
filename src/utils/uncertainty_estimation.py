"""
Оценка неопределённости спектра методом Monte Carlo
Анализирует влияние ошибок в исходных данных (Q) на конечный результат (спектр)
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict
from src.utils.logger import get_logger

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Заглушка для tqdm если не установлен
    def tqdm(iterable, desc=None):
        return iterable


class UncertaintyEstimator:
    """
    Класс для оценки неопределённости предсказаний модели
    """
    
    def __init__(self, model, device=None, verbose=True):
        """
        Инициализация оценщика неопределённости
        
        Args:
            model: Обученная модель для предсказаний
            device: Устройство для вычислений (GPU/CPU)
            verbose: Выводить ли информацию о прогрессе
        """
        self.model = model
        self.device = device
        self.verbose = verbose
        self.logger = get_logger("anfis_shap.uncertainty_estimation")
        if not verbose:
            self.logger.setLevel(30)  # WARNING level
        
        # Определяем устройство если не указано
        if self.device is None:
            if hasattr(model, 'device'):
                self.device = model.device
            elif hasattr(model, 'network') and hasattr(model.network, 'parameters'):
                self.device = next(model.network.parameters()).device
            else:
                self.device = torch.device('cpu')
    
    def estimate_uncertainty(
        self,
        X: np.ndarray,
        n_samples: int = 1000,
        error_percent: float = 5.0,
        seed: Optional[int] = None,
        normalize_sum: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Оценивает неопределённость спектра методом Monte Carlo
        
        Метод:
        1. Добавляет случайную ошибку error_percent% (нормальное распределение) к каждому Q
        2. Восстанавливает спектр с помощью модели
        3. Повторяет N раз
        4. Вычисляет статистики (mean, std, min, max, percentiles)
        
        Args:
            X: Входные данные (N_samples, n_features) - значения Q
            n_samples: Количество Monte Carlo семплов
            error_percent: Процент ошибки (стандартное отклонение в процентах)
            seed: Seed для воспроизводимости
            normalize_sum: Нормализовать ли входы по SUM после добавления ошибки
            
        Returns:
            dict: Словарь с результатами:
                - 'mean': Средний спектр
                - 'std': Стандартное отклонение
                - 'min': Минимальные значения
                - 'max': Максимальные значения
                - 'percentiles': Процентили (5, 25, 50, 75, 95)
                - 'base_prediction': Базовое предсказание без ошибок
                - 'all_predictions': Все предсказания (n_samples, N_samples, n_outputs)
        """
        if seed is not None:
            np.random.seed(seed)
        
        X = np.array(X) if not isinstance(X, np.ndarray) else X
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.size == 0 or X.shape[0] == 0:
            empty = np.empty((0, 0), dtype=float)
            empty_percentiles = {p: empty.copy() for p in (5, 25, 50, 75, 95)}
            return {
                'mean': empty.copy(),
                'std': empty.copy(),
                'min': empty.copy(),
                'max': empty.copy(),
                'percentiles': empty_percentiles,
                'base_prediction': empty.copy(),
                'all_predictions': np.empty((0, 0, 0), dtype=float)
            }

        # Базовое предсказание без ошибок
        if normalize_sum:
            sums = np.sum(X, axis=1, keepdims=True)
            sums = np.where(sums == 0, 1e-12, sums)
            X_base = X / sums
            base_prediction = self._predict(X_base)
        else:
            base_prediction = self._predict(X)
        n_outputs = base_prediction.shape[1] if base_prediction.ndim > 1 else 1
        
        self.logger.info("Оценка неопределённости методом Monte Carlo...")
        self.logger.info(f"Количество семплов: {n_samples}")
        self.logger.info(f"Ошибка входных данных: {error_percent}%")
        self.logger.info(f"Размер входных данных: {X.shape}")
        
        # Сохраняем все предсказания
        all_predictions = []
        
        # Monte Carlo цикл
        if self.verbose and TQDM_AVAILABLE:
            iterator = tqdm(range(n_samples), desc="Monte Carlo семплирование")
        else:
            iterator = range(n_samples)
            if self.verbose:
                self.logger.info(f"Выполняется {n_samples} итераций Monte Carlo...")
        
        for i in iterator:
            # Добавляем случайную ошибку к входным данным
            X_perturbed = self._add_measurement_error(X, error_percent)
            if normalize_sum:
                sums = np.sum(X_perturbed, axis=1, keepdims=True)
                sums = np.where(sums == 0, 1e-12, sums)
                X_perturbed = X_perturbed / sums
            
            # Получаем предсказание
            prediction = self._predict(X_perturbed)
            
            # Обрабатываем размерность
            if prediction.ndim == 1:
                prediction = prediction.reshape(-1, 1)
            
            all_predictions.append(prediction)
        
        # Преобразуем в numpy array: (n_samples, N_samples, n_outputs)
        all_predictions = np.array(all_predictions)
        
        # Вычисляем статистики
        mean_prediction = np.mean(all_predictions, axis=0)
        std_prediction = np.std(all_predictions, axis=0)
        min_prediction = np.min(all_predictions, axis=0)
        max_prediction = np.max(all_predictions, axis=0)
        
        # Вычисляем процентили
        percentiles = {
            5: np.percentile(all_predictions, 5, axis=0),
            25: np.percentile(all_predictions, 25, axis=0),
            50: np.percentile(all_predictions, 50, axis=0),
            75: np.percentile(all_predictions, 75, axis=0),
            95: np.percentile(all_predictions, 95, axis=0)
        }
        
        # Обрабатываем базовое предсказание
        if base_prediction.ndim == 1:
            base_prediction = base_prediction.reshape(-1, 1)
        
        results = {
            'mean': mean_prediction,
            'std': std_prediction,
            'min': min_prediction,
            'max': max_prediction,
            'percentiles': percentiles,
            'base_prediction': base_prediction,
            'all_predictions': all_predictions
        }
        
        self.logger.info("Оценка неопределённости завершена")
        self.logger.info(f"Среднее std по всем выходам: {np.mean(std_prediction):.6f}")
        self.logger.info(f"Максимальное std: {np.max(std_prediction):.6f}")
        
        return results
    
    def _add_measurement_error(
        self,
        X: np.ndarray,
        error_percent: float,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Добавляет случайную ошибку измерения к входным данным
        
        Args:
            X: Входные данные
            error_percent: Процент ошибки (стандартное отклонение)
            
        Returns:
            np.ndarray: Данные с добавленной ошибкой
        """
        # Вычисляем стандартное отклонение для каждого признака
        # error_percent% означает, что std = X * error_percent / 100
        std_values = np.abs(X) * (error_percent / 100.0)
        
        # Избегаем деления на ноль
        std_values = np.maximum(std_values, 1e-10)
        
        # Генерируем случайную ошибку из нормального распределения
        if seed is None:
            noise = np.random.normal(loc=0.0, scale=std_values, size=X.shape)
        else:
            rng = np.random.default_rng(seed)
            noise = rng.normal(loc=0.0, scale=std_values, size=X.shape)
        
        # Добавляем ошибку к данным
        X_perturbed = X + noise
        
        # Обрезаем отрицательные значения (если Q не может быть отрицательным)
        X_perturbed = np.maximum(X_perturbed, 0.0)
        
        return X_perturbed
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Получает предсказания модели
        
        Args:
            X: Входные данные
            
        Returns:
            np.ndarray: Предсказания модели
        """
        # Определяем, как вызывать модель
        if hasattr(self.model, 'predict'):
            # Если модель имеет метод predict
            predictions = self.model.predict(X)
        elif hasattr(self.model, 'network'):
            # Если модель обернута (например, BioAnfisRegressor)
            self.model.network.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                predictions = self.model.network(X_tensor)
                if isinstance(predictions, torch.Tensor):
                    if self.device.type == 'cuda':
                        predictions = predictions.cpu().numpy()
                    else:
                        predictions = predictions.numpy()
                # Обрезаем отрицательные значения
                predictions = np.maximum(predictions, 0.0)
        else:
            # Прямой вызов модели как PyTorch модуля
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                predictions = self.model(X_tensor)
                if isinstance(predictions, torch.Tensor):
                    if self.device.type == 'cuda':
                        predictions = predictions.cpu().numpy()
                    else:
                        predictions = predictions.numpy()
                # Обрезаем отрицательные значения
                predictions = np.maximum(predictions, 0.0)
        
        return predictions
    
    def compute_uncertainty_metrics(
        self,
        uncertainty_results: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Вычисляет метрики неопределённости
        
        Args:
            uncertainty_results: Результаты estimate_uncertainty
            
        Returns:
            dict: Метрики неопределённости
        """
        std = uncertainty_results['std']
        mean = uncertainty_results['mean']
        
        # Относительное стандартное отклонение (коэффициент вариации)
        cv = np.mean(std / (mean + 1e-10))  # Избегаем деления на ноль
        
        # Средняя ширина доверительного интервала (95%)
        percentiles = uncertainty_results['percentiles']
        ci_width = np.mean(percentiles[95] - percentiles[5])
        
        # Максимальная неопределённость
        max_uncertainty = np.max(std)
        
        # Средняя неопределённость
        mean_uncertainty = np.mean(std)
        
        return {
            'coefficient_of_variation': float(cv),
            'mean_ci_width': float(ci_width),
            'max_uncertainty': float(max_uncertainty),
            'mean_uncertainty': float(mean_uncertainty)
        }
