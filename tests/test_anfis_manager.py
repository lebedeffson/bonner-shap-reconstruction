"""
Тесты для модуля anfis_manager
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.anfis_manager import ANFISManager


class TestANFISManager(unittest.TestCase):
    """Тесты для ANFISManager"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        # Минимальная конфигурация для быстрых тестов
        self.config = {
            'model': {
                'num_rules': 3,  # Мало правил для быстрого теста
                'mf_class': 'Gaussian',
                'optim': 'OriginalPSO',
                'optim_params': {
                    'epoch': 2,  # Очень мало эпох для теста
                    'pop_size': 5,  # Маленькая популяция
                    'verbose': False
                },
                'reg_lambda': 0.1,
                'seed': 42,
                'n_workers': 1
            }
        }
        
        # Создаем небольшие тестовые данные
        np.random.seed(42)
        n_samples = 50
        
        self.X_train = np.random.rand(n_samples, 10)
        self.y_train = np.random.rand(n_samples, 60)
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.rand(20, 60)
        
        self.manager = ANFISManager(self.config)
    
    def test_create_model(self):
        """Тест создания модели"""
        model = self.manager.create_model(verbose=False)
        self.assertIsNotNone(model)
        self.assertEqual(self.manager.task_type, 'regression')
    
    def test_calculate_metrics(self):
        """Тест вычисления метрик"""
        # Тестовые предсказания
        y_true = np.random.rand(10, 60)
        y_pred = y_true + np.random.rand(10, 60) * 0.1  # Небольшой шум
        
        metrics = self.manager._calculate_metrics(y_true, y_pred)
        
        # Проверяем наличие всех метрик
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # Проверяем, что метрики положительные
        self.assertGreater(metrics['mse'], 0)
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
    
    def test_extract_feature_importance(self):
        """Тест извлечения важности признаков"""
        # Создаем модель и обучаем на маленьких данных
        model = self.manager.create_model(verbose=False)
        
        # Обучаем на очень маленькой выборке
        try:

            model.fit(self.X_train[:10], self.y_train[:10])

        except Exception as e:

            raise Exception(f"Ошибка при обучении модели: {e}")
        
        importance = self.manager._extract_feature_importance(model, 10)
        
        # Проверяем форму
        self.assertEqual(len(importance), 10)
        
        # Проверяем, что важность неотрицательная
        self.assertTrue(np.all(importance >= 0))


if __name__ == '__main__':
    unittest.main()

