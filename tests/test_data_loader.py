"""
Тесты для модуля data_loader
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import (
    load_data,
    prepare_features_targets,
    split_data,
    denormalize_predictions,
    load_validation_data
)


class TestDataLoader(unittest.TestCase):
    """Тесты для загрузки и обработки данных"""
    
    def setUp(self):
        """Подготовка тестовых данных"""
        # Создаем тестовый DataFrame
        np.random.seed(42)
        n_samples = 100
        
        # Создаем данные: 60 бинов + 10 Q
        data = {}
        for i in range(60):
            data[i] = np.random.rand(n_samples)
        
        for i in range(1, 11):
            data[f'Q{i}'] = np.random.rand(n_samples) * 10
        
        self.test_data = pd.DataFrame(data)
        self.test_file = 'test_data.csv'
        self.test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Очистка после тестов"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_load_data(self):
        """Тест загрузки данных"""
        data = load_data(self.test_file)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertEqual(len(data.columns), 70)  # 60 бинов + 10 Q
    
    def test_prepare_features_targets(self):
        """Тест подготовки признаков и целевых переменных"""
        X, y, SUM = prepare_features_targets(self.test_data, normalize_sum=False)
        
        # Проверяем признаки
        self.assertEqual(X.shape[1], 10)  # Q1-Q10
        self.assertTrue(all(f'Q{i}' in X.columns for i in range(1, 11)))
        
        # Проверяем целевые переменные
        self.assertEqual(y.shape[1], 60)  # 60 бинов
        
        # Проверяем SUM
        self.assertIsNone(SUM)
    
    def test_prepare_features_targets_with_normalization(self):
        """Тест нормализации на SUM"""
        X, y, SUM = prepare_features_targets(self.test_data, normalize_sum=True)
        
        # Проверяем, что SUM вычислен
        self.assertIsNotNone(SUM)
        self.assertEqual(len(SUM), len(self.test_data))
        
        # Проверяем, что нормализация применена
        # Сумма нормализованных Q должна быть близка к 1
        X_sum = X.sum(axis=1)
        np.testing.assert_allclose(X_sum, 1.0, rtol=1e-10)
    
    def test_split_data(self):
        """Тест разделения данных"""
        X = self.test_data.filter(regex='Q', axis=1)
        y = self.test_data.iloc[:, 0:60]
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)
        
        # Проверяем размеры
        self.assertEqual(len(X_train), 75)  # 75% от 100
        self.assertEqual(len(X_test), 25)   # 25% от 100
        self.assertEqual(len(y_train), 75)
        self.assertEqual(len(y_test), 25)
    
    def test_denormalize_predictions(self):
        """Тест денормализации предсказаний"""
        # Тестовые данные
        y_pred_normalized = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]])
        SUM = np.array([10.0, 20.0])
        
        y_pred = denormalize_predictions(y_pred_normalized, SUM)
        
        # Проверяем форму
        self.assertEqual(y_pred.shape, (2, 3))
        
        # Проверяем значения
        np.testing.assert_allclose(y_pred[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(y_pred[1], [4.0, 6.0, 8.0])


if __name__ == '__main__':
    unittest.main()

