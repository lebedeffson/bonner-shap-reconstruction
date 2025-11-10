"""
Тесты валидации и проверки качества кода
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import denormalize_predictions


class TestValidation(unittest.TestCase):
    """Тесты валидации данных и результатов"""
    
    def test_denormalize_predictions_shape(self):
        """Тест формы денормализованных предсказаний"""
        # Тест 1D массива
        y_pred_1d = np.array([0.1, 0.2, 0.3])
        SUM_1d = np.array([10.0])
        result_1d = denormalize_predictions(y_pred_1d, SUM_1d)
        self.assertEqual(result_1d.shape, (3,))
        
        # Тест 2D массива
        y_pred_2d = np.array([[0.1, 0.2], [0.3, 0.4]])
        SUM_2d = np.array([10.0, 20.0])
        result_2d = denormalize_predictions(y_pred_2d, SUM_2d)
        self.assertEqual(result_2d.shape, (2, 2))
    
    def test_denormalize_predictions_values(self):
        """Тест правильности денормализации"""
        y_pred = np.array([[0.1, 0.2], [0.3, 0.4]])
        SUM = np.array([10.0, 20.0])
        
        result = denormalize_predictions(y_pred, SUM)
        
        # Проверяем значения
        expected = np.array([[1.0, 2.0], [6.0, 8.0]])
        np.testing.assert_allclose(result, expected)
    
    def test_negative_values_handling(self):
        """Тест обработки отрицательных значений"""
        # Предсказания могут быть отрицательными
        y_pred = np.array([[-0.1, 0.2], [0.3, -0.4]])
        SUM = np.array([10.0, 20.0])
        
        result = denormalize_predictions(y_pred, SUM)
        
        # Должны корректно обрабатываться
        self.assertEqual(result.shape, (2, 2))
    
    def test_edge_cases(self):
        """Тест граничных случаев"""
        # Нулевые значения
        y_pred_zero = np.array([[0.0, 0.0], [0.0, 0.0]])
        SUM_zero = np.array([1.0, 1.0])
        result_zero = denormalize_predictions(y_pred_zero, SUM_zero)
        np.testing.assert_allclose(result_zero, 0.0)
        
        # Очень большие значения
        y_pred_large = np.array([[1e6, 1e6]])
        SUM_large = np.array([1e-6])
        result_large = denormalize_predictions(y_pred_large, SUM_large)
        self.assertEqual(result_large.shape, (1, 2))


if __name__ == '__main__':
    unittest.main()

