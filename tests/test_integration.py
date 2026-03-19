"""
Интеграционные тесты для проверки всего pipeline
"""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.utils.data_loader import load_data, prepare_features_targets, split_data
from src.models.anfis_manager import ANFISManager


class TestIntegration(unittest.TestCase):
    """Интеграционные тесты"""
    
    @classmethod
    def setUpClass(cls):
        """Подготовка для всех тестов"""
        cls.config_path = "configs/config_integrated_shap.yaml"
        if not os.path.exists(cls.config_path):
            cls.config_path = os.path.join(Path(__file__).parent.parent, "configs", "config_integrated_shap.yaml")
        
        cls.config = load_config(cls.config_path)
        
        # Уменьшаем параметры для быстрых тестов
        cls.config['model']['num_rules'] = 3
        cls.config['model']['optim_params']['epoch'] = 2
        cls.config['model']['optim_params']['pop_size'] = 5
        cls.config['model']['n_workers'] = 1
    
    def test_full_pipeline(self):
        """Тест полного pipeline: загрузка -> подготовка -> обучение"""
        # Проверяем наличие данных
        train_data_path = self.config['dataset']['train_data']
        if not os.path.exists(train_data_path):
            self.skipTest(f"Данные не найдены: {train_data_path}")
        
        # Загрузка данных
        data = load_data(train_data_path)
        self.assertGreater(len(data), 0)
        
        # Подготовка
        X, y, SUM = prepare_features_targets(data, normalize_sum=False)
        self.assertEqual(X.shape[1], 10)
        self.assertEqual(y.shape[1], 60)
        
        # Разделение (на маленькой выборке)
        sample_size = min(100, len(X))
        X_sample = X.iloc[:sample_size] if hasattr(X, 'iloc') else X[:sample_size]
        y_sample = y.iloc[:sample_size] if hasattr(y, 'iloc') else y[:sample_size]
        
        X_train, X_test, y_train, y_test = split_data(
            X_sample, y_sample, test_size=0.25, random_state=42
        )
        
        # Обучение
        manager = ANFISManager(self.config)
        results = manager.train_vanilla_model(X_train, y_train, X_test, y_test)
        
        # Проверяем результаты
        self.assertIn('model', results)
        self.assertIn('metrics', results)
        self.assertIn('feature_importance', results)
        
        # Проверяем метрики
        metrics = results['metrics']
        self.assertIn('mse', metrics)
        self.assertIn('r2', metrics)
        self.assertGreater(metrics['r2'], -1)  # R² может быть отрицательным, но не слишком
    
    def test_config_loading(self):
        """Тест загрузки конфигурации"""
        config = load_config(self.config_path)
        
        # Проверяем наличие необходимых ключей
        self.assertIn('dataset', config)
        self.assertIn('model', config)
        self.assertIn('output', config)
        
        # Проверяем параметры модели
        self.assertIn('num_rules', config['model'])
        self.assertIn('reg_lambda', config['model'])
        self.assertIn('optim_params', config['model'])


if __name__ == '__main__':
    unittest.main()
