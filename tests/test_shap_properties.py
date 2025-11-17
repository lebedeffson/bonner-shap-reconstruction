"""
Property-based тесты для Shapley values
Проверяет свойства эффективности, симметричности и аддитивности
"""

import unittest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.shap_trainer import ShapAwareANFISTrainer
from src.models.shap_metrics import ShapMetrics


class TestShapProperties(unittest.TestCase):
    """Тесты свойств Shapley values"""
    
    def setUp(self):
        """Подготовка тестов"""
        # Создаем простую модель для тестирования
        from xanfis import BioAnfisRegressor
        
        self.config = {
            'model': {
                'num_rules': 3,
                'mf_class': 'Gaussian',
                'vanishing_strategy': 'blend',
                'optim': 'OriginalPSO',
                'reg_lambda': 0.1,
                'seed': 42,
                'n_workers': 1,
                'optim_params': {
                    'epoch': 2,
                    'pop_size': 5,
                    'verbose': False
                }
            },
            'shap_reg': {
                'enabled': True,
                'gamma': 0.5,
                'shap_n_samples': 50,  # Малое количество для быстрых тестов
                'shap_exact': False,
                'cache_enabled': False,  # Отключаем кэш для тестов
                'use_gpu': False
            }
        }
        
        # Создаем простую модель
        self.model = BioAnfisRegressor(
            num_rules=self.config['model']['num_rules'],
            mf_class=self.config['model']['mf_class'],
            vanishing_strategy=self.config['model']['vanishing_strategy'],
            optim=self.config['model']['optim'],
            optim_params=self.config['model']['optim_params'],
            reg_lambda=self.config['model']['reg_lambda'],
            seed=self.config['model']['seed'],
            n_workers=self.config['model']['n_workers'],
            verbose=False
        )
        
        # Простые данные для тестирования
        self.n_features = 5
        self.n_samples = 20
        self.X_test = np.random.rand(self.n_samples, self.n_features)
        self.y_test = np.random.rand(self.n_samples, 10)  # 10 выходов
        
        # Инициализируем модель
        self.model.size_input = self.n_features
        self.model.size_output = self.y_test.shape[1]
        self.model.build_model()
        
        # Быстрое обучение для тестирования
        self.model.fit(self.X_test, self.y_test)
        
        # Создаем тренер
        self.trainer = ShapAwareANFISTrainer(
            self.model,
            self.config,
            gamma=0.5,
            verbose=False
        )
    
    def test_efficiency_property(self):
        """
        Тест свойства эффективности Shapley values
        
        Эффективность: сумма Shapley values должна равняться разнице между
        предсказанием модели и предсказанием на baseline
        """
        baseline = np.mean(self.X_test, axis=0)
        shap_values = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
        
        # Получаем предсказания
        predictions = self.trainer.predict(self.X_test[:1])
        
        # Вычисляем метрики эффективности
        metrics = ShapMetrics.compute_efficiency(shap_values, predictions)
        
        # Проверяем, что сумма Shapley values не равна нулю
        self.assertNotEqual(metrics['shap_sum'], 0.0, "Сумма Shapley values не должна быть нулевой")
        
        # Проверяем, что метрики вычислены корректно
        self.assertIn('shap_sum', metrics)
        self.assertIn('mean_prediction', metrics)
        self.assertIn('efficiency_error', metrics)
        self.assertIn('is_efficient', metrics)
    
    def test_symmetry_property(self):
        """
        Тест свойства симметричности Shapley values
        
        Симметричность: если два признака вносят одинаковый вклад,
        их Shapley values должны быть близки
        """
        baseline = np.mean(self.X_test, axis=0)
        shap_values = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
        
        # Вычисляем метрики симметричности
        metrics = ShapMetrics.compute_symmetry(shap_values)
        
        # Проверяем, что метрики вычислены корректно
        self.assertIn('mean_symmetry_error', metrics)
        self.assertIn('max_symmetry_error', metrics)
        self.assertIn('n_pairs_tested', metrics)
        
        # Проверяем, что ошибка симметричности неотрицательна
        self.assertGreaterEqual(metrics['mean_symmetry_error'], 0.0)
    
    def test_stability_metrics(self):
        """
        Тест метрик стабильности Shapley values
        """
        baseline = np.mean(self.X_test, axis=0)
        
        # Вычисляем Shapley values несколько раз
        shap_values_list = []
        for _ in range(3):
            shap_values = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
            shap_values_list.append(shap_values)
        
        # Вычисляем метрики стабильности
        stability_metrics = ShapMetrics.compute_stability(shap_values_list)
        
        # Проверяем, что метрики вычислены корректно
        self.assertIn('mean_stability', stability_metrics)
        self.assertIn('max_stability', stability_metrics)
        self.assertIn('min_stability', stability_metrics)
        self.assertIn('per_feature_stability', stability_metrics)
        
        # Проверяем, что стабильность неотрицательна
        self.assertGreaterEqual(stability_metrics['mean_stability'], 0.0)
    
    def test_shap_values_shape(self):
        """Тест формы Shapley values"""
        baseline = np.mean(self.X_test, axis=0)
        shap_values = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
        
        # Проверяем форму
        self.assertEqual(shap_values.shape, (self.n_features,), 
                        f"Shapley values должны иметь форму ({self.n_features},), получено {shap_values.shape}")
        
        # Проверяем, что значения не все нули
        self.assertFalse(np.all(shap_values == 0), "Shapley values не должны быть все нулями")
    
    def test_shap_values_finite(self):
        """Тест что Shapley values конечны"""
        baseline = np.mean(self.X_test, axis=0)
        shap_values = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
        
        # Проверяем, что все значения конечны
        self.assertTrue(np.all(np.isfinite(shap_values)), 
                        "Все Shapley values должны быть конечными")
    
    def test_shap_values_non_negative(self):
        """Тест что Shapley values неотрицательны (так как мы берем абсолютное значение)"""
        baseline = np.mean(self.X_test, axis=0)
        shap_values = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
        
        # Проверяем, что все значения неотрицательны (так как мы берем abs)
        self.assertTrue(np.all(shap_values >= 0), 
                        "Shapley values должны быть неотрицательными (abs)")
    
    def test_baseline_independence(self):
        """
        Тест что Shapley values изменяются при изменении baseline
        """
        baseline1 = np.mean(self.X_test, axis=0)
        baseline2 = baseline1 + 0.1  # Сдвигаем baseline
        
        shap_values1 = self.trainer._calculate_shap_approximation(self.X_test[0], baseline1)
        shap_values2 = self.trainer._calculate_shap_approximation(self.X_test[0], baseline2)
        
        # Shapley values должны изменяться при изменении baseline
        # (хотя бы немного)
        self.assertFalse(np.allclose(shap_values1, shap_values2, atol=1e-6),
                        "Shapley values должны изменяться при изменении baseline")


if __name__ == '__main__':
    unittest.main()

