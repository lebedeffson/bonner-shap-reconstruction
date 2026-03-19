"""
Тесты свойств текущей SHAP-аппроксимации
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.shap_trainer_improved import ShapAwareANFISTrainerImproved


class TestShapProperties(unittest.TestCase):
    """Тесты свойств SHAP-аппроксимации"""
    
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
                'true_shap_update_frequency': 1,
                'use_gpu': False,
                'use_improved_shap': True
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
        self.trainer = ShapAwareANFISTrainerImproved(
            self.model,
            self.config,
            gamma=0.5,
            verbose=False
        )
    
    def test_shap_values_reproducible(self):
        """Повторные вычисления на одних и тех же данных должны совпадать"""
        baseline = np.mean(self.X_test, axis=0)
        shap_values_1 = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
        shap_values_2 = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
        np.testing.assert_allclose(shap_values_1, shap_values_2, atol=1e-8)
    
    def test_global_shap_importance_shape(self):
        """Глобальная важность должна совпадать с числом признаков"""
        shap_importance = self.trainer.get_global_shap_importance(self.X_test[:5])
        self.assertEqual(shap_importance.shape, (self.n_features,))
        self.assertTrue(np.all(np.isfinite(shap_importance)))
        self.assertTrue(np.all(shap_importance >= 0))
        self.assertAlmostEqual(float(np.sum(shap_importance)), 1.0, places=6)

    def test_baseline_independence(self):
        """SHAP-аппроксимация должна зависеть от baseline"""
        baseline1 = np.mean(self.X_test, axis=0)
        baseline2 = baseline1 + 0.1
        
        shap_values1 = self.trainer._calculate_shap_approximation(self.X_test[0], baseline1)
        shap_values2 = self.trainer._calculate_shap_approximation(self.X_test[0], baseline2)
        
        self.assertFalse(np.allclose(shap_values1, shap_values2, atol=1e-6))

    def test_single_sample_shap_is_nontrivial(self):
        """Важности не должны схлопываться в нулевой вектор"""
        baseline = np.mean(self.X_test, axis=0)
        shap_values = self.trainer._calculate_shap_approximation(self.X_test[0], baseline)
        self.assertGreater(float(np.sum(shap_values)), 0.0)
    
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
    
if __name__ == '__main__':
    unittest.main()
