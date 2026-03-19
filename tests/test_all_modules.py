"""
Комплексные тесты всех модулей проекта
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModuleImports(unittest.TestCase):
    """Тесты импорта всех модулей"""
    
    def test_shap_trainer_improved_import(self):
        """Тест импорта улучшенного SHAP-тренера"""
        try:
            from src.models.shap_trainer_improved import ShapAwareANFISTrainerImproved
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать ShapAwareANFISTrainerImproved: {e}")
    
    def test_shap_precision_utils_import(self):
        """Тест импорта precision-aware SHAP утилит"""
        try:
            from src.models.shap_trainer_precision_optimized import PrecisionOptimizedSHAPRegularization
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать PrecisionOptimizedSHAPRegularization: {e}")
    
    def test_uncertainty_estimation_import(self):
        """Тест импорта UncertaintyEstimator"""
        try:
            from src.utils.uncertainty_estimation import UncertaintyEstimator
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать UncertaintyEstimator: {e}")
    
    def test_shap_plots_import(self):
        """Тест импорта функций визуализации"""
        try:
            from src.visualization.shap_plots import (
                plot_shap_importance_interactive,
                plot_shap_waterfall_interactive,
                plot_shap_metrics_interactive
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать функции визуализации: {e}")
    
    def test_anfis_manager_import(self):
        """Тест импорта ANFISManager"""
        try:
            from src.models.anfis_manager import ANFISManager
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать ANFISManager: {e}")
    
    def test_config_loader_import(self):
        """Тест импорта load_config"""
        try:
            from src.utils.config_loader import load_config
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать load_config: {e}")
    
    def test_data_loader_import(self):
        """Тест импорта функций загрузки данных"""
        try:
            from src.utils.data_loader import (
                load_training_dataset,
                prepare_features_targets,
                split_data
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Не удалось импортировать функции загрузки данных: {e}")


class TestModuleBasicFunctionality(unittest.TestCase):
    """Базовые тесты функциональности модулей"""
    
    def test_precision_shap_compute_stability(self):
        """Тест вычисления стабильности precision-aware SHAP"""
        import torch
        from src.models.shap_trainer_precision_optimized import PrecisionOptimizedSHAPRegularization

        importance = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.11, 0.19, 0.31],
            [0.09, 0.21, 0.29]
        ], dtype=torch.float32)

        metrics = PrecisionOptimizedSHAPRegularization.compute_precision_aware_stability(
            importance,
            current_main_loss=0.01
        )

        self.assertIn('stability_loss', metrics)
        self.assertGreaterEqual(float(metrics['stability_loss']), 0.0)

    def test_precision_shap_js_divergence_is_symmetric_and_zero_on_match(self):
        """JS-дивергенция должна быть симметричной и нулевой для одинаковых распределений."""
        import torch
        from src.models.shap_trainer_precision_optimized import PrecisionOptimizedSHAPRegularization

        p = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
        q = torch.tensor([0.4, 0.1, 0.5], dtype=torch.float32)

        js_pp = PrecisionOptimizedSHAPRegularization._js_divergence(p, p)
        js_pq = PrecisionOptimizedSHAPRegularization._js_divergence(p, q)
        js_qp = PrecisionOptimizedSHAPRegularization._js_divergence(q, p)

        self.assertAlmostEqual(float(js_pp.item()), 0.0, places=7)
        self.assertGreaterEqual(float(js_pq.item()), 0.0)
        self.assertAlmostEqual(float(js_pq.item()), float(js_qp.item()), places=7)
    
    def test_uncertainty_estimator_init(self):
        """Тест инициализации UncertaintyEstimator"""
        import torch
        from src.utils.uncertainty_estimation import UncertaintyEstimator
        
        # Создаем простую модель-заглушку
        class DummyModel:
            def predict(self, X):
                return np.random.rand(len(X), 10)
        
        model = DummyModel()
        estimator = UncertaintyEstimator(model, device=torch.device('cpu'), verbose=False)
        
        self.assertIsNotNone(estimator)
        self.assertEqual(estimator.model, model)


if __name__ == '__main__':
    unittest.main()
