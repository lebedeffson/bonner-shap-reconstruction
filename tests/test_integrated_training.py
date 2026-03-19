"""
Интеграционные тесты для текущего SHAP fine-tune режима
"""

import unittest
import numpy as np
import sys
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.anfis_manager import ANFISManager
from src.models.shap_trainer_improved import ShapAwareANFISTrainerImproved
from src.utils.config_loader import load_config


class TestShapFineTune(unittest.TestCase):
    """Тесты текущего режима SHAP fine-tune"""
    
    @classmethod
    def setUpClass(cls):
        """Подготовка для всех тестов"""
        cls.config_path = "configs/config_integrated_shap.yaml"
        if not Path(cls.config_path).exists():
            cls.config_path = Path(__file__).parent.parent / "configs" / "config_integrated_shap.yaml"
        
        cls.config = deepcopy(load_config(str(cls.config_path)))
        
        # Уменьшаем параметры для быстрых тестов
        cls.config['model']['num_rules'] = 3
        cls.config['model']['optim_params']['epoch'] = 2
        cls.config['model']['optim_params']['pop_size'] = 5
        cls.config['model']['n_workers'] = 1
        
        # Настройки SHAP для тестов
        cls.config['shap_reg']['epochs'] = 5
        cls.config['shap_reg']['true_shap_update_frequency'] = 1
        cls.config['shap_reg']['use_gpu'] = False
        cls.config['shap_reg']['integrated_training'] = False
        cls.config['shap_reg']['use_pso_init'] = False
    
    def test_shap_trainer_creation(self):
        """Тест создания SHAP fine-tune тренера"""
        manager = ANFISManager(self.config)
        
        n_features = 10
        n_outputs = 60

        model = manager.create_model(
            verbose=False,
            input_dim=n_features,
            output_dim=n_outputs
        )
        
        trainer = ShapAwareANFISTrainerImproved(
            model,
            self.config,
            gamma=0.5,
            verbose=False
        )
        
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.gamma, 0.5)
    
    def test_shap_trainer_fit(self):
        """Тест обучения в текущем SHAP режиме"""
        manager = ANFISManager(self.config)
        
        n_samples = 30
        n_features = 10
        n_outputs = 60
        
        X_train = np.random.rand(n_samples, n_features)
        y_train = np.random.rand(n_samples, n_outputs)
        X_val = np.random.rand(10, n_features)
        y_val = np.random.rand(10, n_outputs)
        
        # Создаем модель
        model = manager.create_model(
            verbose=False,
            input_dim=n_features,
            output_dim=n_outputs
        )
        
        trainer = ShapAwareANFISTrainerImproved(
            model,
            self.config,
            gamma=0.5,
            verbose=False
        )
        
        history = trainer.fit(
            X_train,
            y_train,
            epochs=3,
            batch_size=10,
            lr=0.01
        )
        
        self.assertIsNotNone(history)
        self.assertIn('total_loss', history)
        self.assertIn('main_loss', history)
        self.assertIn('shap_loss', history)
        self.assertIn('tikhonov_loss', history)
        self.assertIn('adaptive_gamma', history)
        
        self.assertGreater(len(history['total_loss']), 0)
        self.assertTrue(all(np.isfinite(v) for v in history['total_loss']))
        self.assertEqual(len(history['shap_consistency']), 3)
        self.assertEqual(len(history['shap_sparsity']), 3)
    
    def test_shap_trainer_predictions(self):
        """Тест предсказаний после SHAP обучения"""
        manager = ANFISManager(self.config)
        
        n_samples = 30
        n_features = 10
        n_outputs = 60
        
        X_train = np.random.rand(n_samples, n_features)
        y_train = np.random.rand(n_samples, n_outputs)
        X_test = np.random.rand(10, n_features)
        
        # Создаем модель
        model = manager.create_model(
            verbose=False,
            input_dim=n_features,
            output_dim=n_outputs
        )
        
        trainer = ShapAwareANFISTrainerImproved(
            model,
            self.config,
            gamma=0.5,
            verbose=False
        )
        
        trainer.fit(
            X_train,
            y_train,
            epochs=3,
            batch_size=10,
            lr=0.01
        )
        
        predictions = trainer.predict(X_test)
        
        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape, (len(X_test), n_outputs))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_shap_trainer_global_importance(self):
        """Тест вычисления глобальной важности признаков"""
        manager = ANFISManager(self.config)
        
        n_samples = 30
        n_features = 10
        n_outputs = 60
        
        X_train = np.random.rand(n_samples, n_features)
        y_train = np.random.rand(n_samples, n_outputs)
        
        # Создаем модель
        model = manager.create_model(
            verbose=False,
            input_dim=n_features,
            output_dim=n_outputs
        )
        
        trainer = ShapAwareANFISTrainerImproved(
            model,
            self.config,
            gamma=0.5,
            verbose=False
        )
        
        trainer.fit(
            X_train,
            y_train,
            epochs=3,
            batch_size=10,
            lr=0.01
        )
        
        shap_importance = trainer.get_global_shap_importance(X_train[:5])
        
        self.assertIsNotNone(shap_importance)
        self.assertEqual(len(shap_importance), n_features)
        self.assertTrue(np.all(np.isfinite(shap_importance)))
        self.assertTrue(np.all(shap_importance >= 0))


if __name__ == '__main__':
    unittest.main()
