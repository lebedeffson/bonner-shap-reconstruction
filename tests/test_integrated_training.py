"""
Интеграционные тесты для интегрированного режима обучения
"""

import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.shap_integrated_trainer import ShapIntegratedANFISTrainer
from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config


class TestIntegratedTraining(unittest.TestCase):
    """Тесты интегрированного режима обучения"""
    
    @classmethod
    def setUpClass(cls):
        """Подготовка для всех тестов"""
        cls.config_path = "configs/config.yaml"
        if not Path(cls.config_path).exists():
            cls.config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        
        cls.config = load_config(str(cls.config_path))
        
        # Уменьшаем параметры для быстрых тестов
        cls.config['model']['num_rules'] = 3
        cls.config['model']['optim_params']['epoch'] = 2
        cls.config['model']['optim_params']['pop_size'] = 5
        cls.config['model']['n_workers'] = 1
        
        # Настройки SHAP для тестов
        cls.config['shap_reg']['epochs'] = 5
        cls.config['shap_reg']['shap_n_samples'] = 20
        cls.config['shap_reg']['use_gpu'] = False
        cls.config['shap_reg']['cache_enabled'] = False
        cls.config['shap_reg']['integrated_training'] = True
        cls.config['shap_reg']['use_pso_init'] = False  # Отключаем PSO для быстрых тестов
    
    def test_integrated_training_creation(self):
        """Тест создания интегрированного тренера"""
        manager = ANFISManager(self.config)
        
        # Создаем простые данные
        n_samples = 50
        n_features = 10
        n_outputs = 60
        
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples, n_outputs)
        
        # Создаем модель
        model = manager.create_model(
            verbose=False,
            input_dim=n_features,
            output_dim=n_outputs
        )
        
        # Создаем интегрированный тренер
        trainer = ShapIntegratedANFISTrainer(
            model,
            self.config,
            gamma=0.5,
            verbose=False
        )
        
        # Проверяем, что тренер создан
        self.assertIsNotNone(trainer)
        self.assertIsNotNone(trainer.model)
        self.assertEqual(trainer.gamma, 0.5)
    
    def test_integrated_training_fit(self):
        """Тест обучения в интегрированном режиме"""
        manager = ANFISManager(self.config)
        
        # Создаем простые данные
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
        
        # Создаем интегрированный тренер
        trainer = ShapIntegratedANFISTrainer(
            model,
            self.config,
            gamma=0.5,
            verbose=False
        )
        
        # Обучаем модель
        history = trainer.fit_from_scratch(
            X_train,
            y_train,
            epochs=3,
            batch_size=10,
            lr=0.01,
            X_val=X_val,
            y_val=y_val
        )
        
        # Проверяем результаты
        self.assertIsNotNone(history)
        self.assertIn('total_loss', history)
        self.assertIn('main_loss', history)
        self.assertIn('shap_loss', history)
        self.assertIn('val_loss', history)
        
        # Проверяем, что потери уменьшаются (или хотя бы вычисляются)
        self.assertGreater(len(history['total_loss']), 0)
        self.assertTrue(all(np.isfinite(v) for v in history['total_loss']))
    
    def test_integrated_training_predictions(self):
        """Тест предсказаний после интегрированного обучения"""
        manager = ANFISManager(self.config)
        
        # Создаем простые данные
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
        
        # Создаем интегрированный тренер
        trainer = ShapIntegratedANFISTrainer(
            model,
            self.config,
            gamma=0.5,
            verbose=False
        )
        
        # Обучаем модель
        trainer.fit_from_scratch(
            X_train,
            y_train,
            epochs=3,
            batch_size=10,
            lr=0.01
        )
        
        # Получаем предсказания
        predictions = trainer.predict(X_test)
        
        # Проверяем предсказания
        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape, (len(X_test), n_outputs))
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertTrue(np.all(predictions >= 0))  # Спектры не могут быть отрицательными
    
    def test_integrated_training_shap_importance(self):
        """Тест вычисления важности признаков в интегрированном режиме"""
        manager = ANFISManager(self.config)
        
        # Создаем простые данные
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
        
        # Создаем интегрированный тренер
        trainer = ShapIntegratedANFISTrainer(
            model,
            self.config,
            gamma=0.5,
            verbose=False
        )
        
        # Обучаем модель
        trainer.fit_from_scratch(
            X_train,
            y_train,
            epochs=3,
            batch_size=10,
            lr=0.01
        )
        
        # Вычисляем важность признаков
        shap_importance = trainer.get_global_shap_importance(X_train[:5])
        
        # Проверяем важность признаков
        self.assertIsNotNone(shap_importance)
        self.assertEqual(len(shap_importance), n_features)
        self.assertTrue(np.all(np.isfinite(shap_importance)))
        self.assertTrue(np.all(shap_importance >= 0))


if __name__ == '__main__':
    unittest.main()

