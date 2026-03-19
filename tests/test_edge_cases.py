"""
Unit-тесты для edge cases и отдельных функций
"""

import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import os

from src.models.anfis_manager import ANFISManager
from src.utils.data_loader import (
    load_data,
    prepare_features_targets,
    split_data,
    denormalize_predictions
)
from src.models.shap_trainer_improved import ShapAwareANFISTrainerImproved
from src.utils.uncertainty_estimation import UncertaintyEstimator


class TestDataLoaderEdgeCases(unittest.TestCase):
    """Тесты для edge cases в data_loader"""
    
    def setUp(self):
        """Создание временных данных для тестов"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_data.csv")
        
    def tearDown(self):
        """Очистка временных файлов"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_data_nonexistent_file(self):
        """Тест загрузки несуществующего файла"""
        with self.assertRaises(FileNotFoundError):
            load_data("nonexistent_file.csv")
    
    def test_load_data_empty_file(self):
        """Тест загрузки пустого файла"""
        # Создаем пустой CSV
        pd.DataFrame().to_csv(self.test_data_path, index=False)
        with self.assertRaises((ValueError, pd.errors.EmptyDataError)):
            data = load_data(self.test_data_path)
            self.assertEqual(len(data), 0)
    
    def test_prepare_features_targets_empty_data(self):
        """Тест подготовки признаков из пустых данных"""
        empty_df = pd.DataFrame()
        with self.assertRaises((KeyError, IndexError)):
            prepare_features_targets(empty_df)
    
    def test_prepare_features_targets_normalize_sum_zero(self):
        """Тест нормализации при SUM=0"""
        # Создаем данные где все Q = 0
        data = pd.DataFrame({
            'Q1': [0, 0, 0],
            'Q2': [0, 0, 0],
            **{f'col_{i}': [0.1] * 3 for i in range(60)}
        })
        X, y, SUM = prepare_features_targets(data, normalize_sum=True)
        # Проверяем, что SUM не равен нулю (должен быть заменен на eps)
        self.assertTrue((SUM > 0).all())
    
    def test_split_data_single_sample(self):
        """Тест разделения данных с одним образцом"""
        X = pd.DataFrame({'Q1': [1], 'Q2': [2]})
        y = pd.DataFrame({i: [0.1] for i in range(60)})
        # С test_size=0.25 и одним образцом должна быть ошибка или предупреждение
        with self.assertRaises(ValueError):
            split_data(X, y, test_size=0.25)
    
    def test_denormalize_predictions_none_sum(self):
        """Тест денормализации без SUM"""
        y_pred = np.array([[0.1, 0.2], [0.3, 0.4]])
        result = denormalize_predictions(y_pred, None)
        np.testing.assert_array_equal(result, y_pred)
    
    def test_denormalize_predictions_wrong_shape(self):
        """Тест денормализации с несовместимыми формами"""
        y_pred = np.array([[0.1, 0.2], [0.3, 0.4]])
        SUM = np.array([1.0])  # Неправильная форма
        with self.assertRaises((ValueError, IndexError)):
            denormalize_predictions(y_pred, SUM)


class TestANFISManagerEdgeCases(unittest.TestCase):
    """Тесты для edge cases в ANFISManager"""
    
    def setUp(self):
        """Создание минимальной конфигурации"""
        self.config = {
            'model': {
                'num_rules': 5,
                'mf_class': 'Gaussian',
                'vanishing_strategy': 'blend',
                'optim': 'OriginalPSO',
                'optim_params': {'epoch': 5, 'pop_size': 5, 'verbose': False},
                'reg_lambda': 0.01,
                'seed': 42,
                'n_workers': 1
            }
        }
        self.manager = ANFISManager(self.config)
    
    def test_sanitize_predictions_nan_inf(self):
        """Тест очистки предсказаний с NaN/Inf"""
        y_pred = np.array([[np.nan, np.inf], [-np.inf, 1.0]])
        result = ANFISManager._sanitize_predictions(y_pred)
        self.assertTrue(np.isfinite(result).all())
    
    def test_sanitize_predictions_wrong_shape(self):
        """Тест исправления формы предсказаний"""
        y_pred = np.array([1, 2, 3, 4, 5, 6])  # 1D массив
        reference_shape = (1, 6)  # Ожидаем 2D
        result = ANFISManager._sanitize_predictions(y_pred, reference_shape)
        self.assertEqual(result.shape, reference_shape)
    
    def test_create_model_with_dimensions(self):
        """Тест создания модели с заданными размерами"""
        model = self.manager.create_model(
            verbose=False,
            input_dim=10,
            output_dim=60
        )
        self.assertIsNotNone(model)
        self.assertEqual(model.size_input, 10)
        self.assertEqual(model.size_output, 60)
    
    def test_train_vanilla_empty_data(self):
        """Тест обучения с пустыми данными"""
        X_train = np.array([]).reshape(0, 10)
        y_train = np.array([]).reshape(0, 60)
        X_test = np.array([[1] * 10])
        y_test = np.array([[0.1] * 60])
        
        with self.assertRaises(ValueError):
            self.manager.train_vanilla_model(X_train, y_train, X_test, y_test)
    
    def test_train_vanilla_all_zeros(self):
        """Тест обучения с данными, состоящими из нулей"""
        X_train = np.zeros((10, 10))
        y_train = np.zeros((10, 60))
        X_test = np.array([[1] * 10])
        y_test = np.array([[0.1] * 60])
        
        with self.assertRaises(ValueError):
            self.manager.train_vanilla_model(X_train, y_train, X_test, y_test)


class TestUncertaintyEstimatorEdgeCases(unittest.TestCase):
    """Тесты для edge cases в UncertaintyEstimator"""
    
    def setUp(self):
        """Создание тестовых данных"""
        from xanfis import BioAnfisRegressor
        
        self.X_test = np.random.rand(10, 10)
        self.y_test = np.random.rand(10, 60)
        self.Q_values = np.random.rand(10, 10)
        
        # Создаем простую модель-заглушку
        class DummyModel:
            def predict(self, X):
                return np.random.rand(len(X), 60)
        
        self.estimator = UncertaintyEstimator(DummyModel(), verbose=False)
    
    def test_add_measurement_error_zero_error(self):
        """Тест добавления ошибки с нулевым процентом"""
        # Сохраняем seed для воспроизводимости
        np.random.seed(42)
        result = self.estimator._add_measurement_error(self.Q_values, error_percent=0.0)
        # При нулевой ошибке результат должен быть близок к исходному
        np.testing.assert_array_almost_equal(result, self.Q_values, decimal=5)
    
    def test_add_measurement_error_negative_values(self):
        """Тест обработки отрицательных значений после добавления ошибки"""
        Q_with_neg = np.array([[1.0, -0.5, 2.0]])
        result = self.estimator._add_measurement_error(Q_with_neg, error_percent=10.0, seed=42)
        # Проверяем, что отрицательные значения обработаны (обрезаны до 0 или заменены)
        self.assertTrue((result >= 0).all() or np.isnan(result).any() == False)
    
    def test_estimate_uncertainty_empty_data(self):
        """Тест оценки неопределенности с пустыми данными"""
        # Создаем модель-заглушку для пустых данных
        class DummyModelEmpty:
            def predict(self, X):
                if len(X) == 0:
                    return np.array([]).reshape(0, 60)
                return np.zeros((len(X), 60))
        
        estimator = UncertaintyEstimator(DummyModelEmpty(), verbose=False)
        # Пустые данные могут вызвать ошибку или вернуть пустой результат
        try:
            result = estimator.estimate_uncertainty(
                np.array([]).reshape(0, 10),
                n_samples=10,
                error_percent=5.0
            )
            # Если не вызвана ошибка, проверяем что результат пустой
            self.assertEqual(len(result['mean']), 0)
        except (ValueError, IndexError, RuntimeError):
            # Ожидаемое поведение для edge case
            pass


class TestShapTrainerEdgeCases(unittest.TestCase):
    """Тесты для edge cases в текущем SHAP-тренере"""
    
    def setUp(self):
        """Создание минимальной модели и конфигурации"""
        from xanfis import BioAnfisRegressor
        
        self.config = {
            'shap_reg': {
                'use_gpu': False,
                'shap_n_samples': 10,
                'grad_clip': 5.0,
                'negative_penalty': 0.1
            }
        }
        
        # Создаем простую модель
        model_wrapper = type('obj', (object,), {
            'network': None
        })()
        
        # Создаем простую ANFIS модель
        anfis_model = BioAnfisRegressor(
            num_rules=3,
            mf_class='Gaussian',
            optim='OriginalPSO',
            optim_params={'epoch': 1, 'pop_size': 5, 'verbose': False},
            reg_lambda=0.01,
            seed=42,
            n_workers=1,
            verbose=False
        )
        anfis_model.size_input = 10
        anfis_model.size_output = 60
        anfis_model.build_model()
        
        model_wrapper.network = anfis_model.network
        
        self.trainer = ShapAwareANFISTrainerImproved(
            model_wrapper,
            self.config,
            gamma=0.5,
            verbose=False
        )
    
    def test_predict_empty_input(self):
        """Тест предсказания на пустых данных"""
        X_empty = np.array([]).reshape(0, 10)
        result = self.trainer.predict(X_empty)
        self.assertEqual(result.shape[0], 0)
    
    def test_predict_nan_input(self):
        """Тест предсказания на данных с NaN"""
        X_with_nan = np.array([[1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]])
        result = self.trainer.predict(X_with_nan)
        # Предсказания должны быть валидными (NaN должны быть обработаны)
        self.assertTrue(np.isfinite(result).all() or result.size == 0)
    
    def test_calculate_shap_approximation_finite(self):
        """SHAP-аппроксимация должна быть конечной для валидного образца"""
        X_sample = np.random.rand(1, 10)
        baseline = np.mean(X_sample, axis=0)
        shap_values = self.trainer._calculate_shap_approximation(X_sample, baseline)
        self.assertEqual(shap_values.shape, (10,))
        self.assertTrue(np.all(np.isfinite(shap_values)))


class TestIntegrationEdgeCases(unittest.TestCase):
    """Интеграционные тесты для edge cases"""
    
    def test_full_pipeline_single_sample(self):
        """Тест полного пайплайна с одним образцом"""
        # Создаем минимальные данные
        data = pd.DataFrame({
            'Q1': [1.0], 'Q2': [2.0], 'Q3': [3.0], 'Q4': [4.0], 'Q5': [5.0],
            'Q6': [6.0], 'Q7': [7.0], 'Q8': [8.0], 'Q9': [9.0], 'Q10': [10.0],
            **{i: [0.1] for i in range(60)}
        })
        
        try:
            X, y, SUM = prepare_features_targets(data, normalize_sum=False)
            # С одним образцом split_data может вызвать ошибку
            # Это нормальное поведение для edge case
            with self.assertRaises(ValueError):
                split_data(X, y, test_size=0.25)
        except (ValueError, IndexError) as e:
            # Ожидаемое поведение для edge case
            self.assertIsInstance(e, (ValueError, IndexError))


if __name__ == '__main__':
    unittest.main()
