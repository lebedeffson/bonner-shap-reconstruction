"""
Тесты для основной тихоновской версии SHAP + ANFIS пайплайна.
"""

import unittest
import numpy as np
import torch
import sys
import json
import tempfile
import csv
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.anfis_manager import ANFISManager
from src.models.shap_trainer_improved import ShapAwareANFISTrainerImproved
from src.utils.config_loader import load_config
from run_ablation_study import _compute_shap_distribution_metrics, _variant_definitions
from train import train_and_save, _split_real_data_for_shap, _prepare_feature_importance


class TestTikhonovVersion(unittest.TestCase):
    """Тесты тихоновской версии как основной постановки."""

    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).parent.parent / "configs" / "config_integrated_shap.yaml"
        v2_config_path = Path(__file__).parent.parent / "configs" / "config_integrated_shap_v2.yaml"
        cls.official_config = deepcopy(load_config(str(config_path)))
        cls.official_v2_config = deepcopy(load_config(str(v2_config_path)))
        cls.base_config = deepcopy(cls.official_config)

        # Облегчаем модель для быстрых тестов
        cls.base_config['model']['num_rules'] = 3
        cls.base_config['model']['optim_params']['epoch'] = 2
        cls.base_config['model']['optim_params']['pop_size'] = 5
        cls.base_config['model']['n_workers'] = 1

        cls.base_config['shap_reg']['use_gpu'] = False
        cls.base_config['shap_reg']['use_true_shap'] = False
        cls.base_config['shap_reg']['use_adaptive_gamma'] = False
        cls.base_config['shap_reg']['gamma'] = 0.0
        cls.base_config['shap_reg']['epochs'] = 2
        cls.base_config['shap_reg']['batch_size'] = 8
        cls.base_config['shap_reg']['lr'] = 0.001

    def _make_trainer(self, config=None):
        cfg = deepcopy(config or self.base_config)
        manager = ANFISManager(cfg)
        model = manager.create_model(
            verbose=False,
            input_dim=10,
            output_dim=60
        )
        return ShapAwareANFISTrainerImproved(
            model,
            cfg,
            gamma=cfg['shap_reg'].get('gamma', 0.0),
            verbose=False
        )

    def test_tikhonov_config_enabled_in_main_version(self):
        """В главном конфиге тихоновская регуляризация должна быть включена."""
        tikh = self.base_config['shap_reg']['tikhonov']
        self.assertTrue(tikh['enabled'])
        self.assertGreater(float(tikh['lambda']), 0.0)
        self.assertEqual(int(tikh['order']), 2)

    def test_main_config_uses_stronger_regularization_profile(self):
        """Основной конфиг должен оставаться в актуальном V2 SHAP+Tikhonov режиме."""
        shap = self.official_config['shap_reg']
        self.assertAlmostEqual(float(shap['gamma']), 0.099, places=8)
        self.assertAlmostEqual(float(shap['gamma_start']), 0.02, places=8)
        self.assertAlmostEqual(float(shap['gamma_end']), 0.099, places=8)
        self.assertAlmostEqual(float(shap['gamma_warmup_epochs']), 0.5, places=8)
        self.assertAlmostEqual(float(shap['target_shap_ratio']), 0.3895, places=8)
        self.assertAlmostEqual(float(shap['min_convergence_slowdown']), 0.25, places=8)
        self.assertTrue(bool(shap['use_true_shap']))
        self.assertTrue(bool(shap['use_adaptive_gamma']))
        self.assertTrue(bool(shap['use_adaptive_weights']))
        self.assertEqual(
            list(shap['active_components']),
            ['consistency', 'sparsity', 'faithfulness', 'stability'],
        )
        self.assertAlmostEqual(float(shap['tikhonov']['lambda']), 0.001, places=8)

    def test_v2_config_enables_energy_aware_tikhonov_nonnegativity_and_band_scalarization(self):
        """V2-конфиг должен включать новые физические priors и band-aware scalarization."""
        shap = self.official_v2_config['shap_reg']
        self.assertAlmostEqual(float(shap['gamma']), 0.099, places=12)
        self.assertAlmostEqual(float(shap['gamma_end']), 0.099, places=12)
        self.assertAlmostEqual(float(shap['target_shap_ratio']), 0.3895, places=12)
        self.assertEqual(shap['scalarization']['mode'], 'band_weighted')
        np.testing.assert_allclose(
            np.asarray(shap['scalarization']['band_weights'], dtype=float),
            np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(bool(shap['tikhonov']['energy_aware']))
        self.assertTrue(bool(shap['nonnegativity']['enabled']))
        self.assertAlmostEqual(float(shap['tikhonov']['lambda']), 0.001, places=12)
        self.assertAlmostEqual(float(shap['nonnegativity']['lambda']), 0.0038, places=12)
        self.assertAlmostEqual(float(shap['nonnegativity']['lambda_start']), 0.0, places=12)
        self.assertAlmostEqual(float(shap['nonnegativity']['lambda_end']), 0.0038, places=12)
        self.assertAlmostEqual(float(shap['nonnegativity']['warmup_epochs']), 0.40, places=12)
        self.assertEqual(shap['nonnegativity']['mode'], 'hybrid_mass_softcount')
        self.assertAlmostEqual(float(shap['nonnegativity']['soft_count_weight']), 0.28, places=12)
        self.assertAlmostEqual(float(shap['nonnegativity']['soft_count_temperature']), 0.012, places=12)

    def test_shap_stage_is_reproducible_with_fixed_seed(self):
        """При фиксированном seed SHAP-stage должен быть воспроизводимым."""
        cfg = deepcopy(self.official_v2_config)
        cfg['model']['num_rules'] = 3
        cfg['model']['optim_params']['epoch'] = 2
        cfg['model']['optim_params']['pop_size'] = 5
        cfg['model']['n_workers'] = 1
        cfg['shap_reg']['use_gpu'] = False
        cfg['shap_reg']['use_true_shap'] = False
        cfg['shap_reg']['epochs'] = 2
        cfg['shap_reg']['batch_size'] = 8
        cfg['shap_reg']['lr'] = 0.001

        rng = np.random.default_rng(20260320)
        X_train = rng.random((16, 10), dtype=np.float32)
        y_train = rng.random((16, 60), dtype=np.float32)

        trainer_a = self._make_trainer(cfg)
        history_a = trainer_a.fit(X_train, y_train, epochs=2, batch_size=8, lr=0.001)
        pred_a = trainer_a.predict(X_train)

        trainer_b = self._make_trainer(cfg)
        history_b = trainer_b.fit(X_train, y_train, epochs=2, batch_size=8, lr=0.001)
        pred_b = trainer_b.predict(X_train)

        for key in ['total_loss', 'main_loss', 'shap_loss', 'tikhonov_loss', 'nonnegativity_loss']:
            np.testing.assert_allclose(history_a[key], history_b[key], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(pred_a, pred_b, rtol=0.0, atol=1e-7)

    def test_order1_tikhonov_zero_for_constant_spectrum(self):
        """Для константного спектра D1 штраф должен быть нулевым."""
        trainer = self._make_trainer()
        trainer.tikhonov_order = 1

        predictions = torch.ones((2, 60), dtype=torch.float32, device=trainer.device)
        loss = trainer._compute_tikhonov_loss(predictions)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=8)

    def test_order2_tikhonov_zero_for_linear_spectrum(self):
        """Для линейного спектра D2 штраф должен быть нулевым."""
        cfg = deepcopy(self.base_config)
        cfg['shap_reg']['tikhonov']['energy_aware'] = False
        trainer = self._make_trainer(cfg)
        trainer.tikhonov_order = 2

        linear = torch.arange(60, dtype=torch.float32, device=trainer.device)
        predictions = linear.unsqueeze(0).repeat(3, 1)
        loss = trainer._compute_tikhonov_loss(predictions)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=8)

    def test_order2_tikhonov_positive_for_quadratic_spectrum(self):
        """Для квадратичного спектра D2 штраф должен быть положительным и предсказуемым."""
        cfg = deepcopy(self.base_config)
        cfg['shap_reg']['tikhonov']['energy_aware'] = False
        trainer = self._make_trainer(cfg)
        trainer.tikhonov_order = 2

        k = torch.arange(60, dtype=torch.float32, device=trainer.device)
        quadratic = (k ** 2).unsqueeze(0)
        loss = trainer._compute_tikhonov_loss(quadratic)

        # Для y_k = k^2 вторая разность постоянна и равна 2, значит mean(diffs^2)=4.
        self.assertAlmostEqual(float(loss.item()), 4.0, places=5)

    def test_energy_aware_order2_tikhonov_zero_for_log_linear_spectrum(self):
        """На log-energy оси линейная по log(E) зависимость должна давать нулевой D2-штраф."""
        cfg = deepcopy(self.official_v2_config)
        cfg['model']['num_rules'] = 3
        cfg['model']['optim_params']['epoch'] = 2
        cfg['model']['optim_params']['pop_size'] = 5
        cfg['model']['n_workers'] = 1
        cfg['shap_reg']['use_gpu'] = False
        trainer = self._make_trainer(cfg)

        energies = np.asarray(trainer.energy_axis, dtype=np.float32)
        self.assertEqual(energies.size, 60)
        log_linear = torch.tensor(np.log(energies), dtype=torch.float32, device=trainer.device).unsqueeze(0)
        loss = trainer._compute_tikhonov_loss(log_linear)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_nonnegativity_penalty_zero_for_nonnegative_predictions(self):
        """Штраф неотрицательности не должен активироваться на неотрицательном спектре."""
        cfg = deepcopy(self.official_v2_config)
        cfg['shap_reg']['use_gpu'] = False
        trainer = self._make_trainer(cfg)
        predictions = torch.rand((2, 60), dtype=torch.float32, device=trainer.device)
        loss = trainer._compute_nonnegativity_loss(predictions)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=8)

    def test_nonnegativity_penalty_positive_for_negative_bins(self):
        """Штраф неотрицательности должен быть положительным при отрицательных бинах."""
        cfg = deepcopy(self.official_v2_config)
        cfg['shap_reg']['use_gpu'] = False
        trainer = self._make_trainer(cfg)
        predictions = torch.zeros((1, 4), dtype=torch.float32, device=trainer.device) - torch.tensor(
            [[0.0, 1.0, 2.0, 0.5]], dtype=torch.float32, device=trainer.device
        )
        loss = trainer._compute_nonnegativity_loss(predictions)
        self.assertGreater(float(loss.item()), 0.0)

    def test_hybrid_nonnegativity_penalty_zero_for_nonnegative_predictions(self):
        """Hybrid nonnegativity не должен штрафовать полностью неотрицательный спектр."""
        cfg = deepcopy(self.official_v2_config)
        cfg['shap_reg']['use_gpu'] = False
        cfg['shap_reg']['nonnegativity']['mode'] = 'hybrid_mass_softcount'
        cfg['shap_reg']['nonnegativity']['soft_count_weight'] = 0.5
        cfg['shap_reg']['nonnegativity']['soft_count_temperature'] = 0.01
        trainer = self._make_trainer(cfg)
        predictions = torch.rand((2, 60), dtype=torch.float32, device=trainer.device)
        loss = trainer._compute_nonnegativity_loss(predictions)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=8)

    def test_hybrid_nonnegativity_penalizes_many_small_negative_bins_stronger_than_mass_ratio(self):
        """Hybrid-режим должен быть чувствительнее mass_ratio к множеству маленьких отрицательных хвостов."""
        cfg_mass = deepcopy(self.official_v2_config)
        cfg_mass['shap_reg']['use_gpu'] = False
        cfg_mass['shap_reg']['nonnegativity']['mode'] = 'mass_ratio'
        trainer_mass = self._make_trainer(cfg_mass)

        cfg_hybrid = deepcopy(self.official_v2_config)
        cfg_hybrid['shap_reg']['use_gpu'] = False
        cfg_hybrid['shap_reg']['nonnegativity']['mode'] = 'hybrid_mass_softcount'
        cfg_hybrid['shap_reg']['nonnegativity']['soft_count_weight'] = 0.5
        cfg_hybrid['shap_reg']['nonnegativity']['soft_count_temperature'] = 0.01
        trainer_hybrid = self._make_trainer(cfg_hybrid)

        predictions = torch.tensor(
            [[0.8, 0.9, 1.1, 1.0, -0.01, -0.01, -0.01, -0.01]],
            dtype=torch.float32,
            device=trainer_mass.device,
        )
        mass_loss = trainer_mass._compute_nonnegativity_loss(predictions)
        hybrid_loss = trainer_hybrid._compute_nonnegativity_loss(predictions)
        self.assertGreater(float(hybrid_loss.item()), float(mass_loss.item()))

    def test_margin_mass_ratio_tolerance_ignores_small_negative_tail(self):
        """Margin-версия nonnegativity не должна штрафовать отрицательность ниже допуска."""
        cfg = deepcopy(self.official_v2_config)
        cfg['shap_reg']['use_gpu'] = False
        cfg['shap_reg']['nonnegativity']['mode'] = 'margin_mass_ratio'
        cfg['shap_reg']['nonnegativity']['power'] = 1
        cfg['shap_reg']['nonnegativity']['tolerance'] = 0.10
        trainer = self._make_trainer(cfg)
        predictions = torch.tensor([[1.0, 1.0, -0.05, 1.0]], dtype=torch.float32, device=trainer.device)
        loss = trainer._compute_nonnegativity_loss(predictions)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=8)

    def test_margin_mass_ratio_penalizes_only_excess_negative_mass(self):
        """Margin-версия должна штрафовать только долю отрицательной массы сверх tolerance."""
        cfg = deepcopy(self.official_v2_config)
        cfg['shap_reg']['use_gpu'] = False
        cfg['shap_reg']['nonnegativity']['mode'] = 'margin_mass_ratio'
        cfg['shap_reg']['nonnegativity']['power'] = 1
        cfg['shap_reg']['nonnegativity']['tolerance'] = 0.10
        trainer = self._make_trainer(cfg)
        predictions = torch.tensor([[1.0, 1.0, -0.5, 1.0]], dtype=torch.float32, device=trainer.device)
        loss = trainer._compute_nonnegativity_loss(predictions)
        expected_ratio = 0.5 / 3.5 - 0.10
        self.assertAlmostEqual(float(loss.item()), expected_ratio, places=6)

    def test_band_weighted_scalarization_respects_band_weights(self):
        """Band-aware scalarization должна корректно агрегировать выход по заданным диапазонам."""
        cfg = deepcopy(self.official_v2_config)
        cfg['shap_reg']['use_gpu'] = False
        trainer = self._make_trainer(cfg)

        predictions = torch.cat([
            torch.full((1, 20), 1.0, dtype=torch.float32, device=trainer.device),
            torch.full((1, 20), 2.0, dtype=torch.float32, device=trainer.device),
            torch.full((1, 20), 4.0, dtype=torch.float32, device=trainer.device),
        ], dim=1)
        scalar = trainer._scalarize_output_tensor(predictions)
        expected = (1.0 + 2.0 + 4.0) / 3.0
        self.assertAlmostEqual(float(scalar.item()), expected, places=6)

    def test_training_history_contains_positive_tikhonov_loss(self):
        """В основном тихоновском режиме история обучения должна содержать tikhonov_loss."""
        trainer = self._make_trainer()
        rng = np.random.default_rng(42)
        X_train = rng.random((24, 10), dtype=np.float32)
        y_train = rng.random((24, 60), dtype=np.float32)

        history = trainer.fit(X_train, y_train, epochs=2, batch_size=8, lr=0.001)

        self.assertIn('tikhonov_loss', history)
        self.assertEqual(len(history['tikhonov_loss']), 2)
        self.assertTrue(all(np.isfinite(v) for v in history['tikhonov_loss']))
        self.assertTrue(any(v > 0 for v in history['tikhonov_loss']))
        self.assertTrue(all(t >= m for t, m in zip(history['total_loss'], history['main_loss'])))

    def test_disabling_tikhonov_makes_history_term_zero(self):
        """При отключении тихоновской регуляризации соответствующий член должен обнулиться."""
        cfg = deepcopy(self.base_config)
        cfg['shap_reg']['tikhonov']['enabled'] = False
        cfg['shap_reg']['tikhonov']['lambda'] = 0.0

        trainer = self._make_trainer(cfg)
        rng = np.random.default_rng(7)
        X_train = rng.random((16, 10), dtype=np.float32)
        y_train = rng.random((16, 60), dtype=np.float32)

        history = trainer.fit(X_train, y_train, epochs=2, batch_size=8, lr=0.001)

        self.assertIn('tikhonov_loss', history)
        self.assertEqual(len(history['tikhonov_loss']), 2)
        self.assertTrue(all(abs(v) < 1e-12 for v in history['tikhonov_loss']))

    def test_fixed_component_weights_are_taken_from_config(self):
        """При отключении адаптивных весов используются коэффициенты из конфига."""
        cfg = deepcopy(self.base_config)
        cfg['shap_reg']['use_adaptive_weights'] = False
        cfg['shap_reg']['gamma_consistency'] = 2.0
        cfg['shap_reg']['gamma_sparsity'] = 3.0
        cfg['shap_reg']['gamma_faithfulness'] = 4.0
        cfg['shap_reg']['gamma_stability'] = 1.0

        trainer = self._make_trainer(cfg)
        weights = trainer.fixed_component_weights

        self.assertAlmostEqual(weights['consistency'], 0.2, places=8)
        self.assertAlmostEqual(weights['sparsity'], 0.3, places=8)
        self.assertAlmostEqual(weights['faithfulness'], 0.4, places=8)
        self.assertAlmostEqual(weights['stability'], 0.1, places=8)

    def test_adaptive_gamma_warmup_is_monotone_and_plateaus(self):
        """Gamma должен линейно расти на warmup и затем удерживаться на gamma_end."""
        cfg = deepcopy(self.base_config)
        cfg['shap_reg']['use_true_shap'] = False
        cfg['shap_reg']['use_adaptive_gamma'] = True
        cfg['shap_reg']['gamma_start'] = 0.0
        cfg['shap_reg']['gamma_end'] = 0.4
        cfg['shap_reg']['gamma_warmup_epochs'] = 0.5
        cfg['shap_reg']['epochs'] = 4
        cfg['shap_reg']['batch_size'] = 8
        cfg['shap_reg']['gamma'] = 0.4

        trainer = self._make_trainer(cfg)
        rng = np.random.default_rng(123)
        X_train = rng.random((8, 10), dtype=np.float32)
        y_train = rng.random((8, 60), dtype=np.float32)

        history = trainer.fit(X_train, y_train, epochs=4, batch_size=8, lr=0.001)

        self.assertIn('adaptive_gamma', history)
        self.assertEqual(len(history['adaptive_gamma']), 4)
        self.assertAlmostEqual(history['adaptive_gamma'][0], 0.0, places=7)
        self.assertAlmostEqual(history['adaptive_gamma'][1], 0.2, places=7)
        self.assertAlmostEqual(history['adaptive_gamma'][2], 0.4, places=7)
        self.assertAlmostEqual(history['adaptive_gamma'][3], 0.4, places=7)
        self.assertTrue(all(
            history['adaptive_gamma'][i] <= history['adaptive_gamma'][i + 1] + 1e-12
            for i in range(len(history['adaptive_gamma']) - 1)
        ))

    def test_regularizer_lambda_warmup_is_monotone_and_plateaus(self):
        """Tikhonov и nonnegativity должны уметь плавно разогреваться по эпохам."""
        cfg = deepcopy(self.base_config)
        cfg['shap_reg']['epochs'] = 4
        cfg['shap_reg']['use_true_shap'] = False
        cfg['shap_reg']['use_adaptive_gamma'] = False
        cfg['shap_reg']['gamma'] = 0.0
        cfg['shap_reg']['tikhonov']['enabled'] = True
        cfg['shap_reg']['tikhonov']['lambda'] = 0.4
        cfg['shap_reg']['tikhonov']['lambda_start'] = 0.0
        cfg['shap_reg']['tikhonov']['lambda_end'] = 0.4
        cfg['shap_reg']['tikhonov']['warmup_epochs'] = 0.5
        cfg['shap_reg']['nonnegativity']['enabled'] = True
        cfg['shap_reg']['nonnegativity']['lambda'] = 0.2
        cfg['shap_reg']['nonnegativity']['lambda_start'] = 0.0
        cfg['shap_reg']['nonnegativity']['lambda_end'] = 0.2
        cfg['shap_reg']['nonnegativity']['warmup_epochs'] = 0.5

        trainer = self._make_trainer(cfg)
        rng = np.random.default_rng(1234)
        X_train = rng.random((8, 10), dtype=np.float32)
        y_train = rng.random((8, 60), dtype=np.float32)

        history = trainer.fit(X_train, y_train, epochs=4, batch_size=8, lr=0.001)

        np.testing.assert_allclose(history['tikhonov_lambda'], [0.0, 0.2, 0.4, 0.4], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(history['nonnegativity_lambda'], [0.0, 0.1, 0.2, 0.2], rtol=0.0, atol=1e-12)
        self.assertTrue(all(
            history['tikhonov_lambda'][i] <= history['tikhonov_lambda'][i + 1] + 1e-12
            for i in range(len(history['tikhonov_lambda']) - 1)
        ))
        self.assertTrue(all(
            history['nonnegativity_lambda'][i] <= history['nonnegativity_lambda'][i + 1] + 1e-12
            for i in range(len(history['nonnegativity_lambda']) - 1)
        ))

    def test_history_tracks_regularization_contributions(self):
        """История должна хранить не только raw loss, но и вклад регуляризаций в total loss."""
        cfg = deepcopy(self.base_config)
        cfg['shap_reg']['use_adaptive_gamma'] = False
        cfg['shap_reg']['gamma'] = 0.2

        trainer = self._make_trainer(cfg)
        rng = np.random.default_rng(2026)
        X_train = rng.random((24, 10), dtype=np.float32)
        y_train = rng.random((24, 60), dtype=np.float32)

        history = trainer.fit(X_train, y_train, epochs=2, batch_size=8, lr=0.001)

        for key in [
            'shap_loss_normalized',
            'shap_scale_factor',
            'shap_contribution',
            'tikhonov_contribution',
            'regularization_share',
            'shap_weight_consistency',
            'shap_weight_sparsity',
            'shap_weight_faithfulness',
            'shap_weight_stability',
        ]:
            self.assertIn(key, history)
            self.assertEqual(len(history[key]), 2)
            self.assertTrue(all(np.isfinite(v) for v in history[key]))

        self.assertTrue(any(v > 0.0 for v in history['shap_contribution']))
        self.assertTrue(any(v > 0.0 for v in history['tikhonov_contribution']))
        self.assertTrue(all(v >= 0.0 for v in history['regularization_share']))
        self.assertTrue(all(abs(v) < 1e-12 for v in history['shap_weight_consistency']))
        self.assertTrue(all(v >= 0.0 for v in history['shap_weight_sparsity']))
        self.assertTrue(all(abs(v) < 1e-12 for v in history['shap_weight_faithfulness']))
        self.assertTrue(all(v >= 0.0 for v in history['shap_weight_stability']))

    def test_v2_history_tracks_nonnegativity_contribution(self):
        """V2-режим должен логировать вклад nonnegativity-члена в историю обучения."""
        cfg = deepcopy(self.official_v2_config)
        cfg['model']['num_rules'] = 3
        cfg['model']['optim_params']['epoch'] = 2
        cfg['model']['optim_params']['pop_size'] = 5
        cfg['model']['n_workers'] = 1
        cfg['shap_reg']['use_gpu'] = False
        cfg['shap_reg']['use_true_shap'] = False
        cfg['shap_reg']['use_adaptive_gamma'] = False
        cfg['shap_reg']['epochs'] = 2
        cfg['shap_reg']['batch_size'] = 8
        cfg['shap_reg']['gamma'] = 0.1

        trainer = self._make_trainer(cfg)
        rng = np.random.default_rng(2028)
        X_train = rng.random((24, 10), dtype=np.float32)
        y_train = rng.random((24, 60), dtype=np.float32)

        history = trainer.fit(X_train, y_train, epochs=2, batch_size=8, lr=0.001)

        self.assertIn('nonnegativity_loss', history)
        self.assertIn('nonnegativity_contribution', history)
        self.assertEqual(len(history['nonnegativity_loss']), 2)
        self.assertEqual(len(history['nonnegativity_contribution']), 2)
        self.assertTrue(all(np.isfinite(v) for v in history['nonnegativity_loss']))
        self.assertTrue(all(np.isfinite(v) for v in history['nonnegativity_contribution']))

    def test_active_components_mask_disables_unlisted_shap_terms(self):
        """Для абляций должны выключаться только явно убранные SHAP-компоненты."""
        cfg = deepcopy(self.base_config)
        cfg['shap_reg']['use_true_shap'] = True
        cfg['shap_reg']['true_shap_update_frequency'] = 1
        cfg['shap_reg']['use_adaptive_weights'] = False
        cfg['shap_reg']['active_components'] = ['sparsity']

        trainer = self._make_trainer(cfg)
        rng = np.random.default_rng(2027)
        X_train = rng.random((16, 10), dtype=np.float32)
        y_train = rng.random((16, 60), dtype=np.float32)

        history = trainer.fit(X_train, y_train, epochs=2, batch_size=8, lr=0.001)

        self.assertAlmostEqual(trainer.fixed_component_weights['sparsity'], 1.0, places=8)
        self.assertAlmostEqual(trainer.fixed_component_weights['consistency'], 0.0, places=8)
        self.assertAlmostEqual(trainer.fixed_component_weights['faithfulness'], 0.0, places=8)
        self.assertAlmostEqual(trainer.fixed_component_weights['stability'], 0.0, places=8)
        self.assertTrue(all(abs(v) < 1e-12 for v in history['shap_consistency']))
        self.assertTrue(all(abs(v) < 1e-12 for v in history['shap_faithfulness']))
        self.assertTrue(all(abs(v) < 1e-12 for v in history['shap_stability']))
        self.assertTrue(any(v > 0.0 for v in history['shap_sparsity']))


class TestTikhonovTrainPipeline(unittest.TestCase):
    """Интеграционные тесты ожидаемого поведения полного train.py для Tikhonov-версии."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).parent.parent
        cls.data_path = cls.repo_root / "normalized_data_with_q_375.csv"
        cls.base_config_path = cls.repo_root / "configs" / "config_integrated_shap.yaml"
        cls.base_config = load_config(str(cls.base_config_path))
        package_summaries = cls.repo_root / "final_send_package_20260320_v2_1" / "summaries"
        cls.baseline_summary_path = package_summaries / "training_summary_20260319_202741_vanilla_full_20260319.json"
        cls.v2_previous_summary_path = package_summaries / "training_summary_20260320_055350_v2_official_det_20260320.json"
        cls.v2_official_summary_path = cls.repo_root / "results" / "training_summary_20260320_062903_v2_1_light_nonneg_20260320.json"
        cls.v2_1_light_summary_path = cls.repo_root / "results" / "training_summary_20260320_062903_v2_1_light_nonneg_20260320.json"

    def _write_smoke_config(
        self,
        results_dir: Path,
        *,
        save_model: bool = True,
        save_predictions: bool = True,
    ) -> Path:
        config_text = f"""dataset:
  train_data: {self.data_path}
  validation_data: {self.data_path}
  normalize_sum: true
  mix_with_real: false
  mix_ratio: 0.0
  test_size: 0.2
  random_state: 42
  feature_prefix: "Q"
  feature_count: 10
  feature_index_start: 1
  target_prefix: ""
  target_count: 60
  target_index_start: 1

model:
  num_rules: 3
  mf_class: Gaussian
  vanishing_strategy: blend
  optim: OriginalPSO
  reg_lambda: 0.001
  seed: 42
  n_workers: 1
  optim_params:
    epoch: 2
    pop_size: 5
    verbose: false

shap_reg:
  enabled: true
  integrated_training: false
  training_mode: real_only
  use_pso_init: false
  pso_epochs: 2
  epochs: 3
  batch_size: 8
  lr: 0.001
  gamma: 0.05
  gamma_start: 0.0
  gamma_end: 0.05
  gamma_warmup_epochs: 0.5
  use_adaptive_gamma: true
  use_adaptive_weights: true
  use_improved_shap: true
  use_true_shap: false
  use_gpu: false
  gamma_sparsity: 0.7
  gamma_consistency: 0.2
  gamma_faithfulness: 0.05
  gamma_stability: 0.05
  tikhonov:
    enabled: true
    lambda: 0.001
    order: 2

output:
  results_dir: {results_dir}
  save_model: {'true' if save_model else 'false'}
  save_predictions: {'true' if save_predictions else 'false'}
  save_plots: false
  save_samples: false
"""
        config_path = results_dir.parent / "smoke_config.yaml"
        config_path.write_text(config_text, encoding="utf-8")
        return config_path

    def test_real_split_keeps_sum_aligned_with_test_subset(self):
        """SUM для denorm должен делиться тем же split, что и X/y тестовой части."""
        X_real = np.column_stack([
            np.arange(20, dtype=np.float32),
            np.linspace(0.0, 1.0, 20, dtype=np.float32),
        ])
        y_real = np.tile(np.arange(60, dtype=np.float32), (20, 1))
        sum_real = 1000.0 + X_real[:, 0]

        (
            _X_train,
            _X_val,
            X_test,
            _y_train,
            _y_val,
            _y_test,
            sum_test,
        ) = _split_real_data_for_shap(
            X_real,
            y_real,
            sum_real,
            normalize_sum=True,
            random_state=42,
        )

        self.assertEqual(len(X_test), 4)
        self.assertIsNotNone(sum_test)
        np.testing.assert_allclose(sum_test, 1000.0 + X_test[:, 0], rtol=0.0, atol=1e-7)

    def test_prepare_feature_importance_normalizes_and_validates_size(self):
        """Сохранение важностей должно давать корректное распределение и ловить mismatch размерности."""
        series = _prepare_feature_importance([2.0, np.nan, -3.0, 2.0], ["a", "b", "c", "d"], normalize=True)
        self.assertAlmostEqual(float(series.sum()), 1.0, places=8)
        self.assertTrue(np.all(np.isfinite(series.to_numpy())))
        self.assertTrue(np.all(series.to_numpy() >= 0.0))
        self.assertAlmostEqual(float(series["a"]), 0.5, places=8)
        self.assertAlmostEqual(float(series["d"]), 0.5, places=8)

        with self.assertRaises(ValueError):
            _prepare_feature_importance([1.0, 2.0], ["a"], normalize=True)

    def test_train_py_smoke_produces_consistent_summary_and_history(self):
        """Tiny run через train.py должен давать корректные артефакты и невырожденные метрики."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_dir = tmp_path / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            config_path = self._write_smoke_config(results_dir)

            args = SimpleNamespace(
                config=str(config_path),
                train_limit=32,
                train_fraction=None,
                tag="pytest_smoke_tikhonov"
            )

            model_state_path, summary_path = train_and_save(args)

            self.assertTrue(Path(model_state_path).exists())
            self.assertTrue(Path(summary_path).exists())

            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            self.assertIn("metrics", summary)
            self.assertIn("dataset_settings", summary)
            self.assertIn("shap_files", summary)

            metrics = summary["metrics"]
            self.assertTrue(np.isfinite(metrics["mse"]))
            self.assertTrue(np.isfinite(metrics["rmse"]))
            self.assertTrue(np.isfinite(metrics["mae"]))
            self.assertTrue(np.isfinite(metrics["r2_weighted"]))
            self.assertLess(metrics["mse"], 0.1)
            self.assertGreater(metrics["r2_weighted"], 0.1)

            dataset_settings = summary["dataset_settings"]
            self.assertAlmostEqual(dataset_settings["test_size"], 0.2, places=12)
            self.assertEqual(dataset_settings["real_data_split"], {"train": 0.6, "validation": 0.2, "test": 0.2})
            self.assertAlmostEqual(dataset_settings["synthetic_test_size"], 0.2, places=12)
            self.assertEqual(summary["shap_train_size"], 225)
            self.assertEqual(summary["test_size"], 75)
            self.assertEqual(summary["vanilla_train_count"], summary["train_size"])
            self.assertEqual(summary["shap_train_count"], summary["shap_train_size"])
            self.assertEqual(summary["real_test_count"], summary["test_size"])
            self.assertGreater(summary["training_time_total"], 0.0)
            self.assertGreater(summary["training_time_shap"], 0.0)
            self.assertGreaterEqual(summary["training_time_total"], summary["training_time_shap"])

            self.assertIn("metrics_denorm", summary)
            self.assertIn("r2_weighted", summary["metrics_denorm"])
            self.assertIn("r2_mean", summary["metrics_denorm"])
            self.assertAlmostEqual(summary["metrics_denorm"]["r2"], summary["metrics_denorm"]["r2_weighted"], places=12)
            self.assertIn("diagnostics", summary)
            self.assertIn("regularization", summary["diagnostics"])
            self.assertIn("dominant_regularizer", summary["diagnostics"]["regularization"])
            self.assertIn("component_terms", summary["diagnostics"]["regularization"])
            self.assertIn("component_weights", summary["diagnostics"]["regularization"])
            self.assertIn("weighted_component_signal", summary["diagnostics"]["regularization"])
            self.assertIn("dominant_shap_component", summary["diagnostics"]["regularization"])

            history_path = results_dir / summary["shap_files"]["history"]
            self.assertTrue(history_path.exists())
            history = json.loads(history_path.read_text(encoding="utf-8"))

            expected_epochs = 3
            for key in [
                "total_loss",
                "main_loss",
                "shap_loss",
                "shap_loss_normalized",
                "tikhonov_loss",
                "shap_contribution",
                "tikhonov_contribution",
                "regularization_share",
                "adaptive_gamma",
                "shap_sparsity",
                "shap_faithfulness",
                "shap_stability",
                "shap_weight_consistency",
                "shap_weight_sparsity",
                "shap_weight_faithfulness",
                "shap_weight_stability",
            ]:
                self.assertIn(key, history)
                self.assertEqual(len(history[key]), expected_epochs)
                self.assertTrue(all(np.isfinite(v) for v in history[key]))

            self.assertEqual(history["adaptive_gamma"][0], 0.0)
            self.assertAlmostEqual(history["adaptive_gamma"][-1], 0.05, places=7)
            self.assertTrue(all(v > 0 for v in history["tikhonov_loss"]))

            shap_csv_path = results_dir / summary["shap_files"]["feature_importance_shap"]
            self.assertTrue(shap_csv_path.exists())
            with shap_csv_path.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            importances = [float(row["importance"]) for row in rows]
            self.assertEqual(len(importances), 10)
            self.assertTrue(all(np.isfinite(v) and v >= 0.0 for v in importances))
            self.assertAlmostEqual(sum(importances), 1.0, places=6)
            self.assertAlmostEqual(
                summary["diagnostics"]["regularization"]["component_weights"]["sparsity"]["mean"],
                14.0 / 15.0,
                places=8,
            )
            self.assertAlmostEqual(
                summary["diagnostics"]["regularization"]["component_weights"]["stability"]["mean"],
                1.0 / 15.0,
                places=8,
            )

    def test_train_py_reports_prediction_stats_without_saving_predictions_and_respects_save_model(self):
        """Диагностика не должна зависеть от save_predictions, а save_model=false не должен тихо писать файл."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            results_dir = tmp_path / "results_no_artifacts"
            results_dir.mkdir(parents=True, exist_ok=True)
            config_path = self._write_smoke_config(
                results_dir,
                save_model=False,
                save_predictions=False,
            )

            args = SimpleNamespace(
                config=str(config_path),
                train_limit=32,
                train_fraction=None,
                tag="pytest_smoke_no_saved_predictions"
            )

            model_state_path, summary_path = train_and_save(args)

            self.assertIsNone(model_state_path)
            self.assertTrue(Path(summary_path).exists())

            summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            prediction_stats = summary["diagnostics"]["prediction_stats"]
            self.assertIn("negative_fraction", prediction_stats)
            self.assertIn("negative_count", prediction_stats)
            self.assertTrue(np.isfinite(prediction_stats["mean"]))
            self.assertTrue(np.isfinite(prediction_stats["std"]))
            self.assertGreaterEqual(prediction_stats["negative_fraction"], 0.0)
            self.assertGreaterEqual(prediction_stats["negative_count"], 0)

            self.assertIsNone(summary["model_state"])
            self.assertIsNone(summary["model_state_path"])
            self.assertNotIn("predictions", summary["saved_files"])

    def test_official_v2_artifact_outperforms_baselines_on_primary_metrics(self):
        """Текущий основной V2.1 run должен быть лучше vanilla baseline и предыдущего official V2."""
        self.assertTrue(self.baseline_summary_path.exists())
        self.assertTrue(self.v2_previous_summary_path.exists())
        self.assertTrue(self.v2_official_summary_path.exists())

        baseline = json.loads(self.baseline_summary_path.read_text(encoding="utf-8"))
        previous_v2 = json.loads(self.v2_previous_summary_path.read_text(encoding="utf-8"))
        v2 = json.loads(self.v2_official_summary_path.read_text(encoding="utf-8"))

        baseline_metrics = baseline["metrics"]
        previous_v2_metrics = previous_v2["metrics"]
        v2_metrics = v2["metrics"]

        self.assertLess(v2_metrics["mse"], baseline_metrics["mse"])
        self.assertLess(v2_metrics["rmse"], baseline_metrics["rmse"])
        self.assertGreater(v2_metrics["r2_weighted"], baseline_metrics["r2_weighted"])
        self.assertGreater(v2_metrics["r2_mean"], baseline_metrics["r2_mean"])

        self.assertLess(v2_metrics["mse"], previous_v2_metrics["mse"])
        self.assertLess(v2_metrics["rmse"], previous_v2_metrics["rmse"])
        self.assertLess(v2_metrics["mae"], previous_v2_metrics["mae"])
        self.assertGreater(v2_metrics["r2_weighted"], previous_v2_metrics["r2_weighted"])

        prediction_stats = v2["diagnostics"]["prediction_stats"]
        previous_prediction_stats = previous_v2["diagnostics"]["prediction_stats"]
        self.assertIn("negative_fraction", prediction_stats)
        self.assertIn("negative_count", prediction_stats)
        self.assertGreaterEqual(prediction_stats["negative_fraction"], 0.0)
        self.assertGreaterEqual(prediction_stats["negative_count"], 0)
        self.assertLess(prediction_stats["negative_fraction"], previous_prediction_stats["negative_fraction"])

    def test_v2_1_light_candidate_improves_primary_metrics_and_reduces_negative_fraction(self):
        """Текущий V2.1 должен улучшать primary metrics и заметно уменьшать долю отрицательных бинов относительно previous official V2."""
        self.assertTrue(self.v2_previous_summary_path.exists())
        self.assertTrue(self.v2_1_light_summary_path.exists())

        official = json.loads(self.v2_previous_summary_path.read_text(encoding="utf-8"))
        candidate = json.loads(self.v2_1_light_summary_path.read_text(encoding="utf-8"))

        official_metrics = official["metrics"]
        candidate_metrics = candidate["metrics"]

        for metric in ["mse", "rmse", "mae"]:
            self.assertLess(candidate_metrics[metric], official_metrics[metric])
        self.assertGreater(candidate_metrics["r2_weighted"], official_metrics["r2_weighted"])

        official_neg = official["diagnostics"]["prediction_stats"]["negative_fraction"]
        candidate_neg = candidate["diagnostics"]["prediction_stats"]["negative_fraction"]
        self.assertLess(candidate_neg, official_neg)
        self.assertLess(candidate_neg, official_neg - 0.05)


class TestAblationStudyUtilities(unittest.TestCase):
    """Тесты утилит абляционного анализа."""

    def test_shap_distribution_metrics_capture_non_uniformity(self):
        """Для неравномерного SHAP-распределения Gini должен быть положительным."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            shap_csv = tmp_path / "feature_importance_shap.csv"
            summary_path = tmp_path / "training_summary.json"

            with shap_csv.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["", "importance"])
                writer.writerow(["Q1", 0.70])
                writer.writerow(["Q2", 0.20])
                writer.writerow(["Q3", 0.10])

            summary = {"shap_files": {"feature_importance_shap": shap_csv.name}}
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            metrics = _compute_shap_distribution_metrics(summary, summary_path)

            self.assertGreater(metrics["shap_gini"], 0.0)
            self.assertLess(metrics["shap_entropy"], 1.0)
            self.assertAlmostEqual(metrics["shap_top3_mass"], 1.0, places=8)
            self.assertEqual(metrics["shap_top3"], "Q1,Q2,Q3")

    def test_component_ablation_can_target_stronger_shap_mode(self):
        """Компонентный ablation должен поддерживать более сильный SHAP-режим."""
        _, component_variants = _variant_definitions(component_mode="stronger")
        variants = dict(component_variants)

        self.assertIn("strong_no_consistency", variants)
        self.assertIn("strong_no_sparsity", variants)
        self.assertIn("strong_no_faithfulness", variants)
        self.assertIn("strong_no_stability", variants)

        strong_no_sparsity = variants["strong_no_sparsity"]["shap_reg"]
        self.assertAlmostEqual(strong_no_sparsity["gamma"], 0.099, places=8)
        self.assertAlmostEqual(strong_no_sparsity["gamma_end"], 0.099, places=8)
        self.assertAlmostEqual(strong_no_sparsity["target_shap_ratio"], 0.3895, places=8)
        self.assertAlmostEqual(strong_no_sparsity["min_convergence_slowdown"], 0.25, places=8)
        self.assertAlmostEqual(strong_no_sparsity["tikhonov"]["lambda"], 0.001, places=8)
        self.assertEqual(
            strong_no_sparsity["active_components"],
            ["consistency", "faithfulness", "stability"],
        )


if __name__ == '__main__':
    unittest.main()
