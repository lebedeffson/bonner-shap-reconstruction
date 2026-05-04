"""
Улучшенный SHAP-регуляризованный тренер ANFIS для восстановления спектра нейтронов
Использует улучшенную SHAP регуляризацию с 4 компонентами: Consistency, Sparsity, Faithfulness, Stability
Работает в двухэтапном режиме: сначала vanilla ANFIS, потом SHAP регуляризация
"""

import random
import time
import copy
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from constants import Ebins_float_IAEA_Comp
from src.models.shap_trainer_precision_optimized import PrecisionOptimizedSHAPRegularization
from src.regularizers.error_aware_attribution import error_importance
from src.utils.logger import get_logger


class ShapAwareANFISTrainerImproved:
    """Улучшенный тренер ANFIS с SHAP-регуляризацией для мультирегрессии"""
    COMPONENT_NAMES = ("consistency", "sparsity", "faithfulness", "stability")

    def __init__(self, model, config, gamma=0.5, verbose=True):
        """
        Инициализация тренера
        
        Args:
            model: ANFIS модель (BioAnfisRegressor)
            config: Конфигурация
            gamma: Коэффициент SHAP-регуляризации
            verbose: Выводить ли информацию
        """
        self.model = model.network
        self.gamma = gamma
        self.verbose = verbose
        self.config = config
        self.task_type = 'regression'  # Всегда регрессия
        self.training_time = 0
        shap_config = config.get('shap_reg', {})
        self.grad_clip = float(shap_config.get('grad_clip', 1.0))
        self.train_output_only = bool(shap_config.get('train_output_only', True))
        self.early_stopping_patience = int(shap_config.get('early_stopping_patience', 8))
        self.early_stopping_min_delta = float(shap_config.get('early_stopping_min_delta', 1e-4))
        self.restore_best_state = bool(shap_config.get('restore_best_state', True))
        guard_cfg = shap_config.get('accuracy_guard', {})
        self.accuracy_guard_enabled = bool(guard_cfg.get('enabled', True))
        self.accuracy_guard_margin = float(guard_cfg.get('margin', 0.02))
        self.accuracy_guard_weight = float(guard_cfg.get('weight', 10.0))
        self.max_shap_ratio = float(shap_config.get('max_shap_ratio', 0.15))
        self.max_guard_ratio = float(shap_config.get('max_guard_ratio', 0.20))
        self.use_distill_anchor = bool(shap_config.get('use_distill_anchor', True))
        self.distill_weight = float(shap_config.get('distill_weight', 0.15))
        self.max_distill_ratio = float(shap_config.get('max_distill_ratio', 0.20))
        self.quality_first = bool(shap_config.get('quality_first', True))
        self.reject_on_val_degrade = bool(shap_config.get('reject_on_val_degrade', True))
        self.quality_tolerance = float(shap_config.get('quality_tolerance', 0.0))
        self.lr_backoff = float(shap_config.get('lr_backoff', 0.5))
        self.min_lr = float(shap_config.get('min_lr', 1e-6))
        self.max_coeff_delta_ratio = float(shap_config.get('max_coeff_delta_ratio', 0.10))
        self.use_feature_gates = bool(shap_config.get('use_feature_gates', True))
        self.gate_init = float(shap_config.get('gate_init', 0.9))
        self.gate_min = float(shap_config.get('gate_min', 0.0))
        self.gate_max = float(shap_config.get('gate_max', 1.0))
        self.autonomous_error_shap = bool(shap_config.get('autonomous_error_shap', True))
        self.error_mse_weight = float(shap_config.get('error_mse_weight', 0.5))
        self.error_js_weight = float(shap_config.get('error_js_weight', 1.0))
        self.gate_js_weight = float(shap_config.get('gate_js_weight', 1.0))
        self.noise_weight = float(shap_config.get('noise_weight', 0.2))
        self.rule_stability_weight = float(shap_config.get('rule_stability_weight', 0.1))
        self.error_importance_ema_beta = float(shap_config.get('error_importance_ema_beta', 0.9))
        self.error_importance_mode = str(shap_config.get('error_importance_mode', 'permute')).strip().lower()
        self.error_importance_target = str(shap_config.get('error_importance_target', 'train')).strip().lower()
        if self.error_importance_target not in {'train', 'val'}:
            self.error_importance_target = 'train'
        self.error_importance_val_batch_size = int(shap_config.get('error_importance_val_batch_size', 0))
        self.error_importance_ema = None
        self.prev_error_importance_raw = None
        self.last_qerr_entropy = float('nan')
        self.last_qerr_gini = float('nan')
        self.last_qerr_corr = float('nan')
        self.last_p_q_corr = float('nan')
        self.last_eta_mean = float('nan')
        self.last_eta_std = float('nan')
        self.last_rank_pairs_count = 0.0
        self.last_rank_violations_count = 0.0
        self._last_rank_loss_tensor = None
        self.grad_importance_ema = None
        self.grad_importance_ema_beta = float(shap_config.get('grad_importance_ema_beta', 0.9))
        self.error_target_rho = float(shap_config.get('error_target_rho', 1.0))
        self.error_target_rho = min(max(self.error_target_rho, 0.0), 1.0)
        self.ea_alignment_loss = str(shap_config.get('ea_alignment_loss', 'cosine_mse')).strip().lower()
        self.ea_alignment_alpha = float(shap_config.get('ea_alignment_alpha', 0.5))
        self.ea_alignment_alpha = min(max(self.ea_alignment_alpha, 0.0), 1.0)
        self.ea_rank_weight = float(shap_config.get('ea_rank_weight', 0.0))
        self.ea_rank_type = str(shap_config.get('ea_rank_type', 'hinge_allpairs')).strip().lower()
        self.ea_rank_margin = float(shap_config.get('ea_rank_margin', 0.01))
        self.ea_rank_delta = float(shap_config.get('ea_rank_delta', 1e-4))
        self.ea_rank_tau = float(shap_config.get('ea_rank_tau', 0.2))
        self.ea_rank_top_k = int(shap_config.get('ea_rank_top_k', 3))
        self.ea_rank_bottom_k = int(shap_config.get('ea_rank_bottom_k', 3))
        self.ea_target_ablation_mode = str(
            shap_config.get('ea_target_ablation_mode', 'none')
        ).strip().lower()
        if self.ea_target_ablation_mode in {'shuffle', 'shuffled'}:
            self.ea_target_ablation_mode = 'shuffled_q_err'
        if self.ea_target_ablation_mode in {'random', 'rand'}:
            self.ea_target_ablation_mode = 'random_target'
        if self.ea_target_ablation_mode in {'frozen', 'freeze', 'frozen_target'}:
            self.ea_target_ablation_mode = 'frozen_q_err'
        if self.ea_target_ablation_mode in {'anti', 'inverse', 'anti_q', 'anti_q_err'}:
            self.ea_target_ablation_mode = 'anti_q_err'
        if self.ea_target_ablation_mode in {'uniform', 'uniform_target', 'flat'}:
            self.ea_target_ablation_mode = 'uniform_target'
        if self.ea_target_ablation_mode not in {'none', 'random_target', 'shuffled_q_err', 'frozen_q_err', 'anti_q_err', 'uniform_target'}:
            self.ea_target_ablation_mode = 'none'
        self.ea_positive_clipping = bool(shap_config.get('ea_positive_clipping', True))
        self.frozen_q_target = None
        self.ea_importance_source = str(shap_config.get('ea_importance_source', 'grad')).strip().lower()
        if self.ea_importance_source not in {'grad', 'gate', 'mixed'}:
            self.ea_importance_source = 'grad'
        self.ea_gate_importance_alpha = float(shap_config.get('ea_gate_importance_alpha', 0.5))
        self.ea_gate_importance_alpha = min(max(self.ea_gate_importance_alpha, 0.0), 1.0)
        self.ea_bottom_invariance_weight = float(shap_config.get('ea_bottom_invariance_weight', 0.0))
        self.ea_bottom_k = int(shap_config.get('ea_bottom_k', 3))
        self.ea_warmup_fraction = float(shap_config.get('ea_warmup_fraction', 0.25))
        self.ea_warmup_fraction = min(max(self.ea_warmup_fraction, 0.0), 0.95)
        self.ea_bypass_legacy_normalization = bool(shap_config.get('ea_bypass_legacy_normalization', True))
        self.debug_grad_norms = bool(shap_config.get('debug_grad_norms', False))
        self.grad_norm_interval = max(int(shap_config.get('grad_norm_interval', 20)), 1)
        self.ea_use_grad_balance = bool(shap_config.get('ea_use_grad_balance', False))
        self.ea_target_grad_ratio = float(shap_config.get('ea_target_grad_ratio', 0.005))
        self.ea_scale_min = float(shap_config.get('ea_scale_min', 0.1))
        self.ea_scale_max = float(shap_config.get('ea_scale_max', 1e4))
        self.ea_scale_min = max(self.ea_scale_min, 1e-6)
        self.ea_scale_max = max(self.ea_scale_max, self.ea_scale_min)
        self.last_ea_scale = 1.0
        self.feature_gate_logits = None
        self.gate_anchor_weight = float(shap_config.get('gate_anchor_weight', 0.05))
        self.gate_blend_warmup_epochs = float(shap_config.get('gate_blend_warmup_epochs', 0.4))
        self.random_seed = int(
            shap_config.get(
                'seed',
                config.get('model', {}).get(
                    'seed',
                    config.get('dataset', {}).get('random_state', 42)
                )
            )
        )
        
        # Параметры для использования настоящих Shapley values
        self.use_true_shap = shap_config.get('use_true_shap', True)
        self.true_shap_update_frequency = shap_config.get('true_shap_update_frequency', 10)
        self.true_shap_batch_count = 0
        self.true_shap_importance = None
        
        # Параметры улучшенной SHAP регуляризации
        self.use_improved_shap = shap_config.get('use_improved_shap', True)
        
        # Адаптивная нормализация SHAP loss
        self.main_loss_ema = None  # Скользящее среднее main loss
        self.ema_alpha = 0.9  # Коэффициент для экспоненциального скользящего среднего
        self.target_shap_ratio = shap_config.get('target_shap_ratio', 0.2)  # Целевое соотношение SHAP/main
        self.min_convergence_slowdown = float(shap_config.get('min_convergence_slowdown', 0.0))
        
        # Адаптивный gamma schedule для плавной сходимости
        self.use_adaptive_gamma = shap_config.get('use_adaptive_gamma', True)  # Использовать адаптивный gamma
        self.gamma_start = shap_config.get('gamma_start', 0.05)  # Начальное значение gamma (малое)
        self.gamma_end = shap_config.get('gamma_end', 0.5)  # Конечное значение gamma
        self.gamma_warmup_epochs = shap_config.get('gamma_warmup_epochs', 0.3)  # Доля эпох для разогрева (30%)
        self.current_epoch = 0  # Текущая эпоха для schedule
        self.total_epochs = None  # Общее количество эпох
        
        # Плавная сходимость: замедление обучения при улучшении
        self.use_convergence_smoothing = shap_config.get('use_convergence_smoothing', True)
        self.convergence_patience = shap_config.get('convergence_patience', 10)  # Терпение для замедления
        self.best_main_loss = None  # Лучший main loss
        self.no_improvement_count = 0  # Счетчик отсутствия улучшения
        
        # Адаптивные веса компонентов
        self.use_adaptive_weights = shap_config.get('use_adaptive_weights', True)
        self.component_weights_history = []  # История весов для анализа
        self.active_components = self._parse_active_components(shap_config.get('active_components'))
        if not self.use_true_shap and 'consistency' in self.active_components:
            self.active_components.discard('consistency')
        if not any(c in self.active_components for c in ('sparsity', 'faithfulness', 'stability')):
            self.active_components.add('sparsity')
        self.fixed_component_weights = self._normalize_component_weights({
            'consistency': float(shap_config.get('gamma_consistency', 0.2)),
            'sparsity': float(shap_config.get('gamma_sparsity', 0.7)),
            'faithfulness': float(shap_config.get('gamma_faithfulness', 0.05)),
            'stability': float(shap_config.get('gamma_stability', 0.05)),
        })
        self.fallback_component_weights = self._normalize_component_weights({
            'consistency': 0.0,
            'sparsity': float(shap_config.get('gamma_sparsity', 0.7)),
            'faithfulness': 0.0,
            'stability': float(shap_config.get('gamma_stability', 0.05)),
        })
        
        # Улучшенная Sparsity с Gini coefficient
        self.use_gini_sparsity = shap_config.get('use_gini_sparsity', True)
        self.target_gini = shap_config.get('target_gini', 0.3)  # Целевое значение Gini

        # Тихоновская регуляризация (гладкость спектра по выходу)
        tikhonov_config = shap_config.get('tikhonov', {})
        self.tikhonov_lambda = float(tikhonov_config.get('lambda', 0.0))
        self.tikhonov_order = int(tikhonov_config.get('order', 2))
        self.tikhonov_enabled = bool(tikhonov_config.get('enabled', self.tikhonov_lambda > 0.0))
        self.tikhonov_lambda_start = float(tikhonov_config.get('lambda_start', self.tikhonov_lambda))
        self.tikhonov_lambda_end = float(tikhonov_config.get('lambda_end', self.tikhonov_lambda))
        self.tikhonov_warmup_epochs = float(tikhonov_config.get('warmup_epochs', 0.0))
        self.tikhonov_energy_aware = bool(
            tikhonov_config.get('energy_aware', False) or
            tikhonov_config.get('log_energy_aware', False)
        )
        self.energy_axis = self._resolve_energy_axis(
            int(config.get('dataset', {}).get('target_count', 0) or 0)
        )

        # Мягкий штраф неотрицательности спектра
        nonneg_config = shap_config.get('nonnegativity', {})
        self.nonnegativity_lambda = float(nonneg_config.get('lambda', 0.0))
        self.nonnegativity_enabled = bool(
            nonneg_config.get('enabled', self.nonnegativity_lambda > 0.0)
        )
        self.nonnegativity_lambda_start = float(nonneg_config.get('lambda_start', self.nonnegativity_lambda))
        self.nonnegativity_lambda_end = float(nonneg_config.get('lambda_end', self.nonnegativity_lambda))
        self.nonnegativity_warmup_epochs = float(nonneg_config.get('warmup_epochs', 0.0))
        self.nonnegativity_power = max(int(nonneg_config.get('power', 2)), 1)
        self.nonnegativity_mode = str(nonneg_config.get('mode', 'power_mean')).strip().lower()
        self.nonnegativity_tolerance = max(float(nonneg_config.get('tolerance', 0.0)), 0.0)
        self.nonnegativity_soft_count_weight = float(nonneg_config.get('soft_count_weight', 0.0))
        self.nonnegativity_soft_count_weight = min(max(self.nonnegativity_soft_count_weight, 0.0), 1.0)
        self.nonnegativity_soft_count_temperature = max(
            float(nonneg_config.get('soft_count_temperature', 1e-2)),
            1e-8,
        )

        # Способ скаляризации выхода для SHAP-компонент
        scalarization_config = shap_config.get('scalarization', {})
        self.scalarization_mode = str(scalarization_config.get('mode', 'mean')).strip().lower()
        self.scalarization_bands = self._parse_band_slices(
            scalarization_config.get('band_slices'),
            config.get('dataset', {}).get('target_count')
        )
        self.scalarization_weights = self._parse_band_weights(
            scalarization_config.get('band_weights'),
            len(self.scalarization_bands)
        )
        
        # Логгер (создаем до использования)
        self.logger = get_logger("anfis_shap.shap_trainer_improved")
        if not verbose:
            self.logger.setLevel(30)  # WARNING level
        
        # Определяем устройство и перемещаем модель на GPU если доступно
        # Проверяем, нужно ли использовать GPU
        use_gpu = shap_config.get('use_gpu', True) and torch.cuda.is_available()
        
        if use_gpu:
            # Перемещаем модель на GPU
            self.model = self.model.to(torch.device('cuda'))
            self.device = torch.device('cuda')
            self.logger.info(f"Модель перемещена на GPU: {torch.cuda.get_device_name(0)}")
        else:
            # Используем CPU
            self.device = next(self.model.parameters()).device
            if shap_config.get('use_gpu', True) and not torch.cuda.is_available():
                self.logger.warning("GPU запрошен, но недоступен. Используется CPU.")
            else:
                self.logger.info(f"Используется устройство: {self.device}")
        if self.train_output_only:
            self.logger.info("SHAP fine-tune: train_output_only=true (обновляются только coeffs)")

    @staticmethod
    def _compute_regularizer_lambda(lambda_start, lambda_end, warmup_epochs, progress):
        """Плавный разогрев коэффициента регуляризации по доле обучения."""
        warmup = float(max(warmup_epochs, 0.0))
        if warmup <= 0.0:
            return float(lambda_end)
        if progress <= 0.0:
            return float(lambda_start)
        if progress >= warmup:
            return float(lambda_end)
        ratio = progress / warmup
        return float(lambda_start + (lambda_end - lambda_start) * ratio)

    def _current_gate_probs(self):
        if self.feature_gate_logits is None:
            return None
        gate = torch.sigmoid(self.feature_gate_logits)
        gate = torch.clamp(gate, min=self.gate_min, max=self.gate_max)
        return gate

    def _gate_blend_alpha(self):
        warmup = max(float(self.gate_blend_warmup_epochs), 0.0)
        if warmup <= 0.0 or self.total_epochs is None or self.total_epochs <= 0:
            return 1.0
        progress = float(self.current_epoch + 1) / float(max(self.total_epochs, 1))
        return float(min(1.0, progress / warmup))

    def _apply_feature_gates(self, x_tensor):
        gate = self._current_gate_probs()
        if gate is None:
            return x_tensor
        alpha = self._gate_blend_alpha() if self.model.training else 1.0
        blended = (1.0 - alpha) + alpha * gate
        return x_tensor * blended.unsqueeze(0)

    @staticmethod
    def _js_divergence(p, q, eps=1e-10):
        p = torch.clamp(p, min=eps)
        q = torch.clamp(q, min=eps)
        p = p / torch.sum(p)
        q = q / torch.sum(q)
        m = 0.5 * (p + q)
        kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)))
        kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)))
        return 0.5 * (kl_pm + kl_qm)

    def _compute_gate_distribution(self, gate_probs, batch_x):
        feature_mass = torch.mean(torch.abs(batch_x), dim=0)
        raw = torch.clamp(gate_probs, min=0.0) * torch.clamp(feature_mass, min=1e-10)
        return raw / (torch.sum(raw) + 1e-10)

    @staticmethod
    def _compute_gini(dist):
        x = torch.clamp(dist, min=1e-12)
        x = x / (torch.sum(x) + 1e-12)
        sorted_x, _ = torch.sort(x)
        n = sorted_x.numel()
        idx = torch.arange(1, n + 1, device=sorted_x.device, dtype=sorted_x.dtype)
        # Gini for non-negative distribution
        return torch.sum((2.0 * idx - n - 1.0) * sorted_x) / (n + 1e-12)

    @staticmethod
    def _vector_corr(a, b):
        a = a - torch.mean(a)
        b = b - torch.mean(b)
        denom = torch.sqrt(torch.sum(a * a) * torch.sum(b * b)) + 1e-12
        return torch.sum(a * b) / denom

    def _compute_alignment_loss(self, p, q):
        p = torch.clamp(p, min=1e-10)
        q = torch.clamp(q, min=1e-10)
        p = p / (torch.sum(p) + 1e-10)
        q = q / (torch.sum(q) + 1e-10)
        js = self._js_divergence(p, q)
        mse = torch.mean((p - q) ** 2)
        cosine = 1.0 - F.cosine_similarity(
            p.unsqueeze(0), q.unsqueeze(0), dim=1, eps=1e-8
        ).squeeze(0)
        mode = self.ea_alignment_loss
        a = self.ea_alignment_alpha
        if mode == 'js':
            loss = js
        elif mode == 'mse':
            loss = mse
        elif mode == 'cosine':
            loss = cosine
        elif mode == 'js_mse':
            loss = a * js + (1.0 - a) * mse
        else:  # cosine_mse
            loss = a * cosine + (1.0 - a) * mse
        return loss, js, mse, cosine

    def _compose_internal_importance(self, grad_importance_normalized, gate_probs, batch_x):
        gate_dist = None
        if gate_probs is not None:
            gate_dist = self._compute_gate_distribution(gate_probs, batch_x)
        src = self.ea_importance_source
        if src == 'gate' and gate_dist is not None:
            return gate_dist, gate_dist
        if src == 'mixed' and gate_dist is not None:
            a = self.ea_gate_importance_alpha
            p_mix = a * grad_importance_normalized + (1.0 - a) * gate_dist
            p_mix = p_mix / (torch.sum(p_mix) + 1e-10)
            return p_mix, gate_dist
        return grad_importance_normalized, gate_dist

    def _compute_bottom_invariance_loss(self, batch_x, predictions, q_target, baseline_values):
        if self.ea_bottom_invariance_weight <= 0.0:
            return torch.tensor(0.0, device=self.device)
        n = int(batch_x.shape[1])
        k = int(max(1, min(self.ea_bottom_k, n)))
        idx = torch.argsort(q_target.detach(), descending=False)[:k]
        base = torch.as_tensor(
            np.asarray(baseline_values, dtype=np.float32),
            device=batch_x.device,
            dtype=batch_x.dtype
        )
        x_masked = batch_x.clone()
        x_masked[:, idx] = base[idx].unsqueeze(0)
        pred_masked = torch.nan_to_num(self.model(x_masked))
        ref = predictions.detach()
        if pred_masked.shape != ref.shape:
            min_dim = min(pred_masked.shape[-1], ref.shape[-1])
            pred_masked = pred_masked[..., :min_dim]
            ref = ref[..., :min_dim]
        return F.mse_loss(pred_masked, ref)

    def _reg_warmup_factor(self):
        if self.total_epochs is None or self.total_epochs <= 0:
            return 1.0
        warmup_epochs = int(self.ea_warmup_fraction * self.total_epochs)
        if warmup_epochs <= 0:
            return 1.0
        if self.current_epoch < warmup_epochs:
            return 0.0
        ramp = max(1, warmup_epochs)
        factor = float(self.current_epoch - warmup_epochs + 1) / float(ramp)
        return float(min(max(factor, 0.0), 1.0))

    @staticmethod
    def _compute_rank_loss_hinge_allpairs(p, q, margin=0.01, delta=1e-4):
        p = torch.clamp(p, min=1e-10)
        q = torch.clamp(q, min=1e-10)
        p = p / (torch.sum(p) + 1e-10)
        q = q / (torch.sum(q) + 1e-10)
        diff_q = q.unsqueeze(1) - q.unsqueeze(0)
        pair_mask = diff_q > float(max(delta, 0.0))
        pairs_count = int(torch.count_nonzero(pair_mask).item())
        if pairs_count == 0:
            z = p.sum() * 0.0
            return z, 0.0, 0.0
        diff_p = p.unsqueeze(1) - p.unsqueeze(0)
        losses = torch.relu(float(max(margin, 0.0)) - diff_p)
        violations = int(torch.count_nonzero(losses[pair_mask] > 0).item())
        return torch.mean(losses[pair_mask]), float(pairs_count), float(violations)

    @staticmethod
    def _compute_rank_loss_ranknet_allpairs(p, q, tau=0.2, delta=0.0):
        p = torch.clamp(p, min=1e-10)
        q = torch.clamp(q, min=1e-10)
        p = p / (torch.sum(p) + 1e-10)
        q = q / (torch.sum(q) + 1e-10)
        s = torch.log(p + 1e-10)
        diff_q = q.unsqueeze(1) - q.unsqueeze(0)
        pair_mask = diff_q > float(max(delta, 0.0))
        pairs_count = int(torch.count_nonzero(pair_mask).item())
        if pairs_count == 0:
            z = p.sum() * 0.0
            return z, 0.0, 0.0
        abs_dq = torch.abs(diff_q[pair_mask])
        w = abs_dq / (torch.mean(abs_dq) + 1e-10)
        w = torch.clamp(w, min=0.0, max=1.0)
        tau_safe = float(max(tau, 1e-6))
        diff_s = (s.unsqueeze(1) - s.unsqueeze(0))[pair_mask]
        losses = F.softplus(-diff_s / tau_safe)
        weighted = w * losses
        violations = int(torch.count_nonzero(diff_s <= 0).item())
        return torch.mean(weighted), float(pairs_count), float(violations)

    @staticmethod
    def _compute_rank_loss_topbottom_ranknet(p, q, top_k=3, bottom_k=3, tau=0.2):
        p = torch.clamp(p, min=1e-10)
        q = torch.clamp(q, min=1e-10)
        p = p / (torch.sum(p) + 1e-10)
        q = q / (torch.sum(q) + 1e-10)
        n = int(p.shape[0])
        tk = int(max(1, min(top_k, n)))
        bk = int(max(1, min(bottom_k, n)))
        top_idx = torch.argsort(q, descending=True)[:tk]
        bottom_idx = torch.argsort(q, descending=False)[:bk]
        s = torch.log(p + 1e-10)
        s_top = s[top_idx].unsqueeze(1)  # [tk,1]
        s_bottom = s[bottom_idx].unsqueeze(0)  # [1,bk]
        tau_safe = float(max(tau, 1e-6))
        diff = (s_top - s_bottom) / tau_safe
        losses = F.softplus(-diff)
        violations = int(torch.count_nonzero(diff <= 0).item())
        pairs_count = int(losses.numel())
        return torch.mean(losses), float(pairs_count), float(violations)

    @staticmethod
    def _compute_loss_grad_stats(loss_tensor, params):
        grads = torch.autograd.grad(
            outputs=loss_tensor,
            inputs=params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        sq_sum = None
        used = 0
        for g in grads:
            if g is None:
                continue
            used += 1
            v = torch.sum(g.detach() ** 2)
            sq_sum = v if sq_sum is None else (sq_sum + v)
        if sq_sum is None:
            return 0.0, 0, len(params)
        norm = float(torch.sqrt(sq_sum + 1e-20).item())
        return norm, used, len(params)

    def _compute_error_aware_importance(self, batch_x_raw, batch_y, batch_main_loss, baseline_values, loss_function):
        q_raw, q_ema, diag = error_importance(
            self.model,
            batch_x_raw,
            batch_y,
            loss_fn=loss_function,
            masking_mode=self.error_importance_mode,
            baseline_values=baseline_values,
            gate_fn=self._apply_feature_gates,
            ema_state=self.error_importance_ema,
            ema_beta=self.error_importance_ema_beta,
            prev_q=self.prev_error_importance_raw,
            positive_clipping=self.ea_positive_clipping,
        )
        self.error_importance_ema = q_ema.detach().clone()
        self.prev_error_importance_raw = torch.clamp(q_raw.detach(), min=1e-10)
        self.prev_error_importance_raw = self.prev_error_importance_raw / (torch.sum(self.prev_error_importance_raw) + 1e-10)
        self.last_eta_mean = float(diag.get('eta_mean', float('nan')))
        self.last_eta_std = float(diag.get('eta_std', float('nan')))
        self.last_qerr_entropy = float(diag.get('q_entropy', float('nan')))
        self.last_qerr_gini = float(diag.get('q_gini', float('nan')))
        self.last_qerr_corr = float(diag.get('q_corr', float('nan')))
        return self.error_importance_ema.detach()

    def fit(
        self,
        X_train,
        y_train,
        epochs=25,
        batch_size=32,
        lr=0.005,
        X_val=None,
        y_val=None,
        y_teacher_train=None,
    ):
        """
        Обучение с улучшенной SHAP-регуляризацией
        
        Args:
            X_train: Тренировочные признаки (N, 10)
            y_train: Тренировочные целевые значения (N, 60)
            epochs: Количество эпох
            batch_size: Размер батча
            lr: Скорость обучения
            
        Returns:
            dict: История потерь
        """
        start_time = time.time()
        self._seed_training()
        self.frozen_q_target = None
        
        # Инициализация для адаптивного gamma
        self.total_epochs = epochs
        self.current_epoch = 0
        self.best_main_loss = None
        self.no_improvement_count = 0
        self.error_importance_ema = None
        self.prev_error_importance_raw = None
        self.grad_importance_ema = None

        # Подготовка данных
        X_train_array = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        X_tensor = torch.tensor(X_train_array, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_train_array, dtype=torch.float32, device=self.device)

        X_tensor = torch.nan_to_num(X_tensor)
        y_tensor = torch.nan_to_num(y_tensor)

        if self.use_feature_gates:
            gate_init = np.clip(self.gate_init, 1e-4, 1 - 1e-4)
            gate_logit = np.log(gate_init / (1.0 - gate_init))
            self.feature_gate_logits = torch.nn.Parameter(
                torch.full((X_tensor.shape[1],), float(gate_logit), device=self.device, dtype=torch.float32)
            )
        else:
            self.feature_gate_logits = None
        
        teacher_tensor = None
        if y_teacher_train is not None:
            y_teacher_array = np.array(y_teacher_train) if not isinstance(y_teacher_train, np.ndarray) else y_teacher_train
            y_teacher_array = np.nan_to_num(y_teacher_array, nan=0.0, posinf=0.0, neginf=0.0)
            if y_teacher_array.shape == y_train_array.shape:
                teacher_tensor = torch.tensor(y_teacher_array, dtype=torch.float32, device=self.device)

        if teacher_tensor is not None:
            training_dataset = TensorDataset(X_tensor, y_tensor, teacher_tensor)
        else:
            training_dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=self._make_dataloader_generator()
        )

        # Базовые значения для SHAP
        baseline_values = np.mean(X_train_array, axis=0)
        baseline_values = np.nan_to_num(baseline_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Оптимизатор и функция потерь
        if self.train_output_only and hasattr(self.model, "coeffs"):
            trainable_params = [self.model.coeffs]
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.feature_gate_logits is not None:
            trainable_params = list(trainable_params) + [self.feature_gate_logits]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        loss_function = torch.nn.MSELoss()

        # История потерь
        history = {
            'total_loss': [],
            'main_loss': [],
            'shap_loss': [],
            'shap_loss_normalized': [],
            'tikhonov_loss': [],
            'nonnegativity_loss': [],
            'tikhonov_lambda': [],
            'nonnegativity_lambda': [],
            'shap_scale_factor': [],
            'shap_contribution': [],
            'tikhonov_contribution': [],
            'nonnegativity_contribution': [],
            'regularization_share': [],
            'accuracy_guard_loss': [],
            'accuracy_guard_contribution': [],
            'distill_loss': [],
            'distill_contribution': [],
            'gate_entropy': [],
            'gate_min': [],
            'gate_max': [],
            'gate_anchor_loss': [],
            'gate_anchor_contribution': [],
            'q_err_entropy': [],
            'q_err_gini': [],
            'q_err_corr': [],
            'p_q_corr': [],
            'ea_ratio': [],
            'main_grad_norm': [],
            'ea_grad_norm': [],
            'ea_main_grad_ratio': [],
            'ea_scale': [],
            'main_grad_params': [],
            'ea_grad_params': [],
            'grad_params_total': [],
            'rank_loss_raw': [],
            'bottom_inv_loss_raw': [],
            'rank_pairs_count': [],
            'rank_violations_count': [],
            'rank_grad_norm': [],
            'rank_main_grad_ratio': [],
            'shap_clip_active': [],
            'epoch_accepted': [],
            'lr': [],
        }

        if self.verbose:
            self.logger.info(f"🟠 Начинаю обучение ANFIS с улучшенной SHAP-регуляризацией...")
            self.logger.info(f"   Эпох: {epochs}, Батч: {batch_size}, LR: {lr}")
            if self.use_adaptive_gamma:
                self.logger.info(f"   Адаптивный Gamma: {self.gamma_start} → {self.gamma_end} (warmup: {self.gamma_warmup_epochs*100:.0f}%)")
            else:
                self.logger.info(f"   Gamma: {self.gamma}")
            if self.use_improved_shap:
                self.logger.info(f"   Используется улучшенная SHAP регуляризация (4 компонента)")
            if self.use_convergence_smoothing:
                self.logger.info(f"   Плавная сходимость: включена (patience: {self.convergence_patience})")
            if self.tikhonov_enabled and self.tikhonov_lambda > 0:
                energy_mode = "log-energy" if self.tikhonov_energy_aware else "uniform-index"
                self.logger.info(
                    f"   Тихонов: порядок D{self.tikhonov_order}, "
                    f"lambda={self.tikhonov_lambda_start} → {self.tikhonov_lambda_end}, "
                    f"warmup={self.tikhonov_warmup_epochs*100:.0f}%, mode={energy_mode}"
                )
            if self.nonnegativity_enabled and self.nonnegativity_lambda > 0:
                self.logger.info(
                    f"   Неотрицательность: lambda={self.nonnegativity_lambda_start} → "
                    f"{self.nonnegativity_lambda_end}, warmup={self.nonnegativity_warmup_epochs*100:.0f}%, "
                    f"power={self.nonnegativity_power}, mode={self.nonnegativity_mode}, "
                    f"tolerance={self.nonnegativity_tolerance:.4f}, "
                    f"soft_count_weight={self.nonnegativity_soft_count_weight:.3f}, "
                    f"soft_count_temperature={self.nonnegativity_soft_count_temperature:.4g}"
                )
            self.logger.info(f"   SHAP scalarization: {self.scalarization_mode}")
            if self.autonomous_error_shap:
                self.logger.info(
                    f"   [EA] mode={self.error_importance_mode}, align={self.ea_alignment_loss}, "
                    f"target={self.error_importance_target}, warmup={self.ea_warmup_fraction}, "
                    f"rho={self.error_target_rho}, ema={self.error_importance_ema_beta}, "
                    f"grad_balance={self.ea_use_grad_balance}, rank={self.ea_rank_type}, rank_w={self.ea_rank_weight}, "
                    f"p_src={self.ea_importance_source}, binv_w={self.ea_bottom_invariance_weight}"
                )

        X_val_tensor = None
        y_val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_array = np.array(X_val) if not isinstance(X_val, np.ndarray) else X_val
            y_val_array = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
            X_val_array = np.nan_to_num(X_val_array, nan=0.0, posinf=0.0, neginf=0.0)
            y_val_array = np.nan_to_num(y_val_array, nan=0.0, posinf=0.0, neginf=0.0)
            X_val_tensor = torch.tensor(X_val_array, dtype=torch.float32, device=self.device)
            y_val_tensor = torch.tensor(y_val_array, dtype=torch.float32, device=self.device)

        best_val_loss = float("inf")
        best_state_dict = None
        best_gate_logits = None
        no_improve_epochs = 0
        prev_val_main_loss = None
        frozen_model = None
        if self.accuracy_guard_enabled or self.use_distill_anchor or self.quality_first:
            frozen_model = copy.deepcopy(self.model).to(self.device)
            frozen_model.eval()
            for p in frozen_model.parameters():
                p.requires_grad_(False)

        if X_val_tensor is not None and y_val_tensor is not None:
            with torch.no_grad():
                base_pred = torch.nan_to_num(self.model(self._apply_feature_gates(X_val_tensor)))
                if base_pred.shape != y_val_tensor.shape:
                    min_dim_b = min(base_pred.shape[-1], y_val_tensor.shape[-1])
                    base_pred = base_pred[..., :min_dim_b]
                    base_target = y_val_tensor[..., :min_dim_b]
                else:
                    base_target = y_val_tensor
                base_val_loss = float(loss_function(base_pred, base_target).item())
            prev_val_main_loss = base_val_loss
            best_val_loss = base_val_loss
            if self.restore_best_state:
                best_state_dict = copy.deepcopy(self.model.state_dict())
                if self.feature_gate_logits is not None:
                    best_gate_logits = self.feature_gate_logits.detach().clone()

        coeff_anchor = None
        coeff_anchor_norm = None
        max_coeff_delta = None
        if self.train_output_only and hasattr(self.model, "coeffs"):
            coeff_anchor = self.model.coeffs.detach().clone()
            coeff_anchor_norm = float(torch.norm(coeff_anchor).item())
            max_coeff_delta = self.max_coeff_delta_ratio * max(coeff_anchor_norm, 1e-8)

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_state = None
            if self.quality_first and X_val_tensor is not None and y_val_tensor is not None:
                epoch_start_state = copy.deepcopy(self.model.state_dict())
            epoch_start_gate = None
            if self.quality_first and self.feature_gate_logits is not None:
                epoch_start_gate = self.feature_gate_logits.detach().clone()
            epoch_losses = {
                'total': [],
                'main': [],
                'shap': [],
                'shap_loss_normalized': [],
                'tikhonov': [],
                'nonnegativity': [],
                'tikhonov_lambda': [],
                'nonnegativity_lambda': [],
                'shap_scale_factor': [],
                'shap_contribution': [],
                'tikhonov_contribution': [],
                'nonnegativity_contribution': [],
                'regularization_share': [],
                'accuracy_guard_loss': [],
                'accuracy_guard_contribution': [],
                'distill_loss': [],
                'distill_contribution': [],
                'gate_entropy': [],
                'gate_min': [],
                'gate_max': [],
                'gate_anchor_loss': [],
                'gate_anchor_contribution': [],
                'q_err_entropy': [],
                'q_err_gini': [],
                'q_err_corr': [],
                'p_q_corr': [],
                'ea_ratio': [],
                'main_grad_norm': [],
                'ea_grad_norm': [],
                'ea_main_grad_ratio': [],
                'ea_scale': [],
                'main_grad_params': [],
                'ea_grad_params': [],
                'grad_params_total': [],
                'rank_loss_raw': [],
                'bottom_inv_loss_raw': [],
                'rank_pairs_count': [],
                'rank_violations_count': [],
                'rank_grad_norm': [],
                'rank_main_grad_ratio': [],
                'shap_clip_active': [],
            }
            epoch_shap_components = {
                'consistency': [], 'sparsity': [], 'faithfulness': [], 'stability': []
            }
            epoch_shap_weights = {
                'consistency': [], 'sparsity': [], 'faithfulness': [], 'stability': []
            }
            
            # Вычисляем адаптивный gamma для текущей эпохи
            if self.use_adaptive_gamma and self.total_epochs:
                progress = epoch / self.total_epochs
                warmup_progress = self.gamma_warmup_epochs
                
                if progress < warmup_progress:
                    # Фаза разогрева: gamma увеличивается от gamma_start
                    gamma_ratio = progress / warmup_progress
                    current_gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * gamma_ratio
                else:
                    # После warmup удерживаем gamma на целевом уровне без скачков.
                    current_gamma = self.gamma_end
            else:
                current_gamma = self.gamma

            progress = epoch / self.total_epochs if self.total_epochs else 0.0
            current_tikhonov_lambda = self._compute_regularizer_lambda(
                self.tikhonov_lambda_start,
                self.tikhonov_lambda_end,
                self.tikhonov_warmup_epochs,
                progress,
            )
            current_nonnegativity_lambda = self._compute_regularizer_lambda(
                self.nonnegativity_lambda_start,
                self.nonnegativity_lambda_end,
                self.nonnegativity_warmup_epochs,
                progress,
            )

            for batch in data_loader:
                if len(batch) == 3:
                    batch_X, batch_y, batch_teacher = batch
                else:
                    batch_X, batch_y = batch
                    batch_teacher = None
                batch_X = torch.nan_to_num(batch_X)
                batch_y = torch.nan_to_num(batch_y)
                if batch_teacher is not None:
                    batch_teacher = torch.nan_to_num(batch_teacher)

                optimizer.zero_grad()
                self._last_rank_loss_tensor = None
                self.last_rank_pairs_count = float('nan')
                self.last_rank_violations_count = float('nan')

                # Прямой проход
                self.model.train()
                batch_X.requires_grad_(True)
                batch_input = self._apply_feature_gates(batch_X)
                predictions = self.model(batch_input)
                predictions = torch.nan_to_num(predictions)
                
                # Для мультирегрессии predictions может быть (batch, 60)
                if predictions.shape != batch_y.shape:
                    min_dim = min(predictions.shape[-1], batch_y.shape[-1])
                    predictions = predictions[..., :min_dim]
                    batch_y = batch_y[..., :min_dim]
                
                main_loss = loss_function(predictions, batch_y)
                if self.accuracy_guard_enabled and frozen_model is not None:
                    with torch.no_grad():
                        baseline_pred = frozen_model(self._apply_feature_gates(batch_X.detach()))
                        baseline_pred = torch.nan_to_num(baseline_pred)
                        if baseline_pred.shape != batch_y.shape:
                            min_dim_b = min(baseline_pred.shape[-1], batch_y.shape[-1])
                            baseline_pred = baseline_pred[..., :min_dim_b]
                            baseline_target = batch_y[..., :min_dim_b]
                        else:
                            baseline_target = batch_y
                        baseline_main_loss = loss_function(baseline_pred, baseline_target)
                    guard_threshold = baseline_main_loss * (1.0 + self.accuracy_guard_margin)
                    accuracy_guard_loss = torch.relu(main_loss - guard_threshold)
                else:
                    accuracy_guard_loss = torch.tensor(0.0, device=self.device)

                if batch_teacher is not None:
                    teacher_pred = batch_teacher
                    if teacher_pred.shape != predictions.shape:
                        min_dim_t = min(teacher_pred.shape[-1], predictions.shape[-1])
                        teacher_pred = teacher_pred[..., :min_dim_t]
                        pred_for_distill = predictions[..., :min_dim_t]
                    else:
                        pred_for_distill = predictions
                    distill_loss = loss_function(pred_for_distill, teacher_pred)
                elif self.use_distill_anchor and frozen_model is not None:
                    with torch.no_grad():
                        teacher_pred = torch.nan_to_num(frozen_model(self._apply_feature_gates(batch_X.detach())))
                        if teacher_pred.shape != predictions.shape:
                            min_dim_t = min(teacher_pred.shape[-1], predictions.shape[-1])
                            teacher_pred = teacher_pred[..., :min_dim_t]
                            pred_for_distill = predictions[..., :min_dim_t]
                        else:
                            pred_for_distill = predictions
                    distill_loss = loss_function(pred_for_distill, teacher_pred)
                else:
                    distill_loss = torch.tensor(0.0, device=self.device)

                # Тихоновская регуляризация (гладкость спектра)
                if self.tikhonov_enabled and current_tikhonov_lambda > 0:
                    tikhonov_loss = self._compute_tikhonov_loss(predictions)
                else:
                    tikhonov_loss = torch.tensor(0.0, device=self.device)

                if self.nonnegativity_enabled and current_nonnegativity_lambda > 0:
                    nonnegativity_loss = self._compute_nonnegativity_loss(predictions)
                else:
                    nonnegativity_loss = torch.tensor(0.0, device=self.device)

                # Улучшенная SHAP регуляризация
                if self.use_improved_shap:
                    # Вычисляем main_loss до вызова регуляризации для адаптации
                    main_loss_value = main_loss.detach().item()
                    q_err_target = None
                    if self.autonomous_error_shap:
                        q_source_x = batch_X
                        q_source_y = batch_y
                        q_source_loss = main_loss
                        use_val_target = (
                            self.error_importance_target == 'val'
                            and X_val_tensor is not None
                            and y_val_tensor is not None
                            and X_val_tensor.shape[0] > 0
                        )
                        if use_val_target:
                            val_bs = self.error_importance_val_batch_size
                            if val_bs <= 0:
                                val_bs = int(batch_X.shape[0])
                            val_bs = int(max(1, min(val_bs, int(X_val_tensor.shape[0]))))
                            val_idx = torch.randint(
                                low=0,
                                high=int(X_val_tensor.shape[0]),
                                size=(val_bs,),
                                device=self.device,
                            )
                            q_source_x = X_val_tensor.index_select(0, val_idx)
                            q_source_y = y_val_tensor.index_select(0, val_idx)
                            with torch.no_grad():
                                q_pred = torch.nan_to_num(self.model(self._apply_feature_gates(q_source_x)))
                                if q_pred.shape != q_source_y.shape:
                                    min_dim_q = min(q_pred.shape[-1], q_source_y.shape[-1])
                                    q_pred = q_pred[..., :min_dim_q]
                                    q_source_y = q_source_y[..., :min_dim_q]
                                q_source_loss = loss_function(q_pred, q_source_y)
                        q_err_target = self._compute_error_aware_importance(
                            q_source_x,
                            q_source_y,
                            q_source_loss,
                            baseline_values,
                            loss_function,
                        )
                    gate_probs = self._current_gate_probs()
                    shap_loss_tensor, shap_components = self._compute_improved_shap_regularization(
                        batch_input,
                        baseline_values,
                        predictions,
                        main_loss_value=main_loss_value,
                        q_err_target=q_err_target,
                        gate_probs=gate_probs,
                    )
                else:
                    # Простая SHAP регуляризация (для совместимости)
                    shap_loss_tensor, shap_components = self._compute_simple_shap_regularization(
                        batch_X, baseline_values, predictions
                    )

                # УЛУЧШЕННАЯ АДАПТИВНАЯ ФУНКЦИЯ ПОТЕРЬ ДЛЯ 2 ЗАДАЧ:
                # 1. Основная задача: предсказание спектра (main_loss)
                # 2. Задача интерпретируемости: SHAP регуляризация (shap_loss)
                
                main_loss_detached = main_loss.detach()
                eps = 1e-8
                
                # Обновляем скользящее среднее main loss для стабильности
                if self.main_loss_ema is None:
                    self.main_loss_ema = main_loss_detached.item()
                else:
                    self.main_loss_ema = self.ema_alpha * self.main_loss_ema + (1 - self.ema_alpha) * main_loss_detached.item()
                
                # АДАПТИВНАЯ НОРМАЛИЗАЦИЯ SHAP loss для балансировки задач
                shap_loss_detached = shap_loss_tensor.detach().item()
                
                # Вычисляем коэффициент замедления сходимости
                convergence_slowdown = 1.0
                if self.use_convergence_smoothing and self.best_main_loss is not None:
                    improvement = (self.best_main_loss - main_loss_detached.item()) / (self.best_main_loss + eps)
                    if improvement < 0.001:  # Улучшение меньше 0.1%
                        self.no_improvement_count += 1
                        # Замедляем обучение при отсутствии улучшения
                        convergence_slowdown = 1.0 / (1.0 + self.no_improvement_count * 0.1)
                    else:
                        self.no_improvement_count = 0
                        if main_loss_detached.item() < self.best_main_loss:
                            self.best_main_loss = main_loss_detached.item()
                else:
                    if self.best_main_loss is None or main_loss_detached.item() < self.best_main_loss:
                        self.best_main_loss = main_loss_detached.item()
                
                convergence_slowdown = max(convergence_slowdown, self.min_convergence_slowdown)
                
                # Адаптивная нормализация SHAP loss
                if self.autonomous_error_shap and self.ea_bypass_legacy_normalization:
                    scale_factor = convergence_slowdown
                    shap_loss_normalized = shap_loss_tensor * scale_factor
                else:
                    if shap_loss_detached > eps and self.main_loss_ema > eps:
                        current_ratio = shap_loss_detached / self.main_loss_ema
                        progress = epoch / self.total_epochs if self.total_epochs else 0.5
                        target_ratio_dynamic = self.target_shap_ratio * (0.5 + 0.5 * progress)
                        if current_ratio > target_ratio_dynamic * 2:
                            scale_factor = target_ratio_dynamic / current_ratio
                        elif current_ratio < target_ratio_dynamic / 2:
                            scale_factor = target_ratio_dynamic / current_ratio
                        else:
                            scale_factor = 1.0
                        scale_factor *= convergence_slowdown
                        shap_loss_normalized = shap_loss_tensor * scale_factor
                    else:
                        scale_factor = self.target_shap_ratio / (shap_loss_detached / (self.main_loss_ema + eps) + eps)
                        scale_factor *= convergence_slowdown
                        shap_loss_normalized = shap_loss_tensor * scale_factor

                # АДАПТИВНАЯ ФУНКЦИЯ ПОТЕРЬ: балансировка двух задач
                # Используем адаптивный gamma (может меняться в процессе обучения)
                effective_gamma = current_gamma if self.use_adaptive_gamma else self.gamma
                ea_scale = self.last_ea_scale
                tikhonov_contribution = current_tikhonov_lambda * tikhonov_loss
                nonnegativity_contribution = current_nonnegativity_lambda * nonnegativity_loss
                accuracy_guard_contribution = self.accuracy_guard_weight * accuracy_guard_loss
                if self.feature_gate_logits is not None:
                    gate_probs_anchor = self._current_gate_probs()
                    gate_anchor_loss = torch.mean((gate_probs_anchor - 1.0) ** 2)
                else:
                    gate_anchor_loss = torch.tensor(0.0, device=self.device)

                # Ограничиваем вклад регуляризаторов относительно main loss,
                # чтобы этап SHAP не разрушал базовую точность.
                main_ref = torch.clamp(main_loss_detached, min=eps)
                shap_contribution_raw_pre_warm = effective_gamma * ea_scale * shap_loss_normalized
                max_shap_value = self.max_shap_ratio * main_ref
                max_guard_value = self.max_guard_ratio * main_ref
                max_distill_value = self.max_distill_ratio * main_ref
                max_gate_anchor_value = 0.05 * main_ref
                shap_contribution = torch.clamp(shap_contribution_raw_pre_warm, min=0.0, max=max_shap_value)
                accuracy_guard_contribution = torch.clamp(accuracy_guard_contribution, min=0.0, max=max_guard_value)
                distill_contribution = torch.clamp(
                    self.distill_weight * distill_loss,
                    min=0.0,
                    max=max_distill_value,
                )
                gate_anchor_contribution = torch.clamp(
                    self.gate_anchor_weight * gate_anchor_loss,
                    min=0.0,
                    max=max_gate_anchor_value,
                )
                reg_warmup = self._reg_warmup_factor() if self.autonomous_error_shap else 1.0
                main_grad_norm = float('nan')
                ea_grad_norm = float('nan')
                ea_main_grad_ratio = float('nan')
                rank_grad_norm = float('nan')
                rank_main_grad_ratio = float('nan')
                main_grad_params = float('nan')
                ea_grad_params = float('nan')
                grad_params_total = float('nan')

                need_grad_probe = (
                    (self.debug_grad_norms or self.ea_use_grad_balance)
                    and (len(epoch_losses['main']) % self.grad_norm_interval == 0)
                )
                if need_grad_probe:
                    params_for_grad = [p for p in trainable_params if p.requires_grad]
                    if params_for_grad:
                        main_grad_norm, main_used, total_used = self._compute_loss_grad_stats(main_loss, params_for_grad)
                        ea_grad_norm, ea_used, _ = self._compute_loss_grad_stats(shap_loss_normalized, params_for_grad)
                        ea_main_grad_ratio = ea_grad_norm / (main_grad_norm + 1e-12)
                        if self._last_rank_loss_tensor is not None:
                            rank_grad_norm, _, _ = self._compute_loss_grad_stats(self._last_rank_loss_tensor, params_for_grad)
                            rank_main_grad_ratio = rank_grad_norm / (main_grad_norm + 1e-12)
                        main_grad_params = float(main_used)
                        ea_grad_params = float(ea_used)
                        grad_params_total = float(total_used)

                if self.ea_use_grad_balance and np.isfinite(ea_main_grad_ratio):
                    target = max(self.ea_target_grad_ratio, 1e-12)
                    raw_scale = target / max(ea_main_grad_ratio, 1e-12)
                    ea_scale = float(np.clip(raw_scale, self.ea_scale_min, self.ea_scale_max))
                    self.last_ea_scale = ea_scale

                shap_contribution = effective_gamma * ea_scale * shap_loss_normalized * reg_warmup
                ea_ratio = float((shap_contribution.detach() / (main_loss_detached + eps)).item())
                shap_clip_active = float((shap_contribution_raw_pre_warm.detach() > max_shap_value.detach()).item())

                total_loss = (
                    main_loss
                    + shap_contribution
                    + tikhonov_contribution
                    + nonnegativity_contribution
                    + accuracy_guard_contribution
                    + distill_contribution
                    + gate_anchor_contribution
                )

                # Обратное распространение
                if not torch.isfinite(total_loss):
                    self.logger.warning("⚠️  SHAP: total_loss содержит NaN/Inf. Пропускаю батч.")
                    continue

                total_loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                optimizer.step()
                if (
                    coeff_anchor is not None
                    and max_coeff_delta is not None
                    and hasattr(self.model, "coeffs")
                    and max_coeff_delta > 0.0
                ):
                    with torch.no_grad():
                        delta = self.model.coeffs - coeff_anchor
                        delta_norm = float(torch.norm(delta).item())
                        if np.isfinite(delta_norm) and delta_norm > max_coeff_delta:
                            scale = max_coeff_delta / (delta_norm + 1e-12)
                            self.model.coeffs.copy_(coeff_anchor + delta * scale)

                # Сохранение потерь
                epoch_losses['total'].append(float(total_loss.item()))
                epoch_losses['main'].append(float(main_loss.item()))
                epoch_losses['shap'].append(float(shap_loss_tensor.item()))
                epoch_losses['shap_loss_normalized'].append(float(shap_loss_normalized.item()))
                epoch_losses['tikhonov'].append(float(tikhonov_loss.item()))
                epoch_losses['nonnegativity'].append(float(nonnegativity_loss.item()))
                epoch_losses['tikhonov_lambda'].append(float(current_tikhonov_lambda))
                epoch_losses['nonnegativity_lambda'].append(float(current_nonnegativity_lambda))
                epoch_losses['shap_scale_factor'].append(float(scale_factor))
                epoch_losses['shap_contribution'].append(float(shap_contribution.item()))
                epoch_losses['tikhonov_contribution'].append(float(tikhonov_contribution.item()))
                epoch_losses['nonnegativity_contribution'].append(float(nonnegativity_contribution.item()))
                epoch_losses['accuracy_guard_loss'].append(float(accuracy_guard_loss.item()))
                epoch_losses['accuracy_guard_contribution'].append(float(accuracy_guard_contribution.item()))
                epoch_losses['distill_loss'].append(float(distill_loss.item()))
                epoch_losses['distill_contribution'].append(float(distill_contribution.item()))
                gate_probs_log = self._current_gate_probs()
                if gate_probs_log is not None:
                    gp = torch.clamp(gate_probs_log, min=1e-8, max=1.0)
                    gate_entropy = float((-gp * torch.log(gp)).sum().item())
                    epoch_losses['gate_entropy'].append(gate_entropy)
                    epoch_losses['gate_min'].append(float(torch.min(gate_probs_log).item()))
                    epoch_losses['gate_max'].append(float(torch.max(gate_probs_log).item()))
                else:
                    epoch_losses['gate_entropy'].append(float('nan'))
                    epoch_losses['gate_min'].append(float('nan'))
                    epoch_losses['gate_max'].append(float('nan'))
                epoch_losses['gate_anchor_loss'].append(float(gate_anchor_loss.item()))
                epoch_losses['gate_anchor_contribution'].append(float(gate_anchor_contribution.item()))
                epoch_losses['q_err_entropy'].append(float(self.last_qerr_entropy))
                epoch_losses['q_err_gini'].append(float(self.last_qerr_gini))
                epoch_losses['q_err_corr'].append(float(self.last_qerr_corr))
                epoch_losses['p_q_corr'].append(float(self.last_p_q_corr))
                epoch_losses['ea_ratio'].append(ea_ratio)
                epoch_losses['main_grad_norm'].append(main_grad_norm)
                epoch_losses['ea_grad_norm'].append(ea_grad_norm)
                epoch_losses['ea_main_grad_ratio'].append(ea_main_grad_ratio)
                epoch_losses['rank_grad_norm'].append(rank_grad_norm)
                epoch_losses['rank_main_grad_ratio'].append(rank_main_grad_ratio)
                epoch_losses['ea_scale'].append(float(ea_scale))
                epoch_losses['main_grad_params'].append(main_grad_params)
                epoch_losses['ea_grad_params'].append(ea_grad_params)
                epoch_losses['grad_params_total'].append(grad_params_total)
                epoch_losses['rank_loss_raw'].append(float(shap_components.get('error_rank', float('nan'))))
                epoch_losses['bottom_inv_loss_raw'].append(float(shap_components.get('error_bottom_invariance', float('nan'))))
                epoch_losses['rank_pairs_count'].append(float(shap_components.get('rank_pairs_count', self.last_rank_pairs_count)))
                epoch_losses['rank_violations_count'].append(float(shap_components.get('rank_violations_count', self.last_rank_violations_count)))
                epoch_losses['shap_clip_active'].append(float(shap_clip_active))
                epoch_losses['regularization_share'].append(
                    float(
                        (
                            shap_contribution.item()
                            + tikhonov_contribution.item()
                            + nonnegativity_contribution.item()
                            + accuracy_guard_contribution.item()
                            + distill_contribution.item()
                            + gate_anchor_contribution.item()
                        ) / (abs(total_loss.item()) + eps)
                    )
                )
                
                # Сохраняем адаптивные параметры для анализа
                if 'adaptive_gamma' not in epoch_losses:
                    epoch_losses['adaptive_gamma'] = []
                if 'convergence_slowdown' not in epoch_losses:
                    epoch_losses['convergence_slowdown'] = []
                epoch_losses['adaptive_gamma'].append(float(effective_gamma))
                epoch_losses['convergence_slowdown'].append(float(convergence_slowdown))
                
                # Сохранение компонентов SHAP
                for component_name in self.COMPONENT_NAMES:
                    epoch_shap_components[component_name].append(float(shap_components.get(component_name, 0.0)))
                    epoch_shap_weights[component_name].append(float(shap_components.get(f'weight_{component_name}', 0.0)))

            # Усреднение потерь по эпохе
            history_sources = {
                'total_loss': 'total',
                'main_loss': 'main',
                'shap_loss': 'shap',
                'shap_loss_normalized': 'shap_loss_normalized',
                'tikhonov_loss': 'tikhonov',
                'nonnegativity_loss': 'nonnegativity',
                'tikhonov_lambda': 'tikhonov_lambda',
                'nonnegativity_lambda': 'nonnegativity_lambda',
                'shap_scale_factor': 'shap_scale_factor',
                'shap_contribution': 'shap_contribution',
                'tikhonov_contribution': 'tikhonov_contribution',
                'nonnegativity_contribution': 'nonnegativity_contribution',
                'regularization_share': 'regularization_share',
                'accuracy_guard_loss': 'accuracy_guard_loss',
                'accuracy_guard_contribution': 'accuracy_guard_contribution',
                'distill_loss': 'distill_loss',
                'distill_contribution': 'distill_contribution',
                'gate_entropy': 'gate_entropy',
                'gate_min': 'gate_min',
                'gate_max': 'gate_max',
                'gate_anchor_loss': 'gate_anchor_loss',
                'gate_anchor_contribution': 'gate_anchor_contribution',
                'q_err_entropy': 'q_err_entropy',
                'q_err_gini': 'q_err_gini',
                'q_err_corr': 'q_err_corr',
                'p_q_corr': 'p_q_corr',
                'ea_ratio': 'ea_ratio',
                'main_grad_norm': 'main_grad_norm',
                'ea_grad_norm': 'ea_grad_norm',
                'ea_main_grad_ratio': 'ea_main_grad_ratio',
                'rank_grad_norm': 'rank_grad_norm',
                'rank_main_grad_ratio': 'rank_main_grad_ratio',
                'ea_scale': 'ea_scale',
                'main_grad_params': 'main_grad_params',
                'ea_grad_params': 'ea_grad_params',
                'grad_params_total': 'grad_params_total',
                'rank_loss_raw': 'rank_loss_raw',
                'bottom_inv_loss_raw': 'bottom_inv_loss_raw',
                'rank_pairs_count': 'rank_pairs_count',
                'rank_violations_count': 'rank_violations_count',
                'shap_clip_active': 'shap_clip_active',
            }
            for history_key, loss_key in history_sources.items():
                values = epoch_losses[loss_key]
                if values:
                    arr = np.asarray(values, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    history[history_key].append(float(np.mean(arr)) if arr.size else float('nan'))
                else:
                    history[history_key].append(float('nan'))
            
            # Сохраняем адаптивные параметры
            if 'adaptive_gamma' in epoch_losses:
                if 'adaptive_gamma' not in history:
                    history['adaptive_gamma'] = []
                history['adaptive_gamma'].append(float(np.mean(epoch_losses['adaptive_gamma'])))
            
            if 'convergence_slowdown' in epoch_losses:
                if 'convergence_slowdown' not in history:
                    history['convergence_slowdown'] = []
                history['convergence_slowdown'].append(float(np.mean(epoch_losses['convergence_slowdown'])))
            
            # Добавляем компоненты SHAP в историю
            if self.use_improved_shap:
                for comp_name in epoch_shap_components:
                    history_key = f'shap_{comp_name}'
                    if history_key not in history:
                        history[history_key] = []
                    comp_values = epoch_shap_components[comp_name]
                    history[history_key].append(float(np.mean(comp_values)) if comp_values else float('nan'))
                for comp_name in epoch_shap_weights:
                    history_key = f'shap_weight_{comp_name}'
                    if history_key not in history:
                        history[history_key] = []
                    weight_values = epoch_shap_weights[comp_name]
                    history[history_key].append(float(np.mean(weight_values)) if weight_values else float('nan'))

            # Early stopping по валидационному main-loss
            if X_val_tensor is not None and y_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = torch.nan_to_num(self.model(self._apply_feature_gates(X_val_tensor)))
                    if val_pred.shape != y_val_tensor.shape:
                        min_dim = min(val_pred.shape[-1], y_val_tensor.shape[-1])
                        val_pred = val_pred[..., :min_dim]
                        val_target = y_val_tensor[..., :min_dim]
                    else:
                        val_target = y_val_tensor
                    val_main_loss = float(loss_function(val_pred, val_target).item())
                if 'val_main_loss' not in history:
                    history['val_main_loss'] = []
                history['val_main_loss'].append(val_main_loss)

                epoch_rejected = False
                if (
                    self.quality_first
                    and self.reject_on_val_degrade
                    and prev_val_main_loss is not None
                    and val_main_loss > (prev_val_main_loss + self.quality_tolerance)
                    and epoch_start_state is not None
                ):
                    self.model.load_state_dict(epoch_start_state)
                    if epoch_start_gate is not None and self.feature_gate_logits is not None:
                        with torch.no_grad():
                            self.feature_gate_logits.copy_(epoch_start_gate)
                    for group in optimizer.param_groups:
                        group['lr'] = max(self.min_lr, group['lr'] * self.lr_backoff)
                    epoch_rejected = True
                else:
                    prev_val_main_loss = val_main_loss
                    if val_main_loss < (best_val_loss - self.early_stopping_min_delta):
                        best_val_loss = val_main_loss
                        no_improve_epochs = 0
                        if self.restore_best_state:
                            best_state_dict = copy.deepcopy(self.model.state_dict())
                            if self.feature_gate_logits is not None:
                                best_gate_logits = self.feature_gate_logits.detach().clone()
                    else:
                        no_improve_epochs += 1
                history['epoch_accepted'].append(0.0 if epoch_rejected else 1.0)
                history['lr'].append(float(optimizer.param_groups[0]['lr']))
            else:
                history['epoch_accepted'].append(1.0)
                history['lr'].append(float(optimizer.param_groups[0]['lr']))

            # Прогресс с адаптивными параметрами
            if self.verbose and (epoch + 1) % 5 == 0:
                msg = f"   Эпоха {epoch + 1}/{epochs}: Total: {history['total_loss'][-1]:.6f}, Main: {history['main_loss'][-1]:.6f}, SHAP: {history['shap_loss'][-1]:.6f}"
                if self.tikhonov_enabled and 'tikhonov_loss' in history:
                    msg += f", Tikh: {history['tikhonov_loss'][-1]:.6f}"
                if self.nonnegativity_enabled and 'nonnegativity_loss' in history:
                    msg += f", NonNeg: {history['nonnegativity_loss'][-1]:.6f}"
                if 'shap_contribution' in history and 'tikhonov_contribution' in history:
                    msg += (
                        f" | Contrib(SHAP): {history['shap_contribution'][-1]:.6f}"
                        f", Contrib(Tikh): {history['tikhonov_contribution'][-1]:.6f}"
                    )
                    if 'nonnegativity_contribution' in history:
                        msg += f", Contrib(NonNeg): {history['nonnegativity_contribution'][-1]:.6f}"
                    if 'accuracy_guard_contribution' in history:
                        msg += f", Contrib(Guard): {history['accuracy_guard_contribution'][-1]:.6f}"
                    if 'distill_contribution' in history:
                        msg += f", Contrib(Distill): {history['distill_contribution'][-1]:.6f}"
                if self.use_improved_shap and epoch_shap_components['consistency']:
                    msg += f" [C:{np.mean(epoch_shap_components['consistency']):.4f}, S:{np.mean(epoch_shap_components['sparsity']):.4f}, F:{np.mean(epoch_shap_components['faithfulness']):.4f}, St:{np.mean(epoch_shap_components['stability']):.4f}]"
                if self.use_adaptive_gamma and 'adaptive_gamma' in history:
                    msg += f" | Gamma: {history['adaptive_gamma'][-1]:.4f}"
                if self.autonomous_error_shap:
                    msg += f" | EA-warm: {self._reg_warmup_factor():.2f}"
                if self.use_convergence_smoothing and 'convergence_slowdown' in history:
                    msg += f" | Slowdown: {history['convergence_slowdown'][-1]:.3f}"
                if 'val_main_loss' in history and history['val_main_loss']:
                    msg += f" | ValMain: {history['val_main_loss'][-1]:.6f}"
                if 'q_err_entropy' in history and history['q_err_entropy']:
                    qent = history['q_err_entropy'][-1]
                    qgini = history['q_err_gini'][-1]
                    qcor = history['q_err_corr'][-1]
                    pcorr = history['p_q_corr'][-1]
                    ear = history['ea_ratio'][-1]
                    msg += (
                        f" | qE:{qent:.3f} qG:{qgini:.3f} qCorr:{qcor:.3f} p~q:{pcorr:.3f} "
                        f"eaR:{ear:.3f} etaM:{self.last_eta_mean:.4g} etaS:{self.last_eta_std:.4g}"
                    )
                if 'ea_main_grad_ratio' in history and history['ea_main_grad_ratio']:
                    gr = history['ea_main_grad_ratio'][-1]
                    if np.isfinite(gr):
                        msg += f" | gradR:{gr:.3g}"
                if 'rank_main_grad_ratio' in history and history['rank_main_grad_ratio']:
                    rgr = history['rank_main_grad_ratio'][-1]
                    if np.isfinite(rgr):
                        msg += f" | rankR:{rgr:.3g}"
                if 'ea_scale' in history and history['ea_scale']:
                    sc = history['ea_scale'][-1]
                    if np.isfinite(sc):
                        msg += f" | eaS:{sc:.2g}"
                if 'rank_pairs_count' in history and history['rank_pairs_count']:
                    rp = history['rank_pairs_count'][-1]
                    rv = history['rank_violations_count'][-1] if ('rank_violations_count' in history and history['rank_violations_count']) else float('nan')
                    if np.isfinite(rp):
                        msg += f" | rPairs:{int(rp)}"
                    if np.isfinite(rv):
                        msg += f"/{int(rv)}"
                if 'shap_clip_active' in history and history['shap_clip_active']:
                    sca = history['shap_clip_active'][-1]
                    if np.isfinite(sca):
                        msg += f" | clip:{sca:.2f}"
                if 'ea_grad_params' in history and 'grad_params_total' in history:
                    egp = history['ea_grad_params'][-1]
                    tgp = history['grad_params_total'][-1]
                    if np.isfinite(egp) and np.isfinite(tgp) and tgp > 0:
                        msg += f" | eaP:{int(egp)}/{int(tgp)}"
                if 'epoch_accepted' in history and history['epoch_accepted']:
                    msg += f" | Accept: {int(history['epoch_accepted'][-1])}"
                if 'lr' in history and history['lr']:
                    msg += f" | LR: {history['lr'][-1]:.6g}"
                self.logger.info(msg)

            if (
                X_val_tensor is not None
                and self.early_stopping_patience > 0
                and no_improve_epochs >= self.early_stopping_patience
            ):
                if self.verbose:
                    self.logger.info(
                        f"⏹️ Ранняя остановка SHAP на эпохе {epoch + 1}: "
                        f"нет улучшения val_main_loss {no_improve_epochs} эпох"
                    )
                break

        if self.restore_best_state and best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            if best_gate_logits is not None and self.feature_gate_logits is not None:
                with torch.no_grad():
                    self.feature_gate_logits.copy_(best_gate_logits)

        self.training_time = time.time() - start_time
        if self.verbose:
            self.logger.info(f"✅ Обучение завершено за {self.training_time:.2f} сек")
        
        return history

    def _seed_training(self):
        """Фиксируем генераторы для воспроизводимого SHAP-stage."""
        seed = int(self.random_seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_dataloader_generator(self):
        generator = torch.Generator()
        generator.manual_seed(int(self.random_seed))
        return generator

    def _compute_tikhonov_loss(self, predictions):
        """
        Тихоновская регуляризация гладкости спектра по выходу.
        Использует разности первого (D1) или второго (D2) порядка.
        """
        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)

        n_bins = predictions.shape[1]
        if self.tikhonov_energy_aware:
            diffs = self._compute_energy_aware_tikhonov_diffs(predictions)
        elif self.tikhonov_order == 1:
            if n_bins < 2:
                return torch.tensor(0.0, device=self.device)
            diffs = predictions[:, 1:] - predictions[:, :-1]
        elif self.tikhonov_order == 2:
            if n_bins < 3:
                return torch.tensor(0.0, device=self.device)
            diffs = predictions[:, 2:] - 2.0 * predictions[:, 1:-1] + predictions[:, :-2]
        else:
            raise ValueError(f"Неподдерживаемый порядок Тихонова: {self.tikhonov_order} (ожидается 1 или 2)")

        return torch.mean(diffs ** 2)

    def _compute_energy_aware_tikhonov_diffs(self, predictions):
        n_bins = predictions.shape[1]
        energy_axis = self.energy_axis
        if energy_axis is None or len(energy_axis) != n_bins or np.any(np.asarray(energy_axis) <= 0):
            if self.tikhonov_order == 1:
                return predictions[:, 1:] - predictions[:, :-1]
            if self.tikhonov_order == 2:
                return predictions[:, 2:] - 2.0 * predictions[:, 1:-1] + predictions[:, :-2]
            raise ValueError(f"Неподдерживаемый порядок Тихонова: {self.tikhonov_order} (ожидается 1 или 2)")

        xi = torch.tensor(
            np.log(np.asarray(energy_axis, dtype=np.float64)),
            dtype=predictions.dtype,
            device=predictions.device
        )
        delta_xi = torch.clamp(xi[1:] - xi[:-1], min=1e-12)

        if self.tikhonov_order == 1:
            return (predictions[:, 1:] - predictions[:, :-1]) / delta_xi.unsqueeze(0)
        if self.tikhonov_order == 2:
            if n_bins < 3:
                return torch.zeros((predictions.shape[0], 0), device=predictions.device, dtype=predictions.dtype)
            slopes = (predictions[:, 1:] - predictions[:, :-1]) / delta_xi.unsqueeze(0)
            return slopes[:, 1:] - slopes[:, :-1]
        raise ValueError(f"Неподдерживаемый порядок Тихонова: {self.tikhonov_order} (ожидается 1 или 2)")

    def _compute_nonnegativity_loss(self, predictions):
        negative_part = torch.relu(-predictions)
        if self.nonnegativity_mode in {'mass_softcount', 'hybrid_mass_softcount', 'softcount_mass_ratio'}:
            if self.nonnegativity_power == 1:
                negative_mass = torch.sum(negative_part, dim=1)
                total_mass = torch.sum(torch.abs(predictions), dim=1) + 1e-8
            else:
                negative_mass = torch.sum(negative_part ** self.nonnegativity_power, dim=1)
                total_mass = torch.sum(torch.abs(predictions) ** self.nonnegativity_power, dim=1) + 1e-8
            mass_ratio = negative_mass / total_mass
            temperature = torch.tensor(
                self.nonnegativity_soft_count_temperature,
                device=predictions.device,
                dtype=predictions.dtype,
            )
            soft_fraction = torch.mean(negative_part / (negative_part + temperature), dim=1)
            blend = self.nonnegativity_soft_count_weight
            return torch.mean((1.0 - blend) * mass_ratio + blend * soft_fraction)
        if self.nonnegativity_mode in {'mass_ratio', 'relative_mass', 'margin_mass_ratio', 'excess_mass_ratio'}:
            if self.nonnegativity_power == 1:
                negative_mass = torch.sum(negative_part, dim=1)
                total_mass = torch.sum(torch.abs(predictions), dim=1) + 1e-8
            else:
                negative_mass = torch.sum(negative_part ** self.nonnegativity_power, dim=1)
                total_mass = torch.sum(torch.abs(predictions) ** self.nonnegativity_power, dim=1) + 1e-8
            ratio = negative_mass / total_mass
            if self.nonnegativity_mode in {'margin_mass_ratio', 'excess_mass_ratio'}:
                ratio = torch.relu(ratio - self.nonnegativity_tolerance)
                if self.nonnegativity_power > 1:
                    ratio = ratio ** self.nonnegativity_power
            return torch.mean(ratio)
        if self.nonnegativity_power == 1:
            return torch.mean(negative_part)
        return torch.mean(negative_part ** self.nonnegativity_power)

    def _compute_improved_shap_regularization(
        self,
        batch_X,
        baseline_values,
        predictions,
        main_loss_value=None,
        q_err_target=None,
        gate_probs=None,
    ):
        """
        Вычисляет улучшенную SHAP регуляризацию с 4 компонентами:
        - Consistency: согласованность с настоящими Shapley values
        - Sparsity: разреженность важности признаков
        - Faithfulness: верность объяснений
        - Stability: стабильность объяснений
        
        Args:
            batch_X: Батч признаков
            baseline_values: Baseline значения
            predictions: Предсказания модели
            main_loss_value: Текущее значение main loss (опционально, используется для адаптации)
        """
        batch_size = batch_X.shape[0]
        n_features = batch_X.shape[1]
        
        # Вычисляем gradient-based importance (дифференцируемо!)
        batch_X.requires_grad_(True)
        
        grad_input = torch.autograd.grad(
            outputs=self._scalarize_output_tensor(predictions),
            inputs=batch_X,
            grad_outputs=torch.ones(batch_size, device=self.device, dtype=predictions.dtype),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # [batch_size, n_features]
        
        # Важность признаков через градиенты
        grad_importance = torch.abs(grad_input) * torch.abs(batch_X)
        importance_per_feature = torch.mean(grad_importance, dim=0)  # [n_features]
        
        min_threshold = torch.max(importance_per_feature) * 1e-6
        importance_per_feature = torch.clamp(importance_per_feature, min=min_threshold)
        
        # L1 нормализация
        importance_sum = torch.sum(importance_per_feature) + 1e-10
        grad_importance_normalized = importance_per_feature / importance_sum
        grad_importance_normalized = torch.clamp(grad_importance_normalized, min=1e-10, max=1.0)
        grad_importance_normalized = grad_importance_normalized / (torch.sum(grad_importance_normalized) + 1e-10)
        if self.grad_importance_ema is None:
            self.grad_importance_ema = grad_importance_normalized.detach().clone()
        else:
            self.grad_importance_ema = (
                self.grad_importance_ema_beta * self.grad_importance_ema
                + (1.0 - self.grad_importance_ema_beta) * grad_importance_normalized.detach()
            )
            self.grad_importance_ema = self.grad_importance_ema / (torch.sum(self.grad_importance_ema) + 1e-10)

        if main_loss_value is not None:
            current_main_loss = main_loss_value if isinstance(main_loss_value, float) else main_loss_value.item()
        elif hasattr(self, 'main_loss_ema') and self.main_loss_ema is not None:
            current_main_loss = self.main_loss_ema
        else:
            current_main_loss = 0.05

        # Автономный error-aware режим: согласуем важность с ростом ошибки при маскировании.
        if self.autonomous_error_shap and q_err_target is not None:
            q_err = torch.clamp(q_err_target.to(self.device, dtype=grad_importance_normalized.dtype), min=1e-10)
            q_err = q_err / (torch.sum(q_err) + 1e-10)
            if self.ea_target_ablation_mode == 'random_target':
                q_for_target = torch.rand_like(q_err)
                q_for_target = q_for_target / (torch.sum(q_for_target) + 1e-10)
            elif self.ea_target_ablation_mode == 'shuffled_q_err':
                perm = torch.randperm(q_err.numel(), device=q_err.device)
                q_for_target = q_err[perm]
                q_for_target = q_for_target / (torch.sum(q_for_target) + 1e-10)
            elif self.ea_target_ablation_mode == 'frozen_q_err':
                if self.frozen_q_target is None:
                    self.frozen_q_target = q_err.detach().clone()
                q_for_target = self.frozen_q_target
                q_for_target = q_for_target / (torch.sum(q_for_target) + 1e-10)
            elif self.ea_target_ablation_mode == 'anti_q_err':
                q_for_target = torch.clamp(1.0 - q_err, min=1e-10)
                q_for_target = q_for_target / (torch.sum(q_for_target) + 1e-10)
            elif self.ea_target_ablation_mode == 'uniform_target':
                q_for_target = torch.ones_like(q_err) / max(1, q_err.numel())
            else:
                q_for_target = q_err
            if self.grad_importance_ema is not None and self.error_target_rho < 1.0:
                q_target = (1.0 - self.error_target_rho) * self.grad_importance_ema + self.error_target_rho * q_for_target
                q_target = q_target / (torch.sum(q_target) + 1e-10)
            else:
                q_target = q_for_target

            p_internal, p_gate = self._compose_internal_importance(
                grad_importance_normalized, gate_probs, batch_X
            )
            error_consistency, js_err, mse_err, cosine_err = self._compute_alignment_loss(
                p_internal, q_target
            )
            rank_mode = self.ea_rank_type
            if rank_mode == 'ranknet_allpairs':
                rank_loss, rank_pairs_count, rank_violations_count = self._compute_rank_loss_ranknet_allpairs(
                    p_internal,
                    q_target,
                    tau=self.ea_rank_tau,
                    delta=self.ea_rank_delta,
                )
            elif rank_mode == 'topbottom_ranknet':
                rank_loss, rank_pairs_count, rank_violations_count = self._compute_rank_loss_topbottom_ranknet(
                    p_internal,
                    q_target,
                    top_k=self.ea_rank_top_k,
                    bottom_k=self.ea_rank_bottom_k,
                    tau=self.ea_rank_tau,
                )
            else:
                rank_loss, rank_pairs_count, rank_violations_count = self._compute_rank_loss_hinge_allpairs(
                    p_internal,
                    q_target,
                    margin=self.ea_rank_margin,
                    delta=self.ea_rank_delta,
                )
            self.last_rank_pairs_count = float(rank_pairs_count)
            self.last_rank_violations_count = float(rank_violations_count)
            self._last_rank_loss_tensor = rank_loss
            self.last_p_q_corr = float(self._vector_corr(p_internal, q_target).detach().item())

            if p_gate is not None:
                gate_js = self._js_divergence(p_gate, q_err)
                noise_loss = torch.mean((1.0 - q_err) * torch.clamp(gate_probs, min=0.0, max=1.0))
            else:
                gate_js = torch.tensor(0.0, device=self.device)
                noise_loss = torch.tensor(0.0, device=self.device)
            bottom_inv_loss = self._compute_bottom_invariance_loss(
                batch_X,
                predictions,
                q_target,
                baseline_values,
            )

            if self._component_enabled('stability') and batch_size > 1:
                importance_per_sample = torch.abs(grad_input) * torch.abs(batch_X)
                stability_result = PrecisionOptimizedSHAPRegularization.compute_precision_aware_stability(
                    importance_per_sample,
                    current_main_loss,
                )
                stability_loss = stability_result['stability_loss']
            else:
                stability_loss = torch.tensor(0.0, device=self.device)

            shap_loss_tensor = (
                self.error_js_weight * error_consistency
                + self.ea_rank_weight * rank_loss
                + self.ea_bottom_invariance_weight * bottom_inv_loss
                + self.gate_js_weight * gate_js
                + self.noise_weight * noise_loss
                + self.rule_stability_weight * stability_loss
            ).requires_grad_(True)

            shap_components = {
                'consistency': float(error_consistency.detach().item()),
                'sparsity': float(noise_loss.detach().item()),
                'faithfulness': float(gate_js.detach().item()),
                'stability': float(stability_loss.detach().item()),
                'error_js': float(js_err.detach().item()),
                'error_mse': float(mse_err.detach().item()),
                'error_cosine': float(cosine_err.detach().item()),
                'error_rank': float(rank_loss.detach().item()),
                'error_bottom_invariance': float(bottom_inv_loss.detach().item()),
                'rank_pairs_count': float(rank_pairs_count),
                'rank_violations_count': float(rank_violations_count),
                'q_err_entropy': float((-q_err * torch.log(q_err)).sum().detach().item()),
                'q_target_entropy': float((-q_target * torch.log(q_target)).sum().detach().item()),
                'p_q_corr': float(self.last_p_q_corr),
                'importance_source': self.ea_importance_source,
                'weight_consistency': self.error_js_weight,
                'weight_rank': self.ea_rank_weight,
                'weight_bottom_invariance': self.ea_bottom_invariance_weight,
                'rank_type': rank_mode,
                'weight_sparsity': self.noise_weight,
                'weight_faithfulness': self.gate_js_weight,
                'weight_stability': self.rule_stability_weight,
                'ea_target_ablation_mode': self.ea_target_ablation_mode,
                'ea_positive_clipping': bool(self.ea_positive_clipping),
            }
            return shap_loss_tensor, shap_components
        
        # 1. CONSISTENCY: согласованность с настоящими Shapley values (МАТЕМАТИЧЕСКИ УЛУЧШЕНО)
        consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        mse_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        js_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Получаем текущий main_loss для адаптации
        # current_main_loss уже определен выше
        
        if self._component_enabled('consistency') and self.use_true_shap:
            self.true_shap_batch_count += 1
            
            update_frequency = self.true_shap_update_frequency
            
            if (self.true_shap_importance is None or 
                self.true_shap_batch_count % update_frequency == 0):
                
                mean_sample = torch.mean(batch_X, dim=0).detach().cpu().numpy()
                true_shap = self._calculate_shap_approximation(mean_sample, baseline_values)
                true_shap = np.nan_to_num(true_shap, nan=0.0, posinf=0.0, neginf=0.0)
                
                if true_shap.ndim != 1 or true_shap.size == 0:
                    true_shap_normalized = np.ones(n_features) / n_features
                else:
                    shap_sum = float(np.sum(true_shap))
                    if shap_sum <= 1e-12 or not np.isfinite(shap_sum):
                        true_shap_normalized = np.ones(n_features) / n_features
                    else:
                        true_shap_normalized = true_shap / shap_sum
                
                self.true_shap_importance = torch.tensor(
                    true_shap_normalized, 
                    device=self.device, 
                    dtype=torch.float32
                )
            
            if self.true_shap_importance is not None:
                true_shap_positive = torch.clamp(self.true_shap_importance, min=0.0)
                true_shap_sum = torch.sum(true_shap_positive) + 1e-10
                
                if true_shap_sum <= 1e-10:
                    true_shap_normalized = torch.ones_like(true_shap_positive) / n_features
                else:
                    true_shap_normalized = true_shap_positive / true_shap_sum
                
                true_shap_normalized = torch.clamp(true_shap_normalized, min=1e-10, max=1.0)
                
                # Используем улучшенную математическую формулу
                consistency_result = PrecisionOptimizedSHAPRegularization.compute_precision_aware_consistency(
                    grad_importance_normalized,
                    true_shap_normalized,
                    current_main_loss,
                    use_adaptive=True
                )
                
                consistency_loss = consistency_result['consistency_loss']
                mse_loss = consistency_result['mse_loss']
                js_loss = consistency_result['js_loss']
        
        # 2. SPARSITY: разреженность важности признаков (МАТЕМАТИЧЕСКИ УЛУЧШЕНО)
        # Адаптивная формула, которая не мешает точности модели
        # current_main_loss уже определен выше
        
        if self._component_enabled('sparsity'):
            # Используем улучшенную математическую формулу с адаптацией к точности
            sparsity_result = PrecisionOptimizedSHAPRegularization.compute_precision_aware_sparsity(
                grad_importance_normalized,
                current_main_loss,
                target_gini=self.target_gini if self.use_gini_sparsity else 0.4,
                precision_weight=0.7
            )
            
            sparsity_loss = sparsity_result['sparsity_loss']
            gini_coefficient = sparsity_result['gini_coefficient']
            entropy_loss = sparsity_result['entropy']
            gini_loss = sparsity_result.get('gini_loss', torch.tensor(0.0, device=self.device))
        else:
            sparsity_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            gini_coefficient = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            gini_loss = torch.tensor(0.0, device=self.device)
        
        # 3. FAITHFULNESS: верность объяснений (МАТЕМАТИЧЕСКИ УЛУЧШЕНО)
        if self._component_enabled('faithfulness'):
            baseline_tensor = torch.zeros(n_features, device=self.device, dtype=torch.float32, requires_grad=False)
            baseline_X = baseline_tensor.unsqueeze(0).expand(batch_size, -1)
            
            baseline_X.requires_grad_(True)
            baseline_pred = self.model(baseline_X)
            
            faithfulness_result = PrecisionOptimizedSHAPRegularization.compute_precision_aware_faithfulness(
                batch_X,
                baseline_X,
                predictions,
                baseline_pred,
                self.model,
                current_main_loss,
                order=1,
                scalarize_fn=self._scalarize_output_tensor,
            )
            
            faithfulness_loss = faithfulness_result['faithfulness_loss']
        else:
            faithfulness_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 4. STABILITY: стабильность объяснений (МАТЕМАТИЧЕСКИ УЛУЧШЕНО)
        if self._component_enabled('stability') and batch_size > 1:
            importance_per_sample = torch.abs(grad_input) * torch.abs(batch_X)
            
            # Используем улучшенную математическую формулу
            stability_result = PrecisionOptimizedSHAPRegularization.compute_precision_aware_stability(
                importance_per_sample,
                current_main_loss
            )
            
            stability_loss = stability_result['stability_loss']
        else:
            stability_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # УЛУЧШЕННАЯ КОМБИНАЦИЯ КОМПОНЕНТОВ с математически обоснованными адаптивными весами
        # Используем формулу, оптимизированную для максимальной точности
        component_weights_used = None
        if self.use_true_shap and self.true_shap_importance is not None:
            if self.use_adaptive_weights:
                # Используем математически обоснованную формулу адаптивных весов
                weights_result = PrecisionOptimizedSHAPRegularization.compute_adaptive_component_weights(
                    current_main_loss,
                    consistency_loss.detach().item(),
                    sparsity_loss.detach().item(),
                    faithfulness_loss.detach().item(),
                    stability_loss.detach().item(),
                    target_main_loss=0.02
                )
                
                adaptive_weights = self._normalize_component_weights(weights_result['weights'])
                
                # Сохраняем веса для анализа
                self.component_weights_history.append(adaptive_weights)
                component_weights_used = adaptive_weights
                
                shap_loss_tensor = (
                    adaptive_weights['consistency'] * consistency_loss +
                    adaptive_weights['sparsity'] * sparsity_loss +
                    adaptive_weights['faithfulness'] * faithfulness_loss +
                    adaptive_weights['stability'] * stability_loss
                )
            else:
                # Фиксированные веса берутся из конфигурации и нормализуются до суммы 1.
                component_weights_used = self.fixed_component_weights
                shap_loss_tensor = (
                    self.fixed_component_weights['consistency'] * consistency_loss +
                    self.fixed_component_weights['sparsity'] * sparsity_loss +
                    self.fixed_component_weights['faithfulness'] * faithfulness_loss +
                    self.fixed_component_weights['stability'] * stability_loss
                )
        else:
            # Если нет true_shap, используем только активные sparsity/stability компоненты.
            component_weights_used = self.fallback_component_weights
            shap_loss_tensor = (
                self.fallback_component_weights['sparsity'] * sparsity_loss +
                self.fallback_component_weights['stability'] * stability_loss
            )
        
        shap_loss_tensor = shap_loss_tensor.requires_grad_(True)
        
        shap_components = {
            'sparsity': sparsity_loss.detach().item(),
            'faithfulness': faithfulness_loss.detach().item(),
            'stability': stability_loss.detach().item()
        }
        
        if self.use_gini_sparsity:
            shap_components['gini_coefficient'] = gini_coefficient.detach().item() if 'gini_coefficient' in locals() else 0.0
            shap_components['gini_loss'] = gini_loss.detach().item() if 'gini_loss' in locals() else 0.0
            shap_components['entropy_loss'] = entropy_loss.detach().item()
        
        if self.use_true_shap and self.true_shap_importance is not None:
            shap_components['consistency'] = consistency_loss.detach().item()
            shap_components['mse'] = mse_loss.detach().item()
            shap_components['js'] = js_loss.detach().item()
        else:
            shap_components['consistency'] = 0.0
        
        # Добавляем именно те веса компонентов, которые использовались для этого батча.
        if component_weights_used is None:
            component_weights_used = self.fixed_component_weights
        shap_components['weight_consistency'] = component_weights_used.get('consistency', 0.0)
        shap_components['weight_sparsity'] = component_weights_used.get('sparsity', 0.0)
        shap_components['weight_faithfulness'] = component_weights_used.get('faithfulness', 0.0)
        shap_components['weight_stability'] = component_weights_used.get('stability', 0.0)
        
        return shap_loss_tensor, shap_components

    def _normalize_component_weights(self, weights):
        cleaned = {}
        for key in self.COMPONENT_NAMES:
            value = weights.get(key, 0.0)
            try:
                cleaned[key] = max(float(value), 0.0) if self._component_enabled(key) else 0.0
            except (TypeError, ValueError):
                cleaned[key] = 0.0

        total = sum(cleaned.values())
        if total <= 0:
            return {key: 0.0 for key in self.COMPONENT_NAMES}

        return {key: value / total for key, value in cleaned.items()}

    def _component_enabled(self, component_name):
        return component_name in self.active_components

    def _parse_active_components(self, raw_value):
        if raw_value is None:
            return set(self.COMPONENT_NAMES)

        if isinstance(raw_value, str):
            raw_items = [item.strip() for item in raw_value.split(',')]
        elif isinstance(raw_value, (list, tuple, set)):
            raw_items = [str(item).strip() for item in raw_value]
        else:
            return set(self.COMPONENT_NAMES)

        parsed = {item for item in raw_items if item in self.COMPONENT_NAMES}
        return parsed if parsed else set(self.COMPONENT_NAMES)

    def _compute_simple_shap_regularization(self, batch_X, baseline_values, predictions):
        """Простая SHAP регуляризация (для совместимости)"""
        batch_size = batch_X.shape[0]
        n_features = batch_X.shape[1]
        
        batch_X.requires_grad_(True)
        grad_input = torch.autograd.grad(
            outputs=self._scalarize_output_tensor(predictions),
            inputs=batch_X,
            grad_outputs=torch.ones(batch_size, device=self.device, dtype=predictions.dtype),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        grad_importance = torch.abs(grad_input) * torch.abs(batch_X)
        importance_per_feature = torch.mean(grad_importance, dim=0)
        
        min_threshold = torch.max(importance_per_feature) * 1e-6
        importance_per_feature = torch.clamp(importance_per_feature, min=min_threshold)
        
        importance_sum = torch.sum(importance_per_feature) + 1e-10
        shap_normalized = importance_per_feature / importance_sum
        
        target_uniform = torch.ones_like(shap_normalized) / n_features
        shap_loss = torch.mean((shap_normalized - target_uniform) ** 2)
        
        shap_loss_tensor = shap_loss.requires_grad_(True)
        shap_components = {'simple': shap_loss.detach().item()}
        
        return shap_loss_tensor, shap_components

    def predict(self, X_test):
        """Получение предсказаний"""
        self.model.eval()
        with torch.no_grad():
            X_test_array = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            X_test_array = np.asarray(X_test_array, dtype=np.float32)
            if X_test_array.ndim == 1:
                X_test_array = X_test_array.reshape(1, -1)
            if X_test_array.size == 0:
                return np.empty((0, 0), dtype=np.float32)
            X_test_array = np.nan_to_num(X_test_array, nan=0.0, posinf=0.0, neginf=0.0)
            X_tensor = torch.tensor(X_test_array, dtype=torch.float32, device=self.device)
            predictions = self.model(self._apply_feature_gates(X_tensor)).cpu().numpy()
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
            return predictions

    def get_global_shap_importance(self, X_sample):
        """Глобальная важность признаков"""
        X_sample_array = np.array(X_sample) if not isinstance(X_sample, np.ndarray) else X_sample
        X_sample_array = np.asarray(X_sample_array, dtype=np.float32)
        if X_sample_array.ndim == 1:
            X_sample_array = X_sample_array.reshape(1, -1)
        if X_sample_array.size == 0:
            return np.empty((0,), dtype=float)
        X_sample_array = np.nan_to_num(X_sample_array, nan=0.0, posinf=0.0, neginf=0.0)
        baseline_values = np.mean(X_sample_array, axis=0)
        shap_values = self._calculate_shap_approximation(X_sample_array, baseline_values)
        return self._normalize_global_importance(shap_values)

    @staticmethod
    def _normalize_global_importance(shap_values):
        shap_values = np.asarray(shap_values, dtype=float).reshape(-1)
        if shap_values.size == 0:
            return shap_values
        shap_values = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)
        shap_values = np.maximum(shap_values, 0.0)
        total = float(np.sum(shap_values))
        if not np.isfinite(total) or total <= 1e-12:
            return np.full(shap_values.shape, 1.0 / shap_values.size, dtype=float)
        return shap_values / total

    def _calculate_shap_approximation(self, X_batch, baseline):
        """Приближенные SHAP значения"""
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X_batch, torch.Tensor):
                X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=self.device)
            else:
                X_tensor = X_batch.to(self.device)

            if X_tensor.ndim == 1:
                X_tensor = X_tensor.unsqueeze(0)
            
            original_predictions = self._scalarize_output_tensor(
                self.model(self._apply_feature_gates(X_tensor))
            ).detach().cpu().numpy()

            shap_values = []
            X_numpy = X_tensor.cpu().numpy()

            for feature_index in range(X_numpy.shape[1]):
                X_masked = X_numpy.copy()
                X_masked[:, feature_index] = baseline[feature_index]

                X_masked_tensor = torch.tensor(X_masked, dtype=torch.float32, device=self.device)
                masked_predictions = self._scalarize_output_tensor(
                    self.model(self._apply_feature_gates(X_masked_tensor))
                ).detach().cpu().numpy()

                if np.isscalar(original_predictions) and np.isscalar(masked_predictions):
                    feature_importance = abs(float(original_predictions) - float(masked_predictions))
                else:
                    feature_importance = float(np.mean(np.abs(original_predictions - masked_predictions)))

                shap_values.append(feature_importance)

        return np.asarray(shap_values, dtype=float)

    @staticmethod
    def _resolve_energy_axis(target_count):
        n_bins = int(target_count) if target_count else 0
        if n_bins <= 0:
            return None
        axis = np.asarray(Ebins_float_IAEA_Comp, dtype=float)
        if axis.size == n_bins + 1:
            axis = axis[:-1]
        elif axis.size != n_bins:
            return None
        axis = np.nan_to_num(axis, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(axis <= 0):
            return None
        return axis

    @staticmethod
    def _parse_band_slices(raw_bands, target_count):
        n_bins = int(target_count) if target_count else 0
        default = [(0, 20), (20, 40), (40, 60)] if n_bins <= 0 or n_bins == 60 else [(0, n_bins)]
        if not raw_bands:
            return default

        parsed = []
        for band in raw_bands:
            if not isinstance(band, (list, tuple)) or len(band) != 2:
                continue
            start, stop = int(band[0]), int(band[1])
            if n_bins > 0:
                start = max(start, 0)
                stop = min(stop, n_bins)
            if stop > start:
                parsed.append((start, stop))
        return parsed or default

    @staticmethod
    def _parse_band_weights(raw_weights, n_bands):
        if n_bands <= 0:
            return np.asarray([], dtype=float)
        if raw_weights is None:
            return np.full(n_bands, 1.0 / n_bands, dtype=float)
        weights = np.asarray(raw_weights, dtype=float).reshape(-1)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.maximum(weights, 0.0)
        if weights.size != n_bands or np.sum(weights) <= 0:
            return np.full(n_bands, 1.0 / n_bands, dtype=float)
        return weights / np.sum(weights)

    def _scalarize_output_tensor(self, predictions):
        if predictions.ndim == 1:
            return predictions
        if predictions.ndim != 2:
            return torch.mean(predictions.view(predictions.shape[0], -1), dim=1)

        if self.scalarization_mode == 'band_weighted':
            band_means = []
            valid_weights = []
            for idx, (start, stop) in enumerate(self.scalarization_bands):
                start = max(int(start), 0)
                stop = min(int(stop), predictions.shape[1])
                if stop <= start:
                    continue
                band_means.append(torch.mean(predictions[:, start:stop], dim=1))
                valid_weights.append(float(self.scalarization_weights[idx]))
            if band_means and np.sum(valid_weights) > 0:
                weights = torch.tensor(
                    np.asarray(valid_weights, dtype=np.float32) / np.sum(valid_weights),
                    device=predictions.device,
                    dtype=predictions.dtype,
                )
                stacked = torch.stack(band_means, dim=1)
                return torch.sum(stacked * weights.unsqueeze(0), dim=1)

        if self.scalarization_mode == 'sum':
            return torch.sum(predictions, dim=1)

        return torch.mean(predictions, dim=1)
