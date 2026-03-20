"""
Улучшенный SHAP-регуляризованный тренер ANFIS для восстановления спектра нейтронов
Использует улучшенную SHAP регуляризацию с 4 компонентами: Consistency, Sparsity, Faithfulness, Stability
Работает в двухэтапном режиме: сначала vanilla ANFIS, потом SHAP регуляризация
"""

import random
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from constants import Ebins_float_IAEA_Comp
from src.models.shap_trainer_precision_optimized import PrecisionOptimizedSHAPRegularization
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

    def fit(self, X_train, y_train, epochs=25, batch_size=32, lr=0.005):
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
        
        # Инициализация для адаптивного gamma
        self.total_epochs = epochs
        self.current_epoch = 0
        self.best_main_loss = None
        self.no_improvement_count = 0

        # Подготовка данных
        X_train_array = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        X_tensor = torch.tensor(X_train_array, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_train_array, dtype=torch.float32, device=self.device)

        X_tensor = torch.nan_to_num(X_tensor)
        y_tensor = torch.nan_to_num(y_tensor)
        
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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

        for epoch in range(epochs):
            self.current_epoch = epoch
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

            for batch_X, batch_y in data_loader:
                batch_X = torch.nan_to_num(batch_X)
                batch_y = torch.nan_to_num(batch_y)

                optimizer.zero_grad()

                # Прямой проход
                self.model.train()
                batch_X.requires_grad_(True)
                predictions = self.model(batch_X)
                predictions = torch.nan_to_num(predictions)
                
                # Для мультирегрессии predictions может быть (batch, 60)
                if predictions.shape != batch_y.shape:
                    min_dim = min(predictions.shape[-1], batch_y.shape[-1])
                    predictions = predictions[..., :min_dim]
                    batch_y = batch_y[..., :min_dim]
                
                main_loss = loss_function(predictions, batch_y)

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
                    shap_loss_tensor, shap_components = self._compute_improved_shap_regularization(
                        batch_X, baseline_values, predictions, main_loss_value=main_loss_value
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
                if shap_loss_detached > eps and self.main_loss_ema > eps:
                    # Вычисляем текущее соотношение
                    current_ratio = shap_loss_detached / self.main_loss_ema
                    
                    # Плавная нормализация с учетом прогресса обучения
                    progress = epoch / self.total_epochs if self.total_epochs else 0.5
                    
                    # На ранних этапах: меньше влияния SHAP (больше focus на основную задачу)
                    # На поздних этапах: больше влияния SHAP (больше focus на интерпретируемость)
                    target_ratio_dynamic = self.target_shap_ratio * (0.5 + 0.5 * progress)
                    
                    # Если соотношение слишком большое или маленькое, нормализуем
                    if current_ratio > target_ratio_dynamic * 2:
                        scale_factor = target_ratio_dynamic / current_ratio
                    elif current_ratio < target_ratio_dynamic / 2:
                        scale_factor = target_ratio_dynamic / current_ratio
                    else:
                        scale_factor = 1.0
                    
                    # Применяем замедление сходимости
                    scale_factor *= convergence_slowdown
                    shap_loss_normalized = shap_loss_tensor * scale_factor
                else:
                    # Fallback: используем простую нормализацию
                    scale_factor = self.target_shap_ratio / (shap_loss_detached / (self.main_loss_ema + eps) + eps)
                    scale_factor *= convergence_slowdown
                    shap_loss_normalized = shap_loss_tensor * scale_factor

                # АДАПТИВНАЯ ФУНКЦИЯ ПОТЕРЬ: балансировка двух задач
                # Используем адаптивный gamma (может меняться в процессе обучения)
                effective_gamma = current_gamma if self.use_adaptive_gamma else self.gamma
                shap_contribution = effective_gamma * shap_loss_normalized
                tikhonov_contribution = current_tikhonov_lambda * tikhonov_loss
                nonnegativity_contribution = current_nonnegativity_lambda * nonnegativity_loss
                total_loss = main_loss + shap_contribution + tikhonov_contribution + nonnegativity_contribution

                # Обратное распространение
                if not torch.isfinite(total_loss):
                    self.logger.warning("⚠️  SHAP: total_loss содержит NaN/Inf. Пропускаю батч.")
                    continue

                total_loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                optimizer.step()

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
                epoch_losses['regularization_share'].append(
                    float(
                        (
                            shap_contribution.item()
                            + tikhonov_contribution.item()
                            + nonnegativity_contribution.item()
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
            }
            for history_key, loss_key in history_sources.items():
                values = epoch_losses[loss_key]
                history[history_key].append(float(np.mean(values)) if values else float('nan'))
            
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
                if self.use_improved_shap and epoch_shap_components['consistency']:
                    msg += f" [C:{np.mean(epoch_shap_components['consistency']):.4f}, S:{np.mean(epoch_shap_components['sparsity']):.4f}, F:{np.mean(epoch_shap_components['faithfulness']):.4f}, St:{np.mean(epoch_shap_components['stability']):.4f}]"
                if self.use_adaptive_gamma and 'adaptive_gamma' in history:
                    msg += f" | Gamma: {history['adaptive_gamma'][-1]:.4f}"
                if self.use_convergence_smoothing and 'convergence_slowdown' in history:
                    msg += f" | Slowdown: {history['convergence_slowdown'][-1]:.3f}"
                self.logger.info(msg)

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

    def _compute_improved_shap_regularization(self, batch_X, baseline_values, predictions, main_loss_value=None):
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
        
        # 1. CONSISTENCY: согласованность с настоящими Shapley values (МАТЕМАТИЧЕСКИ УЛУЧШЕНО)
        consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        mse_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        js_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Получаем текущий main_loss для адаптации
        if main_loss_value is not None:
            current_main_loss = main_loss_value if isinstance(main_loss_value, float) else main_loss_value.item()
        elif hasattr(self, 'main_loss_ema') and self.main_loss_ema is not None:
            current_main_loss = self.main_loss_ema
        else:
            current_main_loss = 0.05  # Значение по умолчанию
        
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
            predictions = self.model(X_tensor).cpu().numpy()
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
                self.model(X_tensor)
            ).detach().cpu().numpy()

            shap_values = []
            X_numpy = X_tensor.cpu().numpy()

            for feature_index in range(X_numpy.shape[1]):
                X_masked = X_numpy.copy()
                X_masked[:, feature_index] = baseline[feature_index]

                X_masked_tensor = torch.tensor(X_masked, dtype=torch.float32, device=self.device)
                masked_predictions = self._scalarize_output_tensor(
                    self.model(X_masked_tensor)
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
