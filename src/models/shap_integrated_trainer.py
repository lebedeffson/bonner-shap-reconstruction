"""
ANFIS тренер с встроенной SHAP-регуляризацией в loss функцию
Реализует обучение с объяснениями (XAI 2.0) - SHAP регуляризация интегрирована в процесс обучения
"""

import time
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import factorial
from itertools import combinations

from src.models.shap_trainer_improved import ShapAwareANFISTrainerImproved
from src.utils.logger import get_logger


class ShapIntegratedANFISTrainer(ShapAwareANFISTrainerImproved):
    """
    Тренер ANFIS с встроенной SHAP-регуляризацией в loss функцию во время обучения
    
    Отличие от ShapAwareANFISTrainer:
    - Обучает модель С НУЛЯ с SHAP регуляризацией в loss функции
    - Не требует предварительного обучения vanilla ANFIS
    - SHAP регуляризация интегрирована в каждый шаг обучения
    
    Реализует концепцию XAI 2.0: обучение с объяснениями
    """
    
    def __init__(self, model, config, gamma=0.5, verbose=True):
        """
        Инициализация тренера
        
        Args:
            model: ANFIS модель (BioAnfisRegressor) - должна быть создана, но не обучена
            config: Конфигурация
            gamma: Коэффициент SHAP-регуляризации
            verbose: Выводить ли информацию
        """
        # Сохраняем ссылку на обертку ДО вызова super().__init__, 
        # так как родительский класс может заменить self.model на self.model.network
        anfis_wrapper = None
        if hasattr(model, 'fit'):
            anfis_wrapper = model
            
        super().__init__(model, config, gamma, verbose)
        
        # Восстанавливаем ссылку на обертку
        if anfis_wrapper is not None:
            self.anfis_wrapper = anfis_wrapper
        
        # GPU уже настроен в родительском классе через _get_device()
        # Модель уже перемещена на GPU в родительском классе
        
        # Дополнительные параметры для интегрированного обучения
        shap_config = config.get('shap_reg', {})
        self.use_pso_init = shap_config.get('use_pso_init', True)  # Использовать PSO для начальной инициализации
        self.pso_epochs = shap_config.get('pso_epochs', 5)  # Количество эпох PSO для инициализации
        
        # Параметры для вычисления Shapley values (из родительского класса)
        self.shap_exact = shap_config.get('shap_exact', True)  # Использовать точное вычисление для n <= 10
        self.shap_n_samples = shap_config.get('shap_n_samples', 100)  # Количество семплов для Monte Carlo
        
        # Дополнительные параметры из родительского класса
        self.negative_penalty = float(shap_config.get('negative_penalty', 0.1))  # Штраф за отрицательные предсказания
        
        # Параметры для использования настоящих Shapley values
        self.use_true_shap = shap_config.get('use_true_shap', True)  # Использовать настоящие Shapley values
        self.true_shap_update_frequency = shap_config.get('true_shap_update_frequency', 10)  # Обновлять каждые N батчей
        self.true_shap_batch_count = 0  # Счетчик батчей для обновления настоящих Shapley values
        self.true_shap_importance = None  # Кэш настоящих Shapley values (глобальная важность)
        
        # Параметры улучшенной SHAP регуляризации
        self.use_improved_shap = shap_config.get('use_improved_shap', True)  # Использовать улучшенную регуляризацию
        self.gamma_sparsity = shap_config.get('gamma_sparsity', 0.8)  # Коэффициент для sparsity регуляризации (увеличено с 0.3)
        self.gamma_consistency = shap_config.get('gamma_consistency', 0.3)  # Коэффициент для consistency регуляризации (увеличено с 0.2)
        self.gamma_stability = shap_config.get('gamma_stability', 0.2)  # Коэффициент для stability регуляризации (увеличено с 0.1)
        self.log_shap_components = shap_config.get('log_shap_components', True)  # Логировать компоненты регуляризации
        
        # Глобальная важность признаков (для consistency регуляризации)
        self.global_shap_importance = None
        self.global_shap_update_frequency = shap_config.get('global_shap_update_frequency', 10)  # Обновлять каждые N батчей
        self.global_shap_batch_count = 0
        
        # Логгер для интегрированного обучения
        self.logger = get_logger("anfis_shap.integrated_trainer")
        if not verbose:
            self.logger.setLevel(30)  # WARNING level
        
    def fit_from_scratch(self, X_train, y_train, epochs=50, batch_size=32, lr=0.005, 
                         X_val=None, y_val=None):
        """
        Обучение модели С НУЛЯ с встроенной SHAP-регуляризацией
        
        Args:
            X_train: Тренировочные признаки (N, 10)
            y_train: Тренировочные целевые значения (N, 60)
            epochs: Количество эпох градиентного обучения
            batch_size: Размер батча
            lr: Скорость обучения
            X_val: Валидационные признаки (опционально)
            y_val: Валидационные целевые значения (опционально)
            
        Returns:
            dict: История потерь и метрики
        """
        start_time = time.time()
        
        # Подготовка данных
        X_train_array = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        # Перемещаем данные на GPU если доступно (устройство уже определено в родительском классе)
        X_tensor = torch.tensor(X_train_array, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_train_array, dtype=torch.float32, device=self.device)
        
        X_tensor = torch.nan_to_num(X_tensor)
        y_tensor = torch.nan_to_num(y_tensor)
        
        training_dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
        
        # Базовые значения для SHAP
        baseline_values = np.mean(X_train_array, axis=0)
        baseline_values = np.nan_to_num(baseline_values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # УЛУЧШЕНО: Группировка параметров с разными LR (как в Hybrid learning)
        # coeffs (линейные) требуют высокого LR (эмуляция LSE), а premises (нелинейные) - аккуратного
        param_groups = []
        coeffs_params = []
        sigma_params = []
        mu_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'coeff' in name:
                coeffs_params.append(param)
            elif 'sigma' in name:
                sigma_params.append(param)
            elif 'mu' in name:
                mu_params.append(param)
            else:
                other_params.append(param)
        
        # LR множители
        lr_coeffs_mult = 5.0  # Ускоряем линейную часть (эмуляция LSE)
        lr_sigma_mult = 0.5   # Замедляем ширину (стабильность)
        
        if coeffs_params:
            param_groups.append({'params': coeffs_params, 'lr': lr * lr_coeffs_mult})
        if sigma_params:
            param_groups.append({'params': sigma_params, 'lr': lr * lr_sigma_mult})
        if mu_params:
            param_groups.append({'params': mu_params, 'lr': lr})
        if other_params:
            param_groups.append({'params': other_params, 'lr': lr})
            
        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(param_groups)
        
        # Планировщик: снижаем LR если вышли на плато
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
        )
        
        loss_function = torch.nn.MSELoss()
        
        # История потерь
        history = {
            'total_loss': [],
            'main_loss': [],
            'shap_loss': [],
            'val_loss': [] if X_val is not None else None
        }
        
        # Инициализация для адаптивного gamma и плавной сходимости
        self.max_epochs = epochs
        self.best_main_loss = None
        self.no_improvement_count = 0
        if not hasattr(self, 'main_loss_ema'):
            self.main_loss_ema = None
        
        n_features = X_train_array.shape[1]
        use_exact = self.shap_exact and n_features <= 10
        mode = "точное (все 2^n подмножеств)" if use_exact else f"Monte Carlo аппроксимация ({self.shap_n_samples} семплов)"
        
        self.logger.info("Начинаю обучение ANFIS С НУЛЯ с встроенной SHAP-регуляризацией...")
        self.logger.info(f"Эпох: {epochs}, Батч: {batch_size}, LR: {lr}, Gamma: {self.gamma}")
        self.logger.info(f"Режим вычисления Shapley values: {mode} (n={n_features} признаков)")
        self.logger.info(f"Размер обучающей выборки: {len(X_train_array)} образцов")
        if self.use_pso_init:
            self.logger.info(f"Начальная инициализация: PSO ({self.pso_epochs} эпох)")
        
        self.logger.info(f"Тип модели (self.model): {type(self.model)}")
        
        # Проверяем, есть ли доступ к обертке с методом fit
        wrapper_with_fit = None
        if hasattr(self, 'anfis_wrapper') and hasattr(self.anfis_wrapper, 'fit'):
            wrapper_with_fit = self.anfis_wrapper
            self.logger.info("Найдена обертка BioAnfisRegressor с методом fit")
        elif hasattr(self.model, 'fit'):
            wrapper_with_fit = self.model
            self.logger.info("Модель имеет метод fit")
            
        self.logger.info(f"use_pso_init: {self.use_pso_init}")
        
        # Начальная инициализация через PSO (опционально)
        if self.use_pso_init and wrapper_with_fit is not None:
            self.logger.info("Начальная инициализация через PSO...")
            sys.stdout.flush()
            try:
                # Используем короткое PSO обучение для начальной инициализации
                # ВАЖНО: Мы не можем использовать wrapper_with_fit.fit() напрямую,
                # так как это перезапишет нашу модель self.model
                # Вместо этого мы создаем временную модель того же типа
                
                original_epochs = self.config['model']['optim_params']['epoch']
                # Используем pso_epochs из конфига integrated trainer
                self.config['model']['optim_params']['epoch'] = self.pso_epochs
                
                # Временно отключаем verbose для PSO
                original_verbose = self.config['model']['optim_params'].get('verbose', False)
                self.config['model']['optim_params']['verbose'] = False
                
                # Быстрое PSO обучение на подвыборке данных (или полных)
                init_size = len(X_train_array) # Используем все данные для качества
                X_init = X_train_array[:init_size]
                y_init = y_train_array[:init_size]
                
                # Создаем временную модель для PSO инициализации
                from xanfis import BioAnfisRegressor
                temp_model = BioAnfisRegressor(
                    num_rules=self.config['model']['num_rules'],
                    mf_class=self.config['model']['mf_class'],
                    vanishing_strategy=self.config['model'].get('vanishing_strategy', 'blend'),
                    optim=self.config['model']['optim'],
                    optim_params=self.config['model']['optim_params'],
                    reg_lambda=self.config['model']['reg_lambda'],
                    seed=self.config['model']['seed'],
                    n_workers=self.config['model'].get('n_workers', 4),
                    verbose=True # Включаем verbose чтобы видеть прогресс PSO
                )
                temp_model.size_input = X_init.shape[1]
                temp_model.size_output = y_init.shape[1] if y_init.ndim > 1 else 1
                temp_model.build_model()
                
                self.logger.info(f"Запуск PSO на {len(X_init)} образцах ({self.pso_epochs} эпох)...")
                sys.stdout.flush()
                
                temp_model.fit(X_init, y_init)
                
                # Копируем веса из временной модели в нашу PyTorch модель
                if hasattr(temp_model, 'network') and temp_model.network is not None:
                    # self.model это уже network (CustomANFIS)
                    self.model.load_state_dict(temp_model.network.state_dict(), strict=False)
                    
                    # Проверка качества после инициализации
                    self.model.eval()
                    with torch.no_grad():
                        X_check = torch.tensor(X_train_array, dtype=torch.float32, device=self.device)
                        y_check = torch.tensor(y_train_array, dtype=torch.float32, device=self.device)
                        preds_check = self.model(X_check)
                        init_mse = torch.nn.functional.mse_loss(preds_check, y_check).item()
                        self.logger.info(f"✅ PSO успешно завершено! MSE: {init_mse:.6f}")
                        
                    self.logger.info("Веса перенесены успешно")
                    sys.stdout.flush()
                
                # Восстанавливаем оригинальные параметры
                self.config['model']['optim_params']['epoch'] = original_epochs
                self.config['model']['optim_params']['verbose'] = original_verbose
                
            except Exception as e:
                self.logger.warning(f"PSO инициализация не удалась: {e}. Продолжаю с случайной инициализацией.")
                import traceback
                self.logger.warning(traceback.format_exc())
                sys.stdout.flush()
        
        # Основной цикл обучения с SHAP регуляризацией
        # (PSO инициализация уже выполнена, если была включена)
        self.logger.info("Начинаю основной цикл обучения с SHAP регуляризацией...")
        self.logger.info(f"Всего батчей в эпохе: {len(data_loader)}")
        sys.stdout.flush()
        
        for epoch in range(epochs):
            # Сохраняем текущую эпоху для адаптивной температуры
            self.current_epoch = epoch
            self.max_epochs = epochs
            
            self.logger.info(f"Начало эпохи {epoch + 1}/{epochs}")
            sys.stdout.flush()
            epoch_losses = {'total': [], 'main': [], 'shap': [], 'shap_components': []}
            skipped_batches = 0
            
            for batch_X, batch_y in data_loader:
                # Очистка входных данных
                batch_X = torch.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)
                batch_y = torch.nan_to_num(batch_y, nan=0.0, posinf=0.0, neginf=0.0)
                
                # ВАЖНО: batch_X должен иметь requires_grad=True для дифференцируемости SHAP регуляризации
                batch_X = batch_X.requires_grad_(True)
                
                # Проверка на пустые батчи
                if batch_X.numel() == 0 or batch_y.numel() == 0:
                    skipped_batches += 1
                    continue
                
                optimizer.zero_grad()
                
                # Прямой проход
                self.model.train()
                try:
                    predictions = self.model(batch_X)
                except Exception as e:
                    self.logger.warning(f"Ошибка при прямом проходе в эпохе {epoch+1}: {e}")
                    skipped_batches += 1
                    continue
                
                # Очистка предсказаний от NaN/Inf
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
                predictions = torch.clamp(predictions, min=0.0)
                
                # Проверка на валидность предсказаний
                if not torch.isfinite(predictions).all():
                    skipped_batches += 1
                    continue
                
                # Приведение к одинаковой размерности
                if predictions.shape != batch_y.shape:
                    min_dim = min(predictions.shape[-1], batch_y.shape[-1])
                    predictions = predictions[..., :min_dim]
                    batch_y = batch_y[..., :min_dim]
                
                # Основная функция потерь
                main_loss = loss_function(predictions, batch_y)
                
                # Штраф за отрицательные предсказания
                if self.negative_penalty > 0:
                    negative_mask = predictions < 0
                    if negative_mask.any():
                        negative_penalty = torch.mean(torch.clamp(-predictions, min=0.0) ** 2)
                        main_loss = main_loss + self.negative_penalty * negative_penalty
                
                # SHAP регуляризация (встроена в loss функцию!)
                # ВКЛЮЧЕНО: используем интегрированный SHAP для улучшения интерпретируемости
                if self.use_improved_shap:
                    shap_loss_tensor, shap_components = self._compute_improved_shap_regularization(
                        batch_X, baseline_values, predictions
                    )
                else:
                    # ИНТЕГРИРОВАННЫЙ SHAP: Используем настоящие Shapley values + дифференцируемую регуляризацию
                    # Гибридный подход: настоящие Shapley values как целевое распределение,
                    # gradient-based importance для дифференцируемости
                    shap_loss_tensor, shap_components = self._compute_integrated_shap_regularization(
                        batch_X, baseline_values, predictions
                    )
                
                # АДАПТИВНАЯ ФУНКЦИЯ ПОТЕРЬ ДЛЯ 2 ЗАДАЧ (используем улучшенную версию из родительского класса)
                # Родительский класс ShapAwareANFISTrainerImproved уже реализует адаптивную функцию потерь
                # Здесь используем ту же логику для интегрированного режима
                
                main_loss_detached = main_loss.detach()
                eps = 1e-8
                
                # Обновляем скользящее среднее main loss для стабильности
                if not hasattr(self, 'main_loss_ema') or self.main_loss_ema is None:
                    self.main_loss_ema = main_loss_detached.item()
                else:
                    ema_alpha = getattr(self, 'ema_alpha', 0.9)
                    self.main_loss_ema = ema_alpha * self.main_loss_ema + (1 - ema_alpha) * main_loss_detached.item()
                
                # Вычисляем адаптивный gamma для текущей эпохи
                # ВАЖНО: Для обучения с нуля (fit_from_scratch) ВСЕГДА используем warmup,
                # даже если use_adaptive_gamma выключена, иначе модель не сойдется.
                if hasattr(self, 'max_epochs') and self.max_epochs:
                    progress = epoch / self.max_epochs
                    # Используем параметр из конфига или дефолтный 30% warmup
                    warmup_frac = getattr(self, 'gamma_warmup_epochs', 0.3)
                    
                    # Если use_adaptive_gamma=True, используем сложную логику из конфига
                    if hasattr(self, 'use_adaptive_gamma') and self.use_adaptive_gamma:
                        if progress < warmup_frac:
                            gamma_ratio = progress / warmup_frac
                            current_gamma = getattr(self, 'gamma_start', 0.0) + (getattr(self, 'gamma_end', self.gamma) - getattr(self, 'gamma_start', 0.0)) * gamma_ratio
                        else:
                            # Логика для post-warmup (можно держать constant или менять)
                            # Если gamma_end не задан, используем self.gamma
                            target_g = getattr(self, 'gamma_end', self.gamma)
                            current_gamma = target_g
                    else:
                        # ПРОСТОЙ WARMUP (для стабильности fit_from_scratch)
                        # Линейно увеличиваем от 0 до self.gamma
                        if progress < warmup_frac:
                            current_gamma = self.gamma * (progress / warmup_frac)
                        else:
                            current_gamma = self.gamma
                else:
                    current_gamma = self.gamma
                
                # Вычисляем коэффициент замедления сходимости
                convergence_slowdown = 1.0
                if hasattr(self, 'use_convergence_smoothing') and self.use_convergence_smoothing:
                    if not hasattr(self, 'best_main_loss') or self.best_main_loss is None:
                        self.best_main_loss = main_loss_detached.item()
                        self.no_improvement_count = 0
                    else:
                        improvement = (self.best_main_loss - main_loss_detached.item()) / (self.best_main_loss + eps)
                        if improvement < 0.001:  # Улучшение меньше 0.1%
                            self.no_improvement_count = getattr(self, 'no_improvement_count', 0) + 1
                            convergence_slowdown = 1.0 / (1.0 + self.no_improvement_count * 0.1)
                        else:
                            self.no_improvement_count = 0
                            if main_loss_detached.item() < self.best_main_loss:
                                self.best_main_loss = main_loss_detached.item()
                
                # Адаптивная нормализация SHAP loss
                shap_loss_detached = shap_loss_tensor.detach().item()
                target_shap_ratio = getattr(self, 'target_shap_ratio', 0.2)
                
                if shap_loss_detached > eps and self.main_loss_ema > eps:
                    current_ratio = shap_loss_detached / self.main_loss_ema
                    progress = epoch / self.max_epochs if hasattr(self, 'max_epochs') and self.max_epochs else 0.5
                    
                    # Динамический target_ratio: больше влияния SHAP на поздних этапах
                    target_ratio_dynamic = target_shap_ratio * (0.5 + 0.5 * progress)
                    
                    if current_ratio > target_ratio_dynamic * 2:
                        scale_factor = target_ratio_dynamic / current_ratio
                    elif current_ratio < target_ratio_dynamic / 2:
                        scale_factor = target_ratio_dynamic / current_ratio
                    else:
                        scale_factor = 1.0
                    
                    scale_factor *= convergence_slowdown
                    shap_loss_normalized = shap_loss_tensor * scale_factor
                else:
                    scale_factor = target_shap_ratio / (shap_loss_detached / (self.main_loss_ema + eps) + eps)
                    scale_factor *= convergence_slowdown
                    shap_loss_normalized = shap_loss_tensor * scale_factor
                
                # АДАПТИВНАЯ ФУНКЦИЯ ПОТЕРЬ: балансировка двух задач
                total_loss = main_loss + current_gamma * shap_loss_normalized
                
                # Обратное распространение
                if not torch.isfinite(total_loss):
                    skipped_batches += 1
                    continue
                
                try:
                    total_loss.backward()
                    
                    # Проверка градиентов на NaN/Inf
                    has_nan_grad = False
                    for param in self.model.parameters():
                        if param.grad is not None:
                            if not torch.isfinite(param.grad).all():
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        optimizer.zero_grad()
                        skipped_batches += 1
                        continue
                    
                    if self.grad_clip and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    
                    optimizer.step()
                except Exception as e:
                    self.logger.warning(f"Ошибка при обратном распространении в эпохе {epoch+1}: {e}")
                    optimizer.zero_grad()
                    skipped_batches += 1
                    continue
                
                # Сохранение потерь
                epoch_losses['total'].append(float(total_loss.item()))
                epoch_losses['main'].append(float(main_loss.item()))
                shap_loss_value = float(shap_loss_tensor.item()) if isinstance(shap_loss_tensor, torch.Tensor) else float(shap_loss_tensor)
                epoch_losses['shap'].append(shap_loss_value)
                # Сохраняем нормализованный SHAP loss для анализа балансировки
                shap_loss_normalized_value = float(shap_loss_normalized.detach().item())
                if 'shap_normalized' not in epoch_losses:
                    epoch_losses['shap_normalized'] = []
                epoch_losses['shap_normalized'].append(shap_loss_normalized_value)
                
                # Сохранение компонентов SHAP регуляризации (для интегрированного и улучшенного SHAP)
                if 'shap_components' in locals() and isinstance(shap_components, dict):
                    epoch_losses['shap_components'].append(shap_components)
            
            # Усреднение потерь по эпохе
            for loss_type in ['total', 'main', 'shap']:
                values = epoch_losses[loss_type]
                history[f'{loss_type}_loss'].append(float(np.mean(values)) if values else float('nan'))
            
            # Сохранение компонентов SHAP регуляризации в историю
            if epoch_losses.get('shap_components'):
                # Усредняем компоненты по эпохе
                avg_components = {}
                # Ключи компонентов для интегрированного SHAP (включая новые: faithfulness, stability)
                component_keys = ['sparsity', 'consistency', 'stability', 'entropy', 'normalized_entropy', 
                                 'cv', 'normalized_cv', 'faithfulness', 'mse', 'kl', 
                                 'importance_mean', 'importance_std', 'max_importance', 'min_importance',
                                 'true_shap_mean', 'true_shap_std']
                
                for key in component_keys:
                    values = [comp.get(key, 0) for comp in epoch_losses['shap_components'] if isinstance(comp, dict) and key in comp]
                    if values:
                        avg_components[key] = float(np.mean(values))
                
                # Сохраняем средние компоненты за эпоху
                if 'shap_components' not in history:
                    history['shap_components'] = []
                history['shap_components'].append(avg_components)
            
            # Валидация (если предоставлена)
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(np.array(X_val), dtype=torch.float32, device=self.device)
                    y_val_tensor = torch.tensor(np.array(y_val), dtype=torch.float32, device=self.device)
                    val_predictions = self.model(X_val_tensor)
                    val_predictions = torch.clamp(val_predictions, min=0.0)
                    val_loss = loss_function(val_predictions, y_val_tensor)
                    val_loss_val = float(val_loss.item())
                    if history['val_loss'] is not None:
                        history['val_loss'].append(val_loss_val)
                    
                    # Шаг планировщика по валидационному лоссу
                    if 'scheduler' in locals():
                        scheduler.step(val_loss_val)
            
            # Прогресс на каждой эпохе
            skipped_info = f", Пропущено батчей: {skipped_batches}" if skipped_batches > 0 else ""
            val_info = f", Val: {history['val_loss'][-1]:.6f}" if history['val_loss'] and len(history['val_loss']) > 0 else ""
            
            # Логируем нормализованный SHAP loss для отслеживания балансировки
            shap_norm_info = ""
            if epoch_losses.get('shap_normalized'):
                avg_shap_norm = np.mean(epoch_losses['shap_normalized'])
                shap_norm_info = f", SHAP_norm: {avg_shap_norm:.4f}"
            
            # Логируем компоненты SHAP регуляризации (для интегрированного и улучшенного SHAP)
            shap_info = ""
            if epoch_losses.get('shap_components'):
                # Берем средние значения компонентов за эпоху
                avg_components = {}
                # Ключи для логирования (включая новые компоненты)
                keys_to_log = ['sparsity', 'consistency', 'stability', 'faithfulness', 'normalized_entropy']
                for key in keys_to_log:
                    values = [comp.get(key, 0) for comp in epoch_losses['shap_components'] if isinstance(comp, dict) and key in comp]
                    if values:
                        avg_components[key] = np.mean(values)
                
                if avg_components:
                    # Формируем информативную строку с компонентами
                    comp_strs = []
                    if 'consistency' in avg_components:
                        comp_strs.append(f"C:{avg_components['consistency']:.4f}")
                    if 'sparsity' in avg_components or 'normalized_entropy' in avg_components:
                        sparsity_val = avg_components.get('sparsity', avg_components.get('normalized_entropy', 0))
                        comp_strs.append(f"S:{sparsity_val:.4f}")
                    if 'faithfulness' in avg_components:
                        comp_strs.append(f"F:{avg_components['faithfulness']:.4f}")
                    if 'stability' in avg_components:
                        comp_strs.append(f"St:{avg_components['stability']:.4f}")
                    
                    if comp_strs:
                        shap_info = f" [{', '.join(comp_strs)}]"
            
            self.logger.info(f"Эпоха {epoch + 1}/{epochs}: "
                  f"Total: {history['total_loss'][-1]:.6f}, "
                  f"Main: {history['main_loss'][-1]:.6f}, "
                  f"SHAP: {history['shap_loss'][-1]:.6f}, "
                  f"Gamma: {current_gamma:.4f}"
                  f"{shap_norm_info}{shap_info}{val_info}{skipped_info}")
            sys.stdout.flush()
            
            if skipped_batches > 0:
                self.logger.warning(f"В эпохе {epoch + 1} пропущено {skipped_batches} батчей из-за ошибок")
                sys.stdout.flush()
        
        self.training_time = time.time() - start_time
        self.logger.info(f"Обучение завершено за {self.training_time:.2f} сек")
        sys.stdout.flush()
        
        return history
    
    def _compute_integrated_shap_regularization(self, batch_X, baseline_values, predictions):
        """
        Вычисляет интегрированную SHAP регуляризацию с использованием НАСТОЯЩИХ Shapley values.
        
        ГИБРИДНЫЙ ПОДХОД:
        - Использует НАСТОЯЩИЕ Shapley values для точной оценки важности (периодически)
        - Использует gradient-based importance для дифференцируемости (каждый батч)
        - Минимизирует разницу между ними (consistency) + sparsity регуляризация
        
        КЛЮЧЕВЫЕ ПРЕИМУЩЕСТВА:
        ✅ Использует настоящие Shapley values (точная оценка важности)
        ✅ Полностью дифференцируема через gradient-based importance
        ✅ Sparsity регуляризация для интерпретируемости
        ✅ Consistency регуляризация для согласованности
        
        МАТЕМАТИЧЕСКИ ОБОСНОВАННАЯ ФОРМУЛА С УЛУЧШЕНИЯМИ:
        
        1. Вычисляем настоящие Shapley values (периодически):
           φ_true = _calculate_shap_approximation(batch_X, baseline)
           φ_true_norm = φ_true / Σ φ_true
        
        2. Вычисляем gradient-based importance (дифференцируемо):
           I_grad = |∂f/∂x| × |x|
           I_grad_norm = softmax(I_grad / τ)  [τ = temperature]
        
        3. РЕГУЛЯРИЗАЦИЯ (четыре компонента для максимальной интерпретируемости):
           a) CONSISTENCY: 0.6 × MSE + 0.4 × KL
              MSE: ||I_grad_norm - φ_true_norm||²
              KL: KL(I_grad_norm || φ_true_norm)
              Обеспечивает согласованность с настоящими Shapley values
           
           b) SPARSITY: H(I_grad_norm) = -Σ I_grad_norm_i × log(I_grad_norm_i)
              Выделяет важные признаки (низкая энтропия = sparse распределение)
           
           c) FAITHFULNESS: ||(f(x) - f(baseline)) - Σ(φ_i × (x_i - baseline_i))||²
              Гарантирует соответствие объяснений реальному влиянию признаков
              Модель корректируется так, чтобы важность отражала реальное влияние
           
           d) STABILITY: Var([φ(x_i) for x_i in batch])
              Минимизирует вариацию важности для похожих образцов
              Обеспечивает стабильность объяснений
        
        4. Комбинируем: L_SHAP = 0.5 × Consistency + 0.2 × Sparsity + 0.15 × Faithfulness + 0.15 × Stability
        
        ПРЕИМУЩЕСТВА:
        ✅ Математически обоснована (четыре компонента для интерпретируемости)
        ✅ Использует настоящие Shapley values для точности
        ✅ Полностью дифференцируема через I_grad
        ✅ Улучшает интерпретируемость (sparsity + faithfulness + stability)
        ✅ Обеспечивает согласованность (consistency)
        ✅ Корректирует модель во время обучения для лучшей интерпретируемости
        
        Args:
            batch_X: Батч признаков (torch.Tensor) - ДОЛЖЕН БЫТЬ requires_grad=True
            baseline_values: Базовые значения для SHAP (np.ndarray)
            predictions: Предсказания модели (torch.Tensor) - ДОЛЖЕН БЫТЬ requires_grad=True
            
        Returns:
            tuple: (shap_loss_tensor, shap_components_dict)
                  shap_loss_tensor - ДИФФЕРЕНЦИРУЕМЫЙ тензор с requires_grad=True
        """
        batch_size = batch_X.shape[0]
        n_features = batch_X.shape[1]
        
        # ШАГ 1: Вычисляем gradient-based importance (дифференцируемо!)
        # Убеждаемся, что batch_X требует градиенты
        batch_X.requires_grad_(True)
        
        # Получаем предсказания для батча (если еще не вычислены)
        if not predictions.requires_grad:
            predictions = self.model(batch_X)
        
        # Вычисляем градиенты предсказаний по входам
        output_dim = predictions.shape[1] if predictions.ndim > 1 else 1
        grad_outputs = torch.ones_like(predictions) / output_dim
        
        # Получаем градиенты для всех признаков за один вызов
        grad_input = torch.autograd.grad(
            outputs=predictions,
            inputs=batch_X,
            grad_outputs=grad_outputs,
            create_graph=True,  # ВАЖНО: создаем граф для прохождения градиентов
            retain_graph=True,
            only_inputs=True
        )[0]  # [batch_size, n_features]
        
        # Вычисление важности признаков через градиенты
        grad_importance = torch.abs(grad_input) * torch.abs(batch_X)
        importance_per_feature = torch.mean(grad_importance, dim=0)  # [n_features]
        
        # Добавляем минимальный порог для стабильности
        min_threshold = torch.max(importance_per_feature) * 1e-6
        importance_per_feature = torch.clamp(importance_per_feature, min=min_threshold)
        
        # МАТЕМАТИЧЕСКИ КОРРЕКТНАЯ НОРМАЛИЗАЦИЯ: L1 нормализация вместо softmax
        # L1 нормализация сохраняет относительные различия между признаками
        # В отличие от softmax, который сглаживает различия
        importance_sum = torch.sum(importance_per_feature) + 1e-10
        grad_importance_normalized = importance_per_feature / importance_sum  # [n_features]
        grad_importance_normalized = torch.clamp(grad_importance_normalized, min=1e-10, max=1.0)
        
        # Сохраняем не нормализованную важность для Faithfulness (нужны абсолютные значения)
        importance_unnormalized = importance_per_feature  # [n_features]
        
        # ШАГ 2: Вычисляем настоящие Shapley values (периодически)
        consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        mse_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        kl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.use_true_shap:
            # Обновляем настоящие Shapley values периодически
            self.true_shap_batch_count += 1
            if (self.true_shap_importance is None or 
                self.true_shap_batch_count % self.true_shap_update_frequency == 0):
                
                # ВОЗВРАЩЕНО К РАБОЧЕЙ ВЕРСИИ: используем средний образец (как в старой версии)
                # Хотя математически менее корректно, но это рабочая версия с R² ≈ 0.7
                # Можно улучшить позже, но сначала вернем работоспособность
                mean_sample = torch.mean(batch_X, dim=0).detach().cpu().numpy()
                
                # Вычисляем настоящие Shapley values для среднего образца
                true_shap = self._calculate_shap_approximation(mean_sample, baseline_values)
                true_shap = np.nan_to_num(true_shap, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Нормализуем
                if true_shap.ndim != 1 or true_shap.size == 0:
                    true_shap_normalized = np.ones(n_features) / n_features
                else:
                    shap_sum = float(np.sum(true_shap))
                    if shap_sum <= 1e-12 or not np.isfinite(shap_sum):
                        true_shap_normalized = np.ones(n_features) / n_features
                    else:
                        true_shap_normalized = true_shap / shap_sum
                
                # Сохраняем как глобальную важность
                self.true_shap_importance = torch.tensor(
                    true_shap_normalized, 
                    device=self.device, 
                    dtype=torch.float32
                )
            
            # МАТЕМАТИЧЕСКИ ОБОСНОВАННАЯ РЕГУЛЯРИЗАЦИЯ: Комбинированный подход
            if self.true_shap_importance is not None:
                # ИСПРАВЛЕНО: Используем только положительные Shapley values для нормализации
                # Shapley values могут быть отрицательными, но для важности признаков нужны только положительные
                # Используем abs() только для получения абсолютных значений важности, затем нормализуем
                true_shap_positive = torch.clamp(self.true_shap_importance, min=0.0)  # Только положительные
                true_shap_sum = torch.sum(true_shap_positive) + 1e-10
                
                # Если все значения отрицательные или нулевые, используем равномерное распределение
                if true_shap_sum <= 1e-10:
                    true_shap_normalized = torch.ones_like(true_shap_positive) / n_features
                else:
                    true_shap_normalized = true_shap_positive / true_shap_sum
                
                true_shap_normalized = torch.clamp(true_shap_normalized, min=1e-10, max=1.0)
                
                # КОМПОНЕНТ 1: MSE (Mean Squared Error) - точное совпадение
                # Оба распределения нормализованы через L1, поэтому сравнение корректно
                mse_loss = torch.mean((grad_importance_normalized - true_shap_normalized) ** 2)
                
                # КОМПОНЕНТ 2: JS Divergence (Jensen-Shannon) - ИСПРАВЛЕНО для гарантии неотрицательности
                # JS Divergence всегда неотрицательна и симметрична
                # JS(P||Q) = (KL(P||M) + KL(Q||M)) / 2, где M = (P + Q) / 2
                m = (grad_importance_normalized + true_shap_normalized) / 2 + 1e-10
                kl_pm = torch.sum(grad_importance_normalized * torch.log(
                    (grad_importance_normalized + 1e-10) / m
                ))
                kl_qm = torch.sum(true_shap_normalized * torch.log(
                    (true_shap_normalized + 1e-10) / m
                ))
                js_loss = (kl_pm + kl_qm) / 2  # JS Divergence всегда ≥ 0
                
                # Комбинируем MSE и JS для consistency
                consistency_loss = 0.6 * mse_loss + 0.4 * js_loss
            else:
                consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ШАГ 3: SPARSITY регуляризация на gradient-based importance
        # МАТЕМАТИЧЕСКИ КОРРЕКТНО: используем L1 нормализованную важность (не softmax!)
        # L1 нормализация сохраняет относительные различия, в отличие от softmax
        # Энтропия Шеннона: H(φ) = -Σ φᵢ log(φᵢ)
        # Низкая энтропия = sparse распределение (несколько признаков важны)
        # Высокая энтропия = равномерное распределение (все признаки одинаково важны)
        # Используем уже L1 нормализованную важность (grad_importance_normalized)
        entropy = -torch.sum(grad_importance_normalized * torch.log(grad_importance_normalized + 1e-10))
        max_entropy = torch.log(torch.tensor(float(n_features), device=self.device, dtype=torch.float32))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        sparsity_loss = normalized_entropy
        
        # УЛУЧШЕНИЕ 1: FAITHFULNESS (Верность объяснений) - ИСПРАВЛЕНО: используем нулевой baseline
        # Гарантирует, что объяснения соответствуют реальному влиянию признаков
        # МАТЕМАТИЧЕСКИ КОРРЕКТНАЯ ФОРМУЛА: Линейное приближение через градиенты на baseline
        # Формула: L_faithfulness = ||(f(x) - f(baseline)) - Σ(∇f(baseline)_i × (x_i - baseline_i))||²
        # Это проверяет, насколько хорошо линейное приближение (через градиенты) предсказывает изменение предсказания
        # ИСПРАВЛЕНО: Используем нулевой baseline вместо среднего, чтобы X_change был значимым
        baseline_tensor = torch.zeros(n_features, device=self.device, dtype=torch.float32, requires_grad=False)
        baseline_X = baseline_tensor.unsqueeze(0).expand(batch_size, -1)  # [batch_size, n_features]
        
        # Вычисляем предсказание на baseline
        baseline_X.requires_grad_(True)
        baseline_pred = self.model(baseline_X)  # [batch_size, output_dim]
        
        # Вычисляем градиенты на baseline (линейное приближение)
        # Градиенты показывают локальную чувствительность модели к изменениям признаков
        output_dim = baseline_pred.shape[1] if baseline_pred.ndim > 1 else 1
        grad_outputs_baseline = torch.ones_like(baseline_pred) / output_dim
        
        grad_at_baseline = torch.autograd.grad(
            outputs=baseline_pred,
            inputs=baseline_X,
            grad_outputs=grad_outputs_baseline,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # [batch_size, n_features]
        
        # Изменение предсказания относительно baseline
        pred_change = predictions - baseline_pred.detach()  # [batch_size, output_dim]
        
        # Линейное приближение изменения предсказания через градиенты на baseline
        # Формула: linear_change = Σ(∇f(baseline)_i × (x_i - baseline_i))
        X_change = batch_X - baseline_X.detach()  # [batch_size, n_features]
        linear_change = torch.sum(grad_at_baseline * X_change, dim=1)  # [batch_size]
        
        # Если векторный выход, используем среднее по выходам для pred_change
        if pred_change.ndim > 1:
            pred_change_scalar = torch.mean(pred_change, dim=1)  # [batch_size]
        else:
            pred_change_scalar = pred_change.squeeze() if pred_change.ndim > 0 else pred_change  # [batch_size]
        
        # Faithfulness loss: минимизируем разницу между реальным изменением и линейным приближением
        # Это проверяет, насколько хорошо градиенты предсказывают изменение предсказания
        # Низкий faithfulness означает, что модель локально линейна (градиенты хорошо предсказывают изменения)
        faithfulness_loss = torch.mean((pred_change_scalar - linear_change) ** 2)
        
        # УЛУЧШЕНИЕ 2: STABILITY (Стабильность объяснений)
        # Гарантирует, что похожие входы получают похожие объяснения
        # Формула: L_stability = Var([φ(x_i) for x_i in batch])
        # Минимизируем вариацию важности по батчу
        if batch_size > 1:
            # Вычисляем важность для каждого образца в батче
            # Используем градиенты для каждого образца отдельно
            importance_per_sample = torch.abs(grad_input) * torch.abs(batch_X)  # [batch_size, n_features]
            
            # Нормализуем важность для каждого образца
            importance_per_sample_sum = torch.sum(importance_per_sample, dim=1, keepdim=True) + 1e-10
            importance_per_sample_norm = importance_per_sample / importance_per_sample_sum
            
            # Вычисляем вариацию важности по батчу для каждого признака
            importance_variance = torch.var(importance_per_sample_norm, dim=0)  # [n_features]
            stability_loss = torch.mean(importance_variance)  # Средняя вариация по всем признакам
        else:
            stability_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # ШАГ 4: Комбинируем компоненты с УЛУЧШЕНИЯМИ ДЛЯ ИНТЕРПРЕТИРУЕМОСТИ
        # УЛУЧШЕННАЯ ФОРМУЛА: добавляем Faithfulness и Stability для лучшей интерпретируемости
        # ВРЕМЕННО УМЕНЬШЕНЫ ВЕСА новых компонентов для стабильности
        # Веса: Consistency (MSE+KL) 60%, Sparsity 25%, Faithfulness 7.5%, Stability 7.5%
        if self.use_true_shap and self.true_shap_importance is not None:
            # Комбинируем все компоненты для максимальной интерпретируемости
            # Consistency обеспечивает согласованность с настоящими Shapley values
            # Sparsity выделяет важные признаки
            # Faithfulness гарантирует соответствие объяснений влиянию признаков (уменьшенный вес)
            # Stability обеспечивает стабильность объяснений (уменьшенный вес)
            shap_loss_tensor = (
                0.6 * consistency_loss +      # 60% - согласованность (MSE + KL) - УВЕЛИЧЕНО
                0.25 * sparsity_loss +        # 25% - разреженность (Entropy) - УВЕЛИЧЕНО
                0.075 * faithfulness_loss +   # 7.5% - верность объяснений - УМЕНЬШЕНО
                0.075 * stability_loss        # 7.5% - стабильность объяснений - УМЕНЬШЕНО
            )
        else:
            # Если настоящие Shapley values не используются, используем только sparsity + stability
            shap_loss_tensor = 0.9 * sparsity_loss + 0.1 * stability_loss
        
        # Убеждаемся, что тензор требует градиенты
        shap_loss_tensor = shap_loss_tensor.requires_grad_(True)
        
        # Компоненты для логирования
        shap_components = {
            'total': shap_loss_tensor.detach().item(),
            'sparsity': sparsity_loss.detach().item(),
            'normalized_entropy': normalized_entropy.detach().item(),
            'entropy': entropy.detach().item(),
            'importance_mean': torch.mean(grad_importance_normalized).detach().item(),
            'importance_std': torch.std(grad_importance_normalized).detach().item(),
            'max_importance': torch.max(grad_importance_normalized).detach().item(),
            'min_importance': torch.min(grad_importance_normalized).detach().item(),
            'faithfulness': faithfulness_loss.detach().item(),
            'stability': stability_loss.detach().item()
        }
        
        if self.use_true_shap and self.true_shap_importance is not None:
            shap_components['consistency'] = consistency_loss.detach().item()
            shap_components['mse'] = mse_loss.detach().item()
            shap_components['js'] = js_loss.detach().item()  # ИСПРАВЛЕНО: JS Divergence вместо KL
            shap_components['true_shap_mean'] = torch.mean(self.true_shap_importance).detach().item()
            shap_components['true_shap_std'] = torch.std(self.true_shap_importance).detach().item()
        
        return shap_loss_tensor, shap_components
    
    def _compute_simple_differentiable_shap_regularization(self, batch_X, baseline_values, predictions):
        """
        Вычисляет простую дифференцируемую SHAP регуляризацию с SPARSITY формулой.
        
        КЛЮЧЕВОЕ УЛУЧШЕНИЕ:
        - Использует gradient-based importance для дифференцируемости (градиенты проходят!)
        - Применяет SPARSITY регуляризацию (лучше для интерпретируемости!)
        - Все операции через torch для прохождения градиентов
        
        ПОЧЕМУ SPARSITY, А НЕ РАВНОМЕРНОЕ РАСПРЕДЕЛЕНИЕ:
        ❌ Равномерное распределение заставляет ВСЕ признаки быть важными
        ❌ Это плохо, если в датасете только несколько признаков важны (Q5, Q7, Q10)
        ❌ Модель будет пытаться сделать все признаки важными, даже если это не так
        
        ✅ SPARSITY поощряет несколько признаков быть важными, остальные - нет
        ✅ Это дает четкое понимание: какие признаки действительно важны
        ✅ Улучшает интерпретируемость: видно доминирующие признаки
        
        ФОРМУЛА:
        1. Вычисляем важность через градиенты (дифференцируемо):
           importance = |grad_input| * |batch_X|
           importance_normalized = softmax(importance / temperature)
        
        2. Применяем SPARSITY регуляризацию (минимизируем энтропию):
           entropy = -Σ(importance_normalized_i * log(importance_normalized_i))
           max_entropy = log(n_features)
           normalized_entropy = entropy / max_entropy
           shap_loss = normalized_entropy  # Минимизируем энтропию = максимизируем sparsity
        
        3. Все через torch операции для прохождения градиентов!
        
        ПРЕИМУЩЕСТВА:
        ✅ Полностью дифференцируема (requires_grad=True)
        ✅ Градиенты проходят через SHAP loss
        ✅ Простая формула (минимизация энтропии)
        ✅ Улучшает интерпретируемость (sparsity = четкое выделение важных признаков)
        ✅ Корректирует loss напрямую
        ✅ Не навязывает равномерное распределение
        
        Args:
            batch_X: Батч признаков (torch.Tensor) - ДОЛЖЕН БЫТЬ requires_grad=True
            baseline_values: Базовые значения для SHAP (np.ndarray) - не используется, но для совместимости
            predictions: Предсказания модели (torch.Tensor) - ДОЛЖЕН БЫТЬ requires_grad=True
            
        Returns:
            tuple: (shap_loss_tensor, shap_components_dict)
                  shap_loss_tensor - ДИФФЕРЕНЦИРУЕМЫЙ тензор с requires_grad=True
        """
        batch_size = batch_X.shape[0]
        n_features = batch_X.shape[1]
        
        # ВЫЧИСЛЕНИЕ ВАЖНОСТИ ПРИЗНАКОВ ЧЕРЕЗ ГРАДИЕНТЫ (дифференцируемо!)
        # Используем gradient-based importance для дифференцируемости
        # Это аппроксимация SHAP через градиенты: I_i ≈ |∂f/∂x_i| * |x_i|
        
        # Убеждаемся, что batch_X требует градиенты
        batch_X.requires_grad_(True)
        
        # Получаем предсказания для батча (если еще не вычислены)
        if not predictions.requires_grad:
            predictions = self.model(batch_X)
        
        # Вычисляем градиенты предсказаний по входам
        output_dim = predictions.shape[1] if predictions.ndim > 1 else 1
        grad_outputs = torch.ones_like(predictions) / output_dim
        
        # Получаем градиенты для всех признаков за один вызов
        grad_input = torch.autograd.grad(
            outputs=predictions,
            inputs=batch_X,
            grad_outputs=grad_outputs,
            create_graph=True,  # ВАЖНО: создаем граф для прохождения градиентов
            retain_graph=True,
            only_inputs=True
        )[0]  # [batch_size, n_features]
        
        # Вычисление важности признаков через градиенты
        # Простая формула: importance = |grad_input| * |batch_X|
        grad_importance = torch.abs(grad_input) * torch.abs(batch_X)
        
        # Усредняем по батчу для каждого признака
        importance_per_feature = torch.mean(grad_importance, dim=0)  # [n_features]
        
        # Добавляем минимальный порог для стабильности
        min_threshold = torch.max(importance_per_feature) * 1e-6
        importance_per_feature = torch.clamp(importance_per_feature, min=min_threshold)
        
        # Нормализация важности через softmax (для получения распределения вероятностей)
        # Используем температуру для контроля "резкости" распределения
        temperature = 0.5  # Температура < 1 делает распределение более "острым" (sparse)
        importance_scaled = importance_per_feature / (torch.max(importance_per_feature) + 1e-10) / temperature
        importance_normalized = torch.softmax(importance_scaled, dim=0)
        importance_normalized = torch.clamp(importance_normalized, min=1e-10, max=1.0)
        
        # SPARSITY РЕГУЛЯРИЗАЦИЯ: Минимизируем энтропию распределения важности
        # Энтропия Шеннона: H(φ) = -Σ φᵢ log(φᵢ)
        # Низкая энтропия = sparse распределение (несколько признаков важны)
        # Высокая энтропия = равномерное распределение (все признаки одинаково важны)
        entropy = -torch.sum(importance_normalized * torch.log(importance_normalized + 1e-10))
        max_entropy = torch.log(torch.tensor(float(n_features), device=self.device, dtype=torch.float32))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        # SHAP потеря: нормализованная энтропия
        # Минимизируем энтропию = максимизируем sparsity = улучшаем интерпретируемость
        # ВСЕ ОПЕРАЦИИ ЧЕРЕЗ TORCH ДЛЯ ПРОХОЖДЕНИЯ ГРАДИЕНТОВ!
        shap_loss_tensor = normalized_entropy
        
        # Убеждаемся, что тензор требует градиенты
        shap_loss_tensor = shap_loss_tensor.requires_grad_(True)
        
        # Компоненты для логирования
        shap_components = {
            'total': shap_loss_tensor.detach().item(),
            'entropy': entropy.detach().item(),
            'normalized_entropy': normalized_entropy.detach().item(),
            'importance_mean': torch.mean(importance_normalized).detach().item(),
            'importance_std': torch.std(importance_normalized).detach().item(),
            'max_importance': torch.max(importance_normalized).detach().item(),
            'min_importance': torch.min(importance_normalized).detach().item()
        }
        
        return shap_loss_tensor, shap_components
    
    def _compute_improved_shap_regularization(self, batch_X, baseline_values, predictions):
        """
        Вычисляет улучшенную SHAP регуляризацию с несколькими компонентами.
        
        МАТЕМАТИЧЕСКИ ОПРАВДАННЫЙ ПОДХОД:
        Использует градиентную важность признаков (gradient-based importance) для дифференцируемости.
        Все операции выполняются через torch для прохождения градиентов.
        
        МЕХАНИЗМ УЛУЧШЕНИЯ ИНТЕРПРЕТИРУЕМОСТИ:
        
        1. SPARSITY регуляризация:
           - Минимизирует энтропию распределения важности признаков
           - Поощряет модель фокусироваться на небольшом наборе действительно важных признаков
           - Улучшает интерпретируемость: вместо равномерного распределения важности получаем
             четкое выделение доминирующих признаков (например, Q5, Q7, Q10)
           - Математически: H(φ) = -Σ φᵢ log(φᵢ) → min, где φᵢ - важность признака i
        
        2. CONSISTENCY регуляризация:
           - Обеспечивает согласованность локальных (для конкретного образца) и глобальных
             (по всему датасету) SHAP значений
           - Улучшает интерпретируемость: объяснения становятся стабильными и предсказуемыми
           - Если признак важен глобально, он должен быть важен и локально (с некоторой вариацией)
           - Математически: ||φ_local - φ_global||² → min
        
        3. STABILITY регуляризация:
           - Минимизирует дисперсию SHAP значений для похожих образцов
           - Улучшает интерпретируемость: похожие входы получают похожие объяснения
           - Делает модель более надежной и предсказуемой в объяснениях
           - Математически: Var(φ|X_similar) → min
        
        Args:
            batch_X: Батч признаков (torch.Tensor) - ДОЛЖЕН БЫТЬ requires_grad=True
            baseline_values: Базовые значения для SHAP (np.ndarray)
            predictions: Предсказания модели (torch.Tensor) - ДОЛЖЕН БЫТЬ requires_grad=True
            
        Returns:
            tuple: (shap_loss_tensor, shap_components_dict)
                  shap_loss_tensor - ДИФФЕРЕНЦИРУЕМЫЙ тензор с requires_grad=True
        """
        batch_size = batch_X.shape[0]
        n_features = batch_X.shape[1]
        
        # ВЫЧИСЛЕНИЕ ВАЖНОСТИ ПРИЗНАКОВ ЧЕРЕЗ ГРАДИЕНТЫ (дифференцируемо!)
        # Используем gradient-based importance вместо SHAP для дифференцируемости
        # Это аппроксимация SHAP через градиенты: I_i ≈ |∂f/∂x_i| * |x_i|
        
        # Вычисляем градиенты предсказаний по входам
        batch_X.requires_grad_(True)
        
        # Получаем предсказания для батча (если еще не вычислены)
        if not predictions.requires_grad:
            predictions = self.model(batch_X)
        
        # УЛУЧШЕННОЕ ВЫЧИСЛЕНИЕ ВАЖНОСТИ ЧЕРЕЗ ГРАДИЕНТЫ (все за раз!)
        # Вычисляем все градиенты одновременно для стабильности и скорости
        output_dim = predictions.shape[1] if predictions.ndim > 1 else 1
        
        # Вычисляем градиенты для всех признаков одновременно
        # grad_outputs: нормализуем по выходным измерениям
        grad_outputs = torch.ones_like(predictions) / output_dim
        
        # Получаем градиенты для всех признаков за один вызов
        grad_input = torch.autograd.grad(
            outputs=predictions,
            inputs=batch_X,
            grad_outputs=grad_outputs,
            create_graph=True,  # ВАЖНО: создаем граф для второго порядка
            retain_graph=True,
            only_inputs=True
        )[0]  # [batch_size, n_features]
        
        # УЛУЧШЕНО: Вычисление важности признаков с учетом нескольких факторов
        # 1. Градиентная важность: |градиент| * |значение|
        grad_importance = torch.abs(grad_input) * torch.abs(batch_X)
        
        # 2. Дополнительно: учитываем знак градиента и значение (для более точной оценки)
        # Если градиент и значение имеют одинаковый знак - признак более важен
        sign_alignment = torch.sign(grad_input) * torch.sign(batch_X)
        alignment_bonus = torch.abs(grad_input) * torch.abs(batch_X) * (sign_alignment + 1.0) / 2.0
        
        # 3. Комбинируем: основная важность + бонус за согласованность знаков
        combined_importance = 0.8 * grad_importance + 0.2 * alignment_bonus
        
        # Усредняем по батчу для каждого признака
        importance_per_feature = torch.mean(combined_importance, dim=0)  # [n_features]
        
        # 4. УЛУЧШЕНО: Добавляем минимальный порог для стабильности
        # Это предотвращает ситуацию, когда все признаки имеют нулевую важность
        min_threshold = torch.max(importance_per_feature) * 1e-6
        importance_per_feature = torch.clamp(importance_per_feature, min=min_threshold)
        
        importance_tensor = importance_per_feature
        
        # УЛУЧШЕНО: Нормализация важности с температурой для более стабильного обучения
        # Используем температурное масштабирование перед softmax для контроля "резкости" распределения
        temperature = 0.5  # Температура < 1 делает распределение более "острым" (sparse)
        importance_scaled = importance_tensor / (torch.max(importance_tensor) + 1e-10) / temperature
        importance_normalized = torch.softmax(importance_scaled, dim=0)
        importance_normalized = torch.clamp(importance_normalized, min=1e-10, max=1.0)
        
        # 1. SPARSITY регуляризация - УЛУЧШЕННАЯ формула (более агрессивная!)
        # Энтропия Шеннона: H(φ) = -Σ φᵢ log(φᵢ)
        entropy = -torch.sum(importance_normalized * torch.log(importance_normalized + 1e-10))
        max_entropy = torch.log(torch.tensor(float(n_features), device=self.device))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        
        # УЛУЧШЕНО: Линейный штраф за энтропию (максимальная стабильность)
        # Убираем экспоненту и другие сложные компоненты, оставляем только normalized entropy
        # Это гарантирует, что loss будет в диапазоне [0, 1] и градиенты не взорвутся
        sparsity_loss = normalized_entropy
        
        # Заглушки для совместимости с логгером
        cv = torch.tensor(0.0, device=self.device)
        normalized_cv = torch.tensor(0.0, device=self.device)
        max_mean_ratio = torch.tensor(0.0, device=self.device)
        
        # 2. CONSISTENCY регуляризация - согласованность локальных и глобальных значений
        # УЛУЧШЕНО: более сильное влияние через нормализованную MSE
        consistency_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if self.global_shap_importance is not None:
            # Конвертируем глобальную важность в тензор
            if isinstance(self.global_shap_importance, np.ndarray):
                global_importance_tensor = torch.tensor(
                    self.global_shap_importance,
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                global_importance_tensor = self.global_shap_importance
            
            # Нормализуем глобальную важность
            global_sum = torch.sum(global_importance_tensor)
            if global_sum > 1e-10:
                global_importance_tensor = global_importance_tensor / global_sum
            
            # УЛУЧШЕНО: Используем нормализованную MSE + KL divergence для более сильного влияния
            mse_loss = torch.mean((importance_normalized - global_importance_tensor) ** 2)
            
            # KL divergence для дополнительного штрафа за расхождение распределений
            kl_loss = torch.sum(
                importance_normalized * torch.log(
                    (importance_normalized + 1e-10) / (global_importance_tensor + 1e-10)
                )
            )
            
            # Комбинируем MSE и KL
            consistency_loss = 0.7 * mse_loss + 0.3 * kl_loss
        else:
            # Инициализируем глобальную важность
            self.global_shap_importance = importance_normalized.detach().clone().cpu().numpy()
        
        # УЛУЧШЕНО: Обновление глобальной важности с адаптивным learning rate
        self.global_shap_batch_count += 1
        if self.global_shap_batch_count % self.global_shap_update_frequency == 0:
            # Адаптивный learning rate: больше в начале, меньше в конце
            # Это позволяет глобальной важности быстро сходиться, но затем стабилизироваться
            progress = min(self.global_shap_batch_count / (self.global_shap_update_frequency * 10), 1.0)
            alpha_base = 0.15  # Базовый learning rate
            alpha = alpha_base * (1.0 - progress * 0.5)  # Уменьшаем от 0.15 до 0.075
            
            current_global = torch.tensor(
                self.global_shap_importance,
                dtype=torch.float32,
                device=self.device
            )
            
            # УЛУЧШЕНО: Используем взвешенное обновление с учетом текущей важности
            # Если текущая важность уже близка к глобальной - меньше обновляем
            similarity = 1.0 - torch.mean(torch.abs(importance_normalized.detach() - current_global))
            adaptive_alpha = alpha * (1.0 + similarity)  # Больше обновляем при большом расхождении
            
            new_global = (1 - adaptive_alpha) * current_global + adaptive_alpha * importance_normalized.detach()
            new_global = new_global / (torch.sum(new_global) + 1e-10)  # Нормализуем
            
            # УЛУЧШЕНО: Сглаживание для предотвращения резких скачков
            new_global = 0.9 * new_global + 0.1 * current_global
            
            self.global_shap_importance = new_global.cpu().numpy()
        
        # 3. STABILITY регуляризация - стабильность для похожих образцов
        # УЛУЧШЕНО: более сильное влияние через дисперсию важности по батчу
        stability_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if batch_size > 1:
            # УЛУЧШЕНО: Более эффективная формула stability
            # Вычисляем важность для каждого образца в батче отдельно
            importance_per_sample = torch.abs(grad_input) * torch.abs(batch_X)
            
            # Нормализуем важность для каждого образца
            importance_per_sample_norm = torch.softmax(
                importance_per_sample / (torch.max(importance_per_sample, dim=1, keepdim=True)[0] + 1e-10),
                dim=1
            )
            
            # УЛУЧШЕНО: Используем комбинацию дисперсии и среднего абсолютного отклонения
            # Дисперсия важности по батчу для каждого признака
            variance_per_feature = torch.var(importance_per_sample_norm, dim=0)  # [n_features]
            
            # Среднее абсолютное отклонение от среднего по батчу
            mean_per_feature = torch.mean(importance_per_sample_norm, dim=0, keepdim=True)
            mad_per_feature = torch.mean(torch.abs(importance_per_sample_norm - mean_per_feature), dim=0)
            
            # Комбинируем: дисперсия + MAD для более сильного влияния
            stability_loss = 0.6 * torch.mean(variance_per_feature) + 0.4 * torch.mean(mad_per_feature)
        
        # УЛУЧШЕНО: Адаптивное масштабирование компонентов регуляризации
        # Масштабируем компоненты относительно их типичных значений для балансировки
        
        # Нормализуем каждый компонент относительно его ожидаемого масштаба
        # Sparsity: обычно в диапазоне [0, 2-3] (exp penalty)
        sparsity_scaled = sparsity_loss / 2.0  # Нормализуем к ~1
        
        # Consistency: обычно в диапазоне [0, 1] (MSE + KL)
        consistency_scaled = consistency_loss / 1.0  # Уже нормализован
        
        # Stability: обычно в диапазоне [0, 0.1-0.5] (дисперсия)
        stability_scaled = stability_loss / 0.2  # Нормализуем к ~1
        
        # Комбинированная SHAP потеря с нормализованными компонентами
        total_shap_loss = (
            self.gamma_sparsity * sparsity_scaled +
            self.gamma_consistency * consistency_scaled +
            self.gamma_stability * stability_scaled
        )
        
        # ВАЖНО: тензор должен быть дифференцируемым!
        # Если total_shap_loss уже тензор с градиентами, используем его напрямую
        if not isinstance(total_shap_loss, torch.Tensor):
            total_shap_loss = torch.tensor(total_shap_loss, device=self.device, requires_grad=True)
        elif not total_shap_loss.requires_grad:
            total_shap_loss = total_shap_loss.requires_grad_(True)
        
        shap_loss_tensor = total_shap_loss
        
        # Сохраняем компоненты для логирования (детализированные значения)
        shap_components = {
            'total': float(shap_loss_tensor.detach().item()),
            'sparsity': float(sparsity_loss.detach().item()),
            'sparsity_scaled': float(sparsity_scaled.detach().item()),
            'consistency': float(consistency_loss.detach().item()),
            'consistency_scaled': float(consistency_scaled.detach().item()),
            'stability': float(stability_loss.detach().item()),
            'stability_scaled': float(stability_scaled.detach().item()),
            'entropy': float(entropy.detach().item()),
            'normalized_entropy': float(normalized_entropy.detach().item()),
            'cv': float(cv.detach().item()),
            'normalized_cv': float(normalized_cv.detach().item()),
            'max_mean_ratio': float(max_mean_ratio.detach().item())
        }
        
        # Логирование компонентов (периодически)
        if self.log_shap_components and hasattr(self, 'global_shap_batch_count'):
            if self.global_shap_batch_count % 50 == 0:
                self.logger.debug(
                    f"SHAP компоненты [батч {self.global_shap_batch_count}]: "
                    f"sparsity={shap_components['sparsity']:.4f}, "
                    f"consistency={shap_components['consistency']:.4f}, "
                    f"stability={shap_components['stability']:.4f}, "
                    f"total={shap_components['total']:.4f}, "
                    f"entropy={shap_components['normalized_entropy']:.4f}, "
                    f"cv={shap_components['normalized_cv']:.4f}"
                )
        
        return shap_loss_tensor, shap_components

