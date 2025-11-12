"""
SHAP-регуляризованный тренер ANFIS для восстановления спектра нейтронов
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)


class ShapAwareANFISTrainer:
    """Тренер ANFIS с SHAP-регуляризацией для мультирегрессии"""

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
        self.grad_clip = float(shap_config.get('grad_clip', 5.0))
        self.negative_penalty = float(shap_config.get('negative_penalty', 0.1))

    def fit(self, X_train, y_train, epochs=25, batch_size=32, lr=0.005):
        """
        Обучение с SHAP-регуляризацией
        
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

        # Подготовка данных
        X_train_array = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        
        X_tensor = torch.tensor(X_train_array, dtype=torch.float32)
        y_tensor = torch.tensor(y_train_array, dtype=torch.float32)

        X_tensor = torch.nan_to_num(X_tensor)
        y_tensor = torch.nan_to_num(y_tensor)
        
        training_dataset = TensorDataset(X_tensor, y_tensor)
        data_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

        # Базовые значения для SHAP
        baseline_values = np.mean(X_train_array, axis=0)
        baseline_values = np.nan_to_num(baseline_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Оптимизатор и функция потерь
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_function = torch.nn.MSELoss()  # Для регрессии

        # История потерь
        history = {
            'total_loss': [],
            'main_loss': [],
            'shap_loss': []
        }

        if self.verbose:
            print(f"🟠 Начинаю обучение ANFIS с SHAP-регуляризацией (регрессия)...")
            print(f"   Эпох: {epochs}, Батч: {batch_size}, LR: {lr}, Gamma: {self.gamma}")

        for epoch in range(epochs):
            epoch_losses = {'total': [], 'main': [], 'shap': []}
            skipped_batches = 0

            for batch_X, batch_y in data_loader:
                # Очистка входных данных
                batch_X = torch.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)
                batch_y = torch.nan_to_num(batch_y, nan=0.0, posinf=0.0, neginf=0.0)
                
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
                    print(f"⚠️  Ошибка при прямом проходе в эпохе {epoch+1}: {e}")
                    skipped_batches += 1
                    continue
                
                # Очистка предсказаний от NaN/Inf
                predictions = torch.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Обрезаем отрицательные значения после предсказания (для стабильности)
                predictions = torch.clamp(predictions, min=0.0)
                
                # Проверка на валидность предсказаний
                if not torch.isfinite(predictions).all():
                    print(f"⚠️  Предсказания содержат NaN/Inf в эпохе {epoch+1}. Пропускаю батч.")
                    skipped_batches += 1
                    continue
                
                # Для мультирегрессии predictions может быть (batch, 60)
                # batch_y тоже (batch, 60)
                if predictions.shape != batch_y.shape:
                    # Если формы не совпадают, пытаемся исправить
                    min_dim = min(predictions.shape[-1], batch_y.shape[-1])
                    predictions = predictions[..., :min_dim]
                    batch_y = batch_y[..., :min_dim]
                
                main_loss = loss_function(predictions, batch_y)
                
                # Штраф за отрицательные предсказания
                if self.negative_penalty > 0:
                    negative_mask = predictions < 0
                    if negative_mask.any():
                        negative_penalty = torch.mean(torch.clamp(-predictions, min=0.0) ** 2)
                        main_loss = main_loss + self.negative_penalty * negative_penalty

                # SHAP регуляризация
                shap_importance = self._calculate_shap_approximation(batch_X, baseline_values)
                shap_importance = np.nan_to_num(shap_importance, nan=0.0, posinf=0.0, neginf=0.0)

                if shap_importance.ndim != 1 or shap_importance.size == 0:
                    shap_normalized = np.ones_like(baseline_values) / len(baseline_values)
                else:
                    shap_sum = float(np.sum(shap_importance))
                    if shap_sum <= 1e-12 or not np.isfinite(shap_sum):
                        shap_normalized = np.ones_like(shap_importance) / len(shap_importance)
                    else:
                        shap_normalized = shap_importance / shap_sum

                # Целевой профиль - равномерное распределение
                target_uniform = np.ones_like(shap_normalized) / len(shap_normalized)

                # SHAP потеря
                shap_regularization_loss = float(np.mean((shap_normalized - target_uniform) ** 2))

                shap_loss_tensor = torch.tensor(
                    shap_regularization_loss,
                    dtype=torch.float32,
                    device=batch_X.device if isinstance(batch_X, torch.Tensor) else X_tensor.device,
                    requires_grad=False
                )

                total_loss = main_loss + self.gamma * shap_loss_tensor

                # Обратное распространение
                if not torch.isfinite(total_loss):
                    print(f"⚠️  SHAP: total_loss содержит NaN/Inf в эпохе {epoch+1}. Пропускаю батч.")
                    skipped_batches += 1
                    continue

                try:
                    total_loss.backward()
                    
                    # Проверка градиентов на NaN/Inf перед обновлением
                    has_nan_grad = False
                    for param in self.model.parameters():
                        if param.grad is not None:
                            if not torch.isfinite(param.grad).all():
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        print(f"⚠️  Обнаружены NaN/Inf в градиентах в эпохе {epoch+1}. Пропускаю обновление.")
                        optimizer.zero_grad()  # Очищаем градиенты
                        skipped_batches += 1
                        continue
                    
                    if self.grad_clip and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    
                    optimizer.step()
                except Exception as e:
                    print(f"⚠️  Ошибка при обратном распространении в эпохе {epoch+1}: {e}")
                    optimizer.zero_grad()  # Очищаем градиенты при ошибке
                    skipped_batches += 1
                    continue

                # Сохранение потерь
                epoch_losses['total'].append(float(total_loss.item()))
                epoch_losses['main'].append(float(main_loss.item()))
                epoch_losses['shap'].append(float(shap_regularization_loss))

            # Усреднение потерь по эпохе
            for loss_type in history:
                loss_key = loss_type.split('_')[0]
                values = epoch_losses[loss_key]
                history[loss_type].append(float(np.mean(values)) if values else float('nan'))

            # Прогресс
            if self.verbose and (epoch + 1) % 5 == 0:
                skipped_info = f", Пропущено батчей: {skipped_batches}" if skipped_batches > 0 else ""
                print(f"   Эпоха {epoch + 1}/{epochs}: "
                      f"Total: {history['total_loss'][-1]:.6f}, "
                      f"Main: {history['main_loss'][-1]:.6f}, "
                      f"SHAP: {history['shap_loss'][-1]:.6f}{skipped_info}")
            
            if skipped_batches > 0:
                print(f"   ⚠️  В эпохе {epoch + 1} пропущено {skipped_batches} батчей из-за ошибок")

        self.training_time = time.time() - start_time
        if self.verbose:
            print(f"✅ Обучение завершено за {self.training_time:.2f} сек")

        return history

    def predict(self, X_test):
        """
        Получение предсказаний
        
        Args:
            X_test: Тестовые признаки
            
        Returns:
            np.array: Предсказания (обрезаны до неотрицательных значений)
        """
        self.model.eval()
        with torch.no_grad():
            X_test_array = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
            X_tensor = torch.tensor(X_test_array, dtype=torch.float32)
            predictions = self.model(X_tensor)
            # Обрезаем отрицательные значения (спектры не могут быть отрицательными)
            predictions = torch.clamp(predictions, min=0.0)
            predictions = predictions.cpu().numpy()
            return predictions

    def get_global_shap_importance(self, X_sample):
        """
        Глобальная важность признаков
        
        Args:
            X_sample: Выборка данных
            
        Returns:
            np.array: SHAP важность для каждого признака
        """
        X_sample_array = np.array(X_sample) if not isinstance(X_sample, np.ndarray) else X_sample
        baseline_values = np.mean(X_sample_array, axis=0)
        return self._calculate_shap_approximation(X_sample_array, baseline_values)

    def _calculate_shap_approximation(self, X_batch, baseline):
        """
        Приближенные SHAP значения
        
        Args:
            X_batch: Батч данных
            baseline: Базовые значения
            
        Returns:
            np.array: SHAP важность для каждого признака
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X_batch, torch.Tensor):
                X_tensor = torch.tensor(X_batch, dtype=torch.float32)
            else:
                X_tensor = X_batch

            X_tensor = X_tensor.to(next(self.model.parameters()).device)
            original_predictions = self.model(X_tensor).detach().cpu().numpy()

            # Для мультирегрессии берем среднее по всем выходам
            if original_predictions.ndim > 1:
                original_predictions = np.mean(original_predictions, axis=1)

            shap_values = []
            X_numpy = X_tensor.cpu().numpy()

            # Вычисляем важность каждого признака
            for feature_index in range(X_numpy.shape[1]):
                X_masked = X_numpy.copy()
                X_masked[:, feature_index] = baseline[feature_index]

                X_masked_tensor = torch.tensor(X_masked, dtype=torch.float32, device=X_tensor.device)
                masked_predictions = self.model(X_masked_tensor).detach().cpu().numpy()

                if masked_predictions.ndim > 1:
                    masked_predictions = np.mean(masked_predictions, axis=1)

                if np.isscalar(original_predictions) and np.isscalar(masked_predictions):
                    feature_importance = abs(float(original_predictions) - float(masked_predictions))
                else:
                    feature_importance = float(np.mean(np.abs(original_predictions - masked_predictions)))

                shap_values.append(feature_importance)

        return np.asarray(shap_values, dtype=float)

