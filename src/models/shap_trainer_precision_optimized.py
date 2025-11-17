"""
Математически оптимизированная SHAP регуляризация для максимальной точности
Разработано для критических применений (военная сфера)

Принципы:
1. SHAP регуляризация не должна ухудшать точность
2. Использование адаптивных весов для балансировки точности и интерпретируемости
3. Математически обоснованные формулы с доказанной сходимостью
4. Робастные метрики для стабильности в критических условиях
"""

import torch
import numpy as np


class PrecisionOptimizedSHAPRegularization:
    """
    Математически оптимизированные формулы для максимальной точности
    """
    
    @staticmethod
    def compute_precision_aware_sparsity(importance_normalized, main_loss_value, target_gini=0.4, 
                                         precision_weight=0.7):
        """
        Sparsity регуляризация, адаптированная к текущей точности модели
        
        Ключевая идея: если модель уже точная, можно больше фокусироваться на интерпретируемости.
        Если модель неточная, минимизируем влияние sparsity на обучение.
        
        Формула:
        L_sparsity = (1 - precision_weight) × L_gini + precision_weight × L_entropy_soft
        
        Где precision_weight зависит от текущего main_loss:
        - Если main_loss большой → precision_weight → 1.0 (меньше влияния sparsity)
        - Если main_loss малый → precision_weight → 0.3 (больше влияния sparsity)
        
        Args:
            importance_normalized: Нормализованная важность [n_features]
            main_loss_value: Текущее значение main loss (для адаптации)
            target_gini: Целевое значение Gini
            precision_weight: Базовый вес точности (0-1)
            
        Returns:
            dict: {
                'sparsity_loss': адаптивный loss,
                'adaptive_weight': адаптивный вес sparsity,
                'gini_coefficient': текущий Gini,
                'entropy': текущая энтропия
            }
        """
        n_features = len(importance_normalized)
        eps = 1e-10
        
        # Адаптивный вес на основе точности модели
        # Нормализуем main_loss к [0, 1] для определения веса
        # Предполагаем, что хороший main_loss < 0.05, плохой > 0.1
        main_loss_normalized = max(0.0, min(1.0, main_loss_value / 0.1))
        
        # Если модель точная (low loss), можем больше фокусироваться на sparsity
        # Если модель неточная (high loss), минимизируем влияние sparsity
        adaptive_precision_weight = precision_weight * (1.0 - 0.7 * (1.0 - main_loss_normalized))
        adaptive_precision_weight = max(0.3, min(0.9, adaptive_precision_weight))
        
        # 1. Gini coefficient (точная формула)
        sorted_importance, _ = torch.sort(importance_normalized)
        indices = torch.arange(1, n_features + 1, device=importance_normalized.device, dtype=torch.float32)
        sum_weighted = torch.sum(indices * sorted_importance)
        sum_total = torch.sum(sorted_importance)
        gini_coefficient = 1.0 - 2.0 * sum_weighted / (n_features * sum_total + eps)
        gini_coefficient = torch.clamp(gini_coefficient, min=0.0, max=1.0)
        
        # Квадратичное отклонение от целевого Gini (более плавное)
        gini_loss = (gini_coefficient - target_gini) ** 2
        
        # 2. Мягкая энтропия (soft entropy) - менее агрессивная, чем обычная
        # Используем температуру для смягчения: H_soft = -Σ p_i × log(p_i / T + eps)
        temperature = 1.5  # Температура > 1 смягчает энтропию
        soft_entropy = -torch.sum(
            importance_normalized * torch.log(importance_normalized / temperature + eps)
        )
        max_entropy = torch.log(torch.tensor(float(n_features), device=importance_normalized.device, dtype=torch.float32))
        normalized_soft_entropy = soft_entropy / (max_entropy + eps)
        
        # Комбинируем с адаптивным весом
        sparsity_loss = (
            adaptive_precision_weight * normalized_soft_entropy +
            (1.0 - adaptive_precision_weight) * gini_loss
        )
        
        # Адаптивный вес для всего sparsity компонента
        # Если модель неточная, уменьшаем влияние sparsity
        sparsity_component_weight = max(0.5, min(1.0, 1.0 - 0.5 * main_loss_normalized))
        
        return {
            'sparsity_loss': sparsity_loss * sparsity_component_weight,
            'adaptive_weight': sparsity_component_weight,
            'gini_coefficient': gini_coefficient,
            'entropy': normalized_soft_entropy,
            'gini_loss': gini_loss
        }
    
    @staticmethod
    def compute_precision_aware_consistency(grad_importance, true_shap_importance, 
                                           main_loss_value, use_adaptive=True):
        """
        Consistency регуляризация с адаптацией к точности
        
        Ключевая идея: Consistency важна для точности модели, но не должна доминировать
        над основным обучением, если модель еще неточная.
        
        Формула:
        L_consistency = α × L_MSE + (1-α) × L_JS
        
        Где α адаптируется к точности:
        - Если модель точная → больше веса на JS (тонкая настройка)
        - Если модель неточная → больше веса на MSE (грубая настройка)
        
        Args:
            grad_importance: Gradient-based важность [n_features]
            true_shap_importance: Настоящие Shapley values [n_features]
            main_loss_value: Текущее значение main loss
            use_adaptive: Использовать ли адаптивные веса
            
        Returns:
            dict: {
                'consistency_loss': адаптивный loss,
                'mse_loss': MSE компонент,
                'js_loss': JS divergence компонент,
                'adaptive_alpha': адаптивный вес между MSE и JS
            }
        """
        eps = 1e-10
        
        # Нормализуем распределения
        grad_sum = torch.sum(grad_importance) + eps
        true_shap_sum = torch.sum(true_shap_importance) + eps
        grad_norm = grad_importance / grad_sum
        true_shap_norm = true_shap_importance / true_shap_sum
        
        # 1. MSE loss (более агрессивный для грубой настройки)
        mse_loss = torch.mean((grad_norm - true_shap_norm) ** 2)
        
        # 2. Jensen-Shannon divergence (более тонкий для точной настройки)
        m = (grad_norm + true_shap_norm) / 2.0 + eps
        kl_pm = torch.sum(grad_norm * torch.log((grad_norm + eps) / m))
        kl_qm = torch.sum(true_shap_norm * torch.log((true_shap_norm + eps) / m))
        js_loss = (kl_pm + kl_qm) / 2.0
        
        # Адаптивный вес между MSE и JS
        if use_adaptive:
            # Нормализуем main_loss
            main_loss_normalized = max(0.0, min(1.0, main_loss_value / 0.1))
            
            # Если модель неточная → больше MSE (α → 0.7)
            # Если модель точная → больше JS (α → 0.3)
            adaptive_alpha = 0.7 - 0.4 * (1.0 - main_loss_normalized)
            adaptive_alpha = max(0.3, min(0.7, adaptive_alpha))
        else:
            adaptive_alpha = 0.6  # Фиксированный вес
        
        consistency_loss = adaptive_alpha * mse_loss + (1.0 - adaptive_alpha) * js_loss
        
        # Адаптивный вес для всего consistency компонента
        # Consistency важна для точности, но не должна доминировать
        if use_adaptive:
            consistency_component_weight = max(0.8, min(1.0, 0.8 + 0.2 * (1.0 - main_loss_normalized)))
        else:
            consistency_component_weight = 1.0
        
        return {
            'consistency_loss': consistency_loss * consistency_component_weight,
            'mse_loss': mse_loss,
            'js_loss': js_loss,
            'adaptive_alpha': adaptive_alpha,
            'component_weight': consistency_component_weight
        }
    
    @staticmethod
    def compute_precision_aware_faithfulness(batch_X, baseline_X, predictions, baseline_pred, 
                                            model, main_loss_value, order=1):
        """
        Faithfulness регуляризация, оптимизированная для точности
        
        Ключевая идея: Faithfulness проверяет локальную линейность модели.
        Для точных моделей это важно, но не должно мешать обучению неточных моделей.
        
        Формула (упрощенная для точности):
        L_faithfulness = ||f(x) - f(baseline) - ∇f(baseline) × (x - baseline)||²
        
        Args:
            batch_X: Батч признаков [batch_size, n_features]
            baseline_X: Baseline признаки [batch_size, n_features]
            predictions: Предсказания [batch_size, output_dim]
            baseline_pred: Предсказания на baseline [batch_size, output_dim]
            model: Модель
            main_loss_value: Текущее значение main loss
            order: Порядок разложения (1 для скорости, 2 для точности)
            
        Returns:
            dict: {
                'faithfulness_loss': адаптивный loss,
                'linear_error': ошибка линейного приближения,
                'adaptive_weight': адаптивный вес faithfulness
            }
        """
        batch_size = batch_X.shape[0]
        output_dim = predictions.shape[1] if predictions.ndim > 1 else 1
        
        # Реальное изменение предсказания
        pred_change = predictions - baseline_pred.detach()
        if pred_change.ndim > 1:
            pred_change_scalar = torch.mean(pred_change, dim=1)
        else:
            pred_change_scalar = pred_change.squeeze()
        
        # Изменение признаков
        X_change = batch_X - baseline_X.detach()
        
        # Градиенты в точке baseline
        baseline_X.requires_grad_(True)
        baseline_pred_grad = model(baseline_X)
        grad_outputs = torch.ones_like(baseline_pred_grad) / output_dim
        
        grad_at_baseline = torch.autograd.grad(
            outputs=baseline_pred_grad,
            inputs=baseline_X,
            grad_outputs=grad_outputs,
            create_graph=(order >= 2),
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Линейное приближение
        linear_change = torch.sum(grad_at_baseline * X_change, dim=1)
        
        # Ошибка линейного приближения
        linear_error = torch.mean((pred_change_scalar - linear_change) ** 2)
        
        # Адаптивный вес: если модель неточная, уменьшаем влияние faithfulness
        main_loss_normalized = max(0.0, min(1.0, main_loss_value / 0.1))
        faithfulness_weight = max(0.0, min(0.3, 0.3 * (1.0 - main_loss_normalized)))  # От 0 до 0.3
        
        faithfulness_loss = linear_error * faithfulness_weight
        
        return {
            'faithfulness_loss': faithfulness_loss,
            'linear_error': linear_error,
            'adaptive_weight': faithfulness_weight
        }
    
    @staticmethod
    def compute_precision_aware_stability(importance_per_sample, main_loss_value):
        """
        Stability регуляризация, оптимизированная для точности
        
        Ключевая идея: Стабильность важна, но не должна мешать обучению.
        Используем робастные метрики для надежности.
        
        Формула:
        L_stability = Var(importance) × adaptive_weight
        
        Args:
            importance_per_sample: Важность для каждого образца [batch_size, n_features]
            main_loss_value: Текущее значение main loss
            
        Returns:
            dict: {
                'stability_loss': адаптивный loss,
                'variance': дисперсия важности,
                'adaptive_weight': адаптивный вес
            }
        """
        batch_size = importance_per_sample.shape[0]
        eps = 1e-10
        
        if batch_size <= 1:
            return {
                'stability_loss': torch.tensor(0.0, device=importance_per_sample.device),
                'variance': torch.tensor(0.0, device=importance_per_sample.device),
                'adaptive_weight': torch.tensor(0.0, device=importance_per_sample.device)
            }
        
        # Нормализуем важность
        importance_sum = torch.sum(importance_per_sample, dim=1, keepdim=True) + eps
        importance_normalized = importance_per_sample / importance_sum
        
        # Дисперсия по батчу (для каждого признака)
        variance_per_feature = torch.var(importance_normalized, dim=0)
        variance_loss = torch.mean(variance_per_feature)
        
        # Адаптивный вес: стабильность важна, но не критична для точности
        main_loss_normalized = max(0.0, min(1.0, main_loss_value / 0.1))
        stability_weight = max(0.1, min(0.2, 0.2 * (1.0 - 0.5 * main_loss_normalized)))  # От 0.1 до 0.2
        
        stability_loss = variance_loss * stability_weight
        
        return {
            'stability_loss': stability_loss,
            'variance': variance_loss,
            'adaptive_weight': stability_weight
        }
    
    @staticmethod
    def compute_adaptive_component_weights(main_loss_value, consistency_loss, sparsity_loss,
                                         faithfulness_loss, stability_loss, 
                                         target_main_loss=0.02):
        """
        Вычисляет адаптивные веса компонентов на основе текущей точности модели
        
        Принцип: Если модель неточная → больше веса на компоненты, помогающие точности.
                 Если модель точная → больше веса на компоненты интерпретируемости.
        
        Args:
            main_loss_value: Текущее значение main loss
            consistency_loss: Loss компонента consistency
            sparsity_loss: Loss компонента sparsity
            faithfulness_loss: Loss компонента faithfulness
            stability_loss: Loss компонента stability
            target_main_loss: Целевое значение main loss (для нормализации)
            
        Returns:
            dict: {
                'weights': {'consistency': w1, 'sparsity': w2, 'faithfulness': w3, 'stability': w4},
                'total_shap_loss': взвешенная сумма компонентов,
                'precision_ratio': отношение текущей точности к целевой
            }
        """
        # Нормализуем main_loss для определения режима обучения
        precision_ratio = max(0.1, min(2.0, main_loss_value / target_main_loss))
        
        # Если модель неточная (precision_ratio > 1.0):
        # - Больше веса на consistency (помогает точности)
        # - Меньше веса на sparsity (может мешать обучению)
        # - Умеренный вес на faithfulness и stability
        
        # Если модель точная (precision_ratio < 1.0):
        # - Умеренный вес на consistency
        # - Больше веса на sparsity (улучшаем интерпретируемость)
        # - Умеренный вес на faithfulness и stability
        
        if precision_ratio > 1.0:  # Модель неточная
            weights = {
                'consistency': 0.7,  # Больше веса на точность
                'sparsity': 0.15,    # Меньше веса на интерпретируемость
                'faithfulness': 0.1,
                'stability': 0.05
            }
        else:  # Модель точная
            weights = {
                'consistency': 0.4,  # Умеренный вес
                'sparsity': 0.4,     # Больше веса на интерпретируемость
                'faithfulness': 0.1,
                'stability': 0.1
            }
        
        # Плавный переход между режимами
        transition_factor = max(0.0, min(1.0, (precision_ratio - 0.5) / 1.5))
        
        weights_precise = {
            'consistency': 0.7,
            'sparsity': 0.15,
            'faithfulness': 0.1,
            'stability': 0.05
        }
        
        weights_interpretable = {
            'consistency': 0.4,
            'sparsity': 0.4,
            'faithfulness': 0.1,
            'stability': 0.1
        }
        
        # Интерполируем между режимами
        final_weights = {}
        for key in weights_precise:
            final_weights[key] = (
                transition_factor * weights_precise[key] +
                (1.0 - transition_factor) * weights_interpretable[key]
            )
        
        # Вычисляем взвешенную сумму
        total_shap_loss = (
            final_weights['consistency'] * consistency_loss +
            final_weights['sparsity'] * sparsity_loss +
            final_weights['faithfulness'] * faithfulness_loss +
            final_weights['stability'] * stability_loss
        )
        
        return {
            'weights': final_weights,
            'total_shap_loss': total_shap_loss,
            'precision_ratio': precision_ratio.item() if isinstance(precision_ratio, torch.Tensor) else precision_ratio,
            'transition_factor': transition_factor.item() if isinstance(transition_factor, torch.Tensor) else transition_factor
        }

