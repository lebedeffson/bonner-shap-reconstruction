"""
Менеджер ANFIS моделей для восстановления спектра нейтронов
"""

import time
import inspect
import importlib
import logging
import numpy as np
import torch
import mealpy
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from xanfis import BioAnfisRegressor
from src.utils.logger import get_logger

logging.getLogger("mealpy").setLevel(logging.ERROR)


class ANFISManager:
    """Менеджер для обучения ANFIS моделей восстановления спектра"""

    MF_CLASS_ALIASES = {
        'gaussmf': 'Gaussian',
        'gaussian': 'Gaussian',
        'gbellmf': 'GBell',
        'bellmf': 'Bell',
        'trimf': 'Triangular',
        'trapmf': 'Trapezoidal',
        'sigmf': 'Sigmoid',
    }

    OPTIMIZER_CLASS_PATHS = {
        'OriginalPSO': 'mealpy.swarm_based.PSO.OriginalPSO',
        'AIW_PSO': 'mealpy.swarm_based.PSO.AIW_PSO',
        'LDW_PSO': 'mealpy.swarm_based.PSO.LDW_PSO',
        'P_PSO': 'mealpy.swarm_based.PSO.P_PSO',
    }

    @staticmethod
    def _sanitize_predictions(y_pred, reference_shape=None, context=""):
        """
        Приведение предсказаний к корректному виду и очистка от NaN/Inf
        """
        y_pred = np.asarray(y_pred, dtype=float)

        if reference_shape is not None:
            if isinstance(reference_shape, tuple) and len(reference_shape) == 2:
                expected_cols = reference_shape[1]
                if y_pred.ndim == 1 and expected_cols > 0 and y_pred.size % expected_cols == 0:
                    y_pred = y_pred.reshape(-1, expected_cols)
            elif isinstance(reference_shape, tuple) and len(reference_shape) == 1 and y_pred.ndim == 2:
                y_pred = y_pred.ravel()

        if not np.isfinite(y_pred).all():
            logger = get_logger("anfis_shap.anfis_manager")
            logger.warning(f"Предсказания содержат NaN/Inf (контекст: {context}). Выполняю очистку.")
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        return y_pred

    def __init__(self, config):
        """
        Инициализация менеджера
        
        Args:
            config: Словарь конфигурации модели
        """
        self.config = config
        self.model_config = config['model']
        self.task_type = 'regression'  # Всегда регрессия для нашей задачи
        self.logger = get_logger("anfis_shap.anfis_manager")
        self.feature_names = self._resolve_feature_names(config)

    @staticmethod
    def _resolve_feature_names(config):
        dataset_config = config.get('dataset', {})
        feature_columns = dataset_config.get('feature_columns')
        if feature_columns:
            return list(feature_columns)
        feature_prefix = dataset_config.get('feature_prefix')
        feature_count = dataset_config.get('feature_count')
        if feature_prefix is not None and feature_count is not None:
            start = int(dataset_config.get('feature_index_start', 1))
            count = int(feature_count)
            return [f"{feature_prefix}{i}" for i in range(start, start + count)]
        return None

    def set_feature_names(self, feature_names):
        if feature_names is None:
            return
        names = list(feature_names)
        if len(names) == 0:
            return
        self.feature_names = names

    def create_model(self, verbose=True, input_dim=None, output_dim=None):
        """
        Создание модели ANFIS
        
        Args:
            verbose: Выводить ли информацию о процессе
            
        Returns:
            BioAnfisRegressor: Созданная модель
        """
        optim_value, optim_params_value = self._resolve_optimizer_for_xanfis(
            self.model_config['optim'],
            self.model_config['optim_params']
        )

        base_params = {
            'num_rules': self.model_config['num_rules'],
            'mf_class': self._normalize_mf_class(self.model_config['mf_class']),
            'vanishing_strategy': self.model_config.get('vanishing_strategy', 'prod'),
            'optim': optim_value,
            'optim_params': optim_params_value,
            'reg_lambda': self.model_config['reg_lambda'],
            'seed': self.model_config['seed'],
            'n_workers': self.model_config.get('n_workers', 4),
            'verbose': verbose
        }

        # Совместимость с разными версиями xanfis:
        # часть релизов не поддерживает n_workers и другие новые kwargs.
        try:
            sig = inspect.signature(BioAnfisRegressor.__init__)
            allowed = set(sig.parameters.keys())
            filtered_params = {k: v for k, v in base_params.items() if k in allowed}
        except (TypeError, ValueError):
            filtered_params = base_params

        model = BioAnfisRegressor(**filtered_params)

        # Если заранее известны размеры входа и выхода, строим сеть,
        # чтобы избежать ситуаций с model.network = None.
        if input_dim is not None and output_dim is not None:
            model.size_input = int(input_dim)
            model.size_output = int(output_dim)
            model.build_model()

        return model

    def _resolve_optimizer_for_xanfis(self, optim_name, optim_params):
        """
        Совместимость с mealpy>=3 / xanfis:
        передаём в xanfis готовый объект Optimizer, а не строку.
        """
        if not isinstance(optim_name, str):
            return optim_name, optim_params

        class_path = self.OPTIMIZER_CLASS_PATHS.get(optim_name)
        if class_path:
            try:
                module_name, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_name)
                opt_class = getattr(module, class_name)
                params = dict(optim_params) if isinstance(optim_params, dict) else {}
                params.setdefault('log_to', 'none')
                return opt_class(**params), None
            except Exception as exc:
                self.logger.warning(
                    f"Не удалось создать объект оптимизатора '{optim_name}' через class_path: {exc}. "
                    "Пробую fallback через mealpy.get_all_optimizers()."
                )

        try:
            all_opts = mealpy.get_all_optimizers(verbose=False)
            opt_class = all_opts.get(optim_name)
            if opt_class is None:
                self.logger.warning(f"Оптимизатор '{optim_name}' не найден в mealpy.get_all_optimizers(). Оставляю как строку.")
                return optim_name, optim_params
            params = optim_params if isinstance(optim_params, dict) else {}
            opt_instance = opt_class(**params)
            return opt_instance, None
        except Exception as exc:
            self.logger.warning(f"Не удалось создать объект оптимизатора '{optim_name}': {exc}. Оставляю как строку.")
            return optim_name, optim_params

    @classmethod
    def _normalize_mf_class(cls, mf_class):
        if not isinstance(mf_class, str):
            return mf_class
        return cls.MF_CLASS_ALIASES.get(mf_class, mf_class)

    @staticmethod
    def _log_and_clean_state(model, *, consequent_clip_enabled=False, consequent_clip_value=0.0):
        """
        Логирование и устранение NaN/Inf в параметрах модели.
        Возвращает статистику по очищенным параметрам.
        """
        if not hasattr(model, 'network') or model.network is None:
            logger = get_logger("anfis_shap.anfis_manager")
            logger.warning("Модель не имеет атрибута network. Пропускаю очистку.")
            return {}
        
        state_dict = model.network.state_dict()
        cleaned = {}
        report = {}

        clip_report = {}
        clip_value = float(consequent_clip_value) if consequent_clip_enabled else 0.0

        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor):
                cleaned[name] = tensor
                continue

            array = tensor.detach().cpu().numpy()
            mask = ~np.isfinite(array)
            if mask.any():
                nan_count = int(np.isnan(array).sum())
                posinf_count = int(np.isposinf(array).sum())
                neginf_count = int(np.isneginf(array).sum())
                total = int(mask.sum())

                sample_indices = np.argwhere(mask)
                max_examples = 10
                examples = []
                raw_array = tensor.detach().cpu().numpy()
                for coords in sample_indices[:max_examples]:
                    coords_tuple = tuple(int(c) for c in coords)
                    raw_value = raw_array[coords_tuple]
                    examples.append({
                        'index': coords_tuple,
                        'value': float(raw_value) if np.isfinite(raw_value) else str(raw_value)
                    })

                report[name] = {
                    'nan': nan_count,
                    'posinf': posinf_count,
                    'neginf': neginf_count,
                    'shape': array.shape,
                    'total_nonfinite': total,
                    'examples': examples
                }
                
                # Более умная замена NaN: используем медиану вместо нуля для коэффициентов
                if name == 'coeffs' and nan_count > 0:
                    # Для коэффициентов используем медиану ненулевых значений
                    finite_values = array[np.isfinite(array)]
                    if len(finite_values) > 0:
                        replacement_value = np.median(finite_values[finite_values != 0])
                        if not np.isfinite(replacement_value) or replacement_value == 0:
                            replacement_value = 0.001  # Малое значение по умолчанию
                    else:
                        replacement_value = 0.001
                    array = np.where(np.isnan(array), replacement_value, array)
                    array = np.where(np.isinf(array), np.sign(array) * replacement_value, array)
                else:
                    # Для других параметров используем стандартную замену
                    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

            if consequent_clip_enabled and name == 'coeffs' and clip_value > 0:
                abs_before = float(np.max(np.abs(array))) if array.size else 0.0
                clipped = np.clip(array, -clip_value, clip_value)
                n_clipped = int(np.sum(np.abs(array) > clip_value))
                if n_clipped > 0:
                    clip_report[name] = {
                        'n_coeff_clipped': n_clipped,
                        'max_coeff_abs_before': abs_before,
                        'max_coeff_abs_after': float(np.max(np.abs(clipped))) if clipped.size else 0.0,
                        'clip_value': clip_value,
                    }
                array = clipped

            cleaned_tensor = torch.from_numpy(array).to(tensor.device).type(tensor.dtype)
            cleaned[name] = cleaned_tensor

        if report:
            logger = get_logger("anfis_shap.anfis_manager")
            logger.warning("Выявлены нечисловые параметры после обучения ANFIS. Значения будут очищены.")
            for name, stats in report.items():
                nan = stats['nan']
                posinf = stats['posinf']
                neginf = stats['neginf']
                shape = stats['shape']
                total = stats.get('total_nonfinite', nan + posinf + neginf)
                logger.warning(f"{name}: NaN={nan}, +Inf={posinf}, -Inf={neginf}, shape={shape}, total={total}")
                examples = stats.get('examples', [])
                if examples and len(examples) <= 5:  # Показываем только если немного примеров
                    logger.debug(f"Примеры позиций с некорректными значениями для {name}:")
                    for ex in examples:
                        logger.debug(f"  index={ex['index']} value={ex['value']}")
            
            try:
                model.network.load_state_dict(cleaned, strict=False)
                logger.info("Состояние модели успешно очищено и обновлено")
            except Exception as e:
                logger.error(f"Ошибка при загрузке очищенного состояния: {e}")
                logger.warning("Модель может работать некорректно. Рекомендуется переобучение.")

        if clip_report:
            report['__coefficient_clipping__'] = clip_report
        return report

    def train_vanilla_model(self, X_train, y_train, X_test, y_test):
        """
        Обучение стандартной ANFIS модели
        
        Args:
            X_train: Тренировочные признаки (N, 10)
            y_train: Тренировочные целевые значения (N, 60)
            X_test: Тестовые признаки
            y_test: Тестовые целевые значения
            
        Returns:
            dict: Результаты обучения
        """
        self.logger.info("Обучение Vanilla ANFIS (Регрессия)...")
        self.logger.info(f"Размер train: {X_train.shape[0]} образцов")
        self.logger.info(f"Размер test: {X_test.shape[0]} образцов")
        
        start_time = time.time()

        # Преобразование в numpy массивы
        X_train_array = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        y_train_array = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train

        # Проверка и очистка входных данных от NaN/Inf
        if not np.isfinite(X_train_array).all():
            nan_count_x = int(np.isnan(X_train_array).sum())
            inf_count_x = int(np.isinf(X_train_array).sum())
            self.logger.warning(f"Обнаружено NaN/Inf во входных данных X: NaN={nan_count_x}, Inf={inf_count_x}. Выполняю очистку.")
            X_train_array = np.nan_to_num(X_train_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if not np.isfinite(y_train_array).all():
            nan_count_y = int(np.isnan(y_train_array).sum())
            inf_count_y = int(np.isinf(y_train_array).sum())
            self.logger.warning(f"Обнаружено NaN/Inf в целевых данных y: NaN={nan_count_y}, Inf={inf_count_y}. Выполняю очистку.")
            y_train_array = np.nan_to_num(y_train_array, nan=0.0, posinf=0.0, neginf=0.0)

        # Проверка на пустые или нулевые данные
        if X_train_array.size == 0 or y_train_array.size == 0:
            raise ValueError("Пустые данные для обучения!")
        
        if np.all(X_train_array == 0):
            raise ValueError("Все входные данные равны нулю!")
        
        if np.all(y_train_array == 0):
            raise ValueError("Все целевые данные равны нулю!")

        # Создание и инициализация модели
        model = self.create_model(
            verbose=True,
            input_dim=X_train_array.shape[1],
            output_dim=y_train_array.shape[1] if y_train_array.ndim > 1 else 1
        )
        
        self.logger.info("Начало обучения...")
        try:
            model.fit(X_train_array, y_train_array)
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {e}")
            # Проверяем состояние модели перед повторной попыткой
            if hasattr(model, 'network') and model.network is not None:
                self.logger.info("Попытка очистки состояния модели и повторного обучения...")
                # Очищаем состояние перед повторной попыткой
                state_dict = model.network.state_dict()
                cleaned_state = {}
                for name, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        array = tensor.detach().cpu().numpy()
                        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
                        cleaned_state[name] = torch.from_numpy(array).to(tensor.device).type(tensor.dtype)
                    else:
                        cleaned_state[name] = tensor
                model.network.load_state_dict(cleaned_state, strict=False)
            raise

        # Проверяем и чистим состояние модели сразу после обучения
        # Делаем это ДО получения предсказаний, чтобы избежать NaN в предсказаниях
        nonfinite_report = self._log_and_clean_state(
            model,
            consequent_clip_enabled=bool(self.model_config.get('consequent_clip_enabled', False)),
            consequent_clip_value=float(self.model_config.get('consequent_clip_value', 0.0) or 0.0),
        )
        
        # Дополнительная проверка: если слишком много NaN, предупреждаем
        total_params = sum(p.numel() for p in model.network.parameters())
        total_nonfinite = sum(stats.get('total_nonfinite', 0) for stats in nonfinite_report.values())
        if total_nonfinite > 0:
            nan_ratio = total_nonfinite / total_params if total_params > 0 else 0
            if nan_ratio > 0.05:  # Более 5% параметров содержат NaN
                self.logger.warning(f"ВНИМАНИЕ: Обнаружено {total_nonfinite} некорректных параметров из {total_params} ({nan_ratio*100:.2f}%)")
                self.logger.warning("Рекомендуется увеличить reg_lambda или уменьшить сложность модели")

        coeff_condition_threshold = float(self.model_config.get('coeff_condition_threshold', 1e8) or 1e8)
        coeff_tensor = model.network.state_dict().get('coeffs') if hasattr(model, "network") else None
        if isinstance(coeff_tensor, torch.Tensor):
            coeff_np = coeff_tensor.detach().cpu().numpy().astype(float)
            coeff_abs_max = float(np.max(np.abs(coeff_np))) if coeff_np.size else 0.0
            if np.isfinite(coeff_abs_max) and coeff_abs_max > coeff_condition_threshold:
                self.logger.warning(
                    "Коэффициенты consequents имеют аномально большой масштаб "
                    f"(max|coeff|={coeff_abs_max:.3e} > threshold={coeff_condition_threshold:.3e}). "
                    "Рекомендуется повысить reg_lambda / включить coefficient clipping."
                )
        
        training_time = time.time() - start_time

        # Получение предсказаний
        X_test_array = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
        y_pred = model.predict(X_test_array)
        y_test_array = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
        y_pred = self._sanitize_predictions(y_pred, reference_shape=y_test_array.shape, context="vanilla")
        
        # Обрезаем отрицательные значения (спектры не могут быть отрицательными)
        y_pred = np.clip(y_pred, a_min=0.0, a_max=None)
        
        metrics = self._calculate_metrics(y_test_array, y_pred)

        # Важность признаков
        feature_importance = self._extract_feature_importance(model, X_train_array.shape[1])

        results = {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'training_time': training_time,
            'nonfinite_report': nonfinite_report
        }

        self._print_results(results, "Vanilla ANFIS")
        return results

    def _calculate_metrics(self, y_true, y_pred):
        """
        Вычисление метрик для мультирегрессии
        
        Args:
            y_true: Истинные значения (N, 60)
            y_pred: Предсказанные значения (N, 60)
            
        Returns:
            dict: Словарь метрик
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = self._sanitize_predictions(y_pred, reference_shape=y_true.shape, context="metrics")

        logger = get_logger("anfis_shap.anfis_manager")
        if not np.isfinite(y_true).all():
            logger.warning("Истинные значения содержат NaN/Inf. Выполняю очистку.")
            y_true = np.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(y_pred).all():
            logger.warning("Предсказания содержат NaN/Inf после очистки. Дополнительная очистка.")
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        if y_pred.shape != y_true.shape:
            logger.warning(f"Предупреждение: формы не совпадают! y_true: {y_true.shape}, y_pred: {y_pred.shape}")
            # Пытаемся исправить
            min_samples = min(y_true.shape[0], y_pred.shape[0])
            min_features = min(y_true.shape[1] if y_true.ndim > 1 else 1, 
                              y_pred.shape[1] if y_pred.ndim > 1 else 1)
            y_true = y_true[:min_samples, :min_features] if y_true.ndim > 1 else y_true[:min_samples]
            y_pred = y_pred[:min_samples, :min_features] if y_pred.ndim > 1 else y_pred[:min_samples]
        
        mse = mean_squared_error(y_true, y_pred, multioutput='uniform_average')
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred, multioutput='uniform_average')

        variances = np.var(y_true, axis=0, ddof=0)
        mask = variances > 1e-12
        if mask.any():
            ss_res = np.sum((y_true[:, mask] - y_pred[:, mask]) ** 2, axis=0)
            means = np.mean(y_true[:, mask], axis=0)
            ss_tot = np.sum((y_true[:, mask] - means) ** 2, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                r2_per_output = 1.0 - ss_res / ss_tot
            r2_per_output = np.where(np.isfinite(r2_per_output), r2_per_output, 0.0)
            weights = variances[mask]
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                r2_weighted = float(np.sum(r2_per_output * weights) / weight_sum)
            else:
                r2_weighted = float(np.mean(r2_per_output))
            r2_mean = float(np.mean(r2_per_output))
        else:
            r2_weighted = 0.0
            r2_mean = 0.0

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2_weighted),
            'r2_weighted': float(r2_weighted),
            'r2_mean': float(r2_mean)
        }

    def _extract_feature_importance(self, model, n_features):
        """
        Извлечение важности признаков из модели
        
        Args:
            model: Обученная ANFIS модель
            n_features: Количество признаков
            
        Returns:
            np.array: Массив важности признаков
        """
        try:
            coefficients = model.network.state_dict()['coeffs'].detach().cpu().numpy()
            if np.isnan(coefficients).any() or np.isinf(coefficients).any():
                nan_count = int(np.isnan(coefficients).sum())
                inf_count = int(np.isinf(coefficients).sum())
                logger = get_logger("anfis_shap.anfis_manager")
                logger.warning(f"Коэффициенты модели содержат некорректные значения (NaN: {nan_count}, Inf: {inf_count}). Выполняю очистку.")
                coefficients = np.nan_to_num(coefficients, nan=0.0, posinf=0.0, neginf=0.0)
            # Для мультирегрессии берем первый выход (или среднее по всем выходам)
            if coefficients.ndim == 3:
                # Форма: [num_rules, num_features+1, num_outputs]
                # Берем среднее по всем выходам
                importance = np.mean(np.abs(coefficients[:, :-1, :]), axis=2)
                # Суммируем по правилам
                importance = np.sum(importance, axis=0)
            else:
                # Старый формат
                importance = np.sum(np.abs(coefficients[:, :-1, 0]), axis=0)
            importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            return importance
        except Exception as e:
            logger = get_logger("anfis_shap.anfis_manager")
            logger.warning(f"Не удалось извлечь важность признаков: {e}")
            return np.ones(n_features) / n_features

    def _print_results(self, results, model_name):
        """
        Вывод результатов обучения
        
        Args:
            results: Словарь с результатами обучения
            model_name: Название модели
        """
        metrics = results['metrics']
        
        self.logger.info(f"{model_name} обучен успешно!")
        self.logger.info(f"MSE: {metrics['mse']:.6f}")
        self.logger.info(f"RMSE: {metrics['rmse']:.6f}")
        self.logger.info(f"MAE: {metrics['mae']:.6f}")
        self.logger.info(f"R² (variance-weighted): {metrics['r2_weighted']:.6f}")
        self.logger.info(f"R² (mean across outputs): {metrics['r2_mean']:.6f}")
        self.logger.info(f"Время обучения: {results['training_time']:.2f} сек")

        nonfinite_report = results.get('nonfinite_report') or {}
        if nonfinite_report:
            self.logger.warning("В процессе обучения обнаружены нечисловые значения в параметрах (очищены):")
            for name, stats in nonfinite_report.items():
                self.logger.warning(
                    f"{name}: NaN={stats['nan']}, +Inf={stats['posinf']}, "
                    f"-Inf={stats['neginf']}, shape={stats['shape']}"
                )
        
        # Вывод важности признаков
        if 'feature_importance' in results:
            self.logger.info("Важность признаков:")
            names = self.feature_names
            if not names or len(names) != len(results['feature_importance']):
                names = [f"X{i+1}" for i in range(len(results['feature_importance']))]
            for name, imp in zip(names, results['feature_importance']):
                self.logger.info(f"{name}: {imp:.4f}")
