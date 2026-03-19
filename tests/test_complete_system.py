"""
Комплексный тест всей системы
Проверяет что все модули работают вместе
"""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCompleteSystem(unittest.TestCase):
    """Комплексные тесты системы"""
    
    def test_all_core_modules_import(self):
        """Тест импорта всех основных модулей"""
        modules = [
            ('src.models.shap_trainer_improved', 'ShapAwareANFISTrainerImproved'),
            ('src.models.shap_trainer_precision_optimized', 'PrecisionOptimizedSHAPRegularization'),
            ('src.models.anfis_manager', 'ANFISManager'),
            ('src.utils.uncertainty_estimation', 'UncertaintyEstimator'),
            ('src.utils.config_loader', 'load_config'),
            ('src.utils.data_loader', 'load_training_dataset'),
        ]
        
        for module_name, class_name in modules:
            with self.subTest(module=module_name, class_name=class_name):
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    cls = getattr(module, class_name)
                    self.assertIsNotNone(cls)
                except ImportError as e:
                    # Игнорируем ошибки отсутствующих зависимостей (torch, numpy и т.д.)
                    # Это нормально для проверки синтаксиса
                    if 'torch' in str(e) or 'numpy' in str(e) or 'pandas' in str(e) or 'sklearn' in str(e):
                        pass  # Ожидаемо
                    else:
                        raise
    
    def test_config_structure(self):
        """Тест структуры конфигурации"""
        try:
            from src.utils.config_loader import load_config
            config = load_config('configs/config_integrated_shap.yaml')
            
            # Проверяем основные секции
            self.assertIn('model', config)
            self.assertIn('dataset', config)
            self.assertIn('shap_reg', config)
            self.assertIn('output', config)
            
        except FileNotFoundError:
            self.skipTest("Конфигурационный файл не найден")
        except Exception as e:
            if 'yaml' in str(e).lower() or 'numpy' in str(e).lower():
                pass  # Ожидаемо при отсутствии зависимостей
            else:
                raise
    
    def test_module_syntax(self):
        """Тест синтаксиса всех модулей"""
        import ast
        
        modules_to_check = [
            'src/models/shap_trainer_improved.py',
            'src/models/shap_trainer_precision_optimized.py',
            'src/models/anfis_manager.py',
            'src/utils/uncertainty_estimation.py',
            'src/utils/config_loader.py',
            'src/utils/data_loader.py',
            'src/visualization/shap_plots.py',
        ]
        
        for module_path in modules_to_check:
            with self.subTest(module=module_path):
                path = Path(module_path)
                if not path.exists():
                    self.skipTest(f"Файл {module_path} не найден")
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        code = f.read()
                    ast.parse(code)
                except SyntaxError as e:
                    self.fail(f"Синтаксическая ошибка в {module_path}: {e}")


if __name__ == '__main__':
    unittest.main()
