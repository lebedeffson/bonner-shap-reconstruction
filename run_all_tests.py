#!/usr/bin/env python3
"""
Запуск всех тестов и анализ результатов
"""

import unittest
import sys
import os
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

def run_tests():
    """Запуск всех тестов"""
    
    print("=" * 80)
    print("🧪 ЗАПУСК ВСЕХ ТЕСТОВ")
    print("=" * 80)
    
    # Находим все тесты
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавляем тесты
    test_modules = [
        'tests.test_data_loader',
        'tests.test_anfis_manager',
        'tests.test_integration',
        'tests.test_validation'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            print(f"✅ Загружены тесты из {module_name}")
        except ImportError as e:
            print(f"⚠️  Не удалось загрузить {module_name}: {e}")
    
    # Запускаем тесты
    print(f"\n{'='*80}")
    print("🚀 ЗАПУСК ТЕСТОВ")
    print(f"{'='*80}\n")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Итоги
    print(f"\n{'='*80}")
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print(f"{'='*80}")
    print(f"   Всего тестов: {result.testsRun}")
    print(f"   ✅ Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   ❌ Провалено: {len(result.failures)}")
    print(f"   ⚠️  Ошибок: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ ПРОВАЛЕННЫЕ ТЕСТЫ:")
        for test, traceback in result.failures:
            print(f"   - {test}")
    
    if result.errors:
        print(f"\n⚠️  ОШИБКИ:")
        for test, traceback in result.errors:
            print(f"   - {test}")
    
    # Общая оценка
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n📈 Успешность: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("   🎉 Все тесты пройдены!")
    elif success_rate >= 80:
        print("   ✅ Большинство тестов пройдено")
    else:
        print("   ⚠️  Требуется исправление ошибок")
    
    print("=" * 80)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)

