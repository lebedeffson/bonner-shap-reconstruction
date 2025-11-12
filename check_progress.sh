#!/bin/bash
# Скрипт для проверки прогресса подбора гиперпараметров

echo "=========================================="
echo "📊 СТАТУС ПОДБОРА ГИПЕРПАРАМЕТРОВ"
echo "=========================================="
echo ""

# Проверка процесса
if pgrep -f "hyperparameter_tuning.py" > /dev/null; then
    PID=$(pgrep -f "hyperparameter_tuning.py" | head -1)
    echo "✅ Процесс запущен (PID: $PID)"
    
    # Использование CPU и памяти
    ps -p $PID -o pid,pcpu,pmem,etime,cmd --no-headers | awk '{print "   CPU: "$2"% | Память: "$3"% | Время работы: "$4}'
else
    echo "❌ Процесс не найден"
fi

echo ""

# Проверка лог файла
if [ -f "results/hyperparameter_tuning_full.log" ]; then
    echo "📄 Лог файл: results/hyperparameter_tuning_full.log"
    echo "   Размер: $(du -h results/hyperparameter_tuning_full.log | cut -f1)"
    echo ""
    echo "📝 Последние строки лога:"
    tail -20 results/hyperparameter_tuning_full.log | grep -E "(Комбинация|R²|✅|❌|🔍)" || tail -10 results/hyperparameter_tuning_full.log
else
    echo "⚠️  Лог файл не найден"
fi

echo ""

# Проверка результатов
CSV_FILES=$(ls -t results/hyperparameter_tuning_*.csv 2>/dev/null | head -1)
if [ -n "$CSV_FILES" ]; then
    echo "📊 Результаты CSV:"
    echo "   Файл: $CSV_FILES"
    
    if command -v python3 &> /dev/null; then
        TOTAL=$(tail -n +2 "$CSV_FILES" | wc -l)
        SUCCESS=$(tail -n +2 "$CSV_FILES" | grep -c ",True," || echo "0")
        echo "   Всего комбинаций: $TOTAL"
        echo "   Успешных: $SUCCESS"
        
        if [ "$SUCCESS" -gt 0 ]; then
            echo ""
            echo "🏆 ТОП-5 по R²:"
            python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$CSV_FILES')
    df_success = df[df['success'] == True].nlargest(5, 'r2')
    if len(df_success) > 0:
        cols = []
        if 'optim' in df_success.columns:
            cols.append('optim')
        cols.extend(['num_rules', 'mf_class', 'r2', 'mse', 'mae'])
        available_cols = [c for c in cols if c in df_success.columns]
        print(df_success[available_cols].to_string(index=False))
    else:
        print('   Нет успешных результатов')
except Exception as e:
    print(f'   Ошибка: {e}')
" 2>/dev/null || echo "   Не удалось прочитать результаты"
        fi
    fi
else
    echo "⚠️  CSV файлы с результатами еще не созданы"
fi

echo ""
echo "=========================================="

