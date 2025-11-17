#!/bin/bash
# Обучение чистой ANFIS модели ТОЛЬКО на реальных данных (без SHAP)

cd /home/lebedeffson/PycharmProjects/ОИЯИ-ШАБД
source ~/venv/bin/activate

echo "🚀 Запуск обучения чистой ANFIS модели (только реальные данные, без SHAP)..."
echo ""

python train_vanilla_real_only.py \
    --config configs/config_vanilla_real_only.yaml \
    --tag vanilla_real_only \
    2>&1 | tee training_log_vanilla_real_only.txt

echo ""
echo "✅ Обучение завершено. Логи сохранены в training_log_vanilla_real_only.txt"

