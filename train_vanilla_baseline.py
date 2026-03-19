#!/usr/bin/env python3
"""Обучение чистого vanilla-baseline на тех же данных и split'ах, что и основной two-stage pipeline."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config
from src.utils.data_loader import (
    denormalize_predictions,
    load_training_dataset,
    load_validation_data,
    prepare_features_targets,
    split_data,
)
from train import (
    ENERGY_BANDS,
    REAL_DATA_SPLIT,
    REAL_TEST_FRACTION,
    _compute_band_metrics,
    _prepare_feature_importance,
    _split_real_data_for_shap,
    _to_serializable,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение vanilla-baseline для сравнения")
    parser.add_argument("--config", default="configs/config_integrated_shap.yaml", help="Путь к YAML конфигурации")
    parser.add_argument("--output-dir", default="results/vanilla_baseline", help="Куда сохранить артефакты")
    parser.add_argument("--tag", default="vanilla_baseline", help="Суффикс к timestamp")
    parser.add_argument("--train-limit", type=int, help="Переопределяет dataset.train_limit")
    parser.add_argument("--train-fraction", type=float, help="Переопределяет dataset.train_fraction")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_config = config["dataset"]
    normalize_sum = dataset_config.get("normalize_sum", False)

    if args.train_limit is not None:
        dataset_config["train_limit"] = args.train_limit
    if args.train_fraction is not None:
        dataset_config["train_fraction"] = args.train_fraction

    print("=" * 80)
    print("🤖 ОБУЧЕНИЕ VANILLA BASELINE")
    print("=" * 80)
    print(f"\n⚙️  Конфигурация: {args.config}")

    real_data_path = dataset_config.get("validation_data")
    if not real_data_path or not os.path.exists(real_data_path):
        raise FileNotFoundError(f"Файл с реальными данными не найден: {real_data_path}")

    print("\n📂 Загрузка реальных данных...")
    X_real, y_real, SUM_real = load_validation_data(
        real_data_path,
        normalize_sum=normalize_sum,
        dataset_config=dataset_config,
    )

    print("\n📂 Загрузка обучающих данных...")
    data = load_training_dataset(dataset_config)
    X, y, SUM_train = prepare_features_targets(
        data,
        normalize_sum=normalize_sum,
        dataset_config=dataset_config,
    )

    print("\n🔀 Разделение синтетических данных...")
    X_train, _, y_train, _ = split_data(
        X,
        y,
        test_size=dataset_config.get("test_size", 0.25),
        random_state=dataset_config.get("random_state", 42),
    )

    random_state = dataset_config.get("random_state", 42)
    (
        _X_real_shap,
        X_real_val,
        X_real_test,
        _y_real_shap,
        y_real_val,
        y_real_test,
        SUM_real_test,
    ) = _split_real_data_for_shap(
        X_real,
        y_real,
        SUM_real,
        normalize_sum=normalize_sum,
        random_state=random_state,
    )

    print(f"   ▶️ Vanilla train: {len(X_train)} synthetic samples")
    print(f"   ▶️ Real validation: {len(X_real_val)} samples")
    print(f"   ▶️ Real test: {len(X_real_test)} samples")

    X_train_array = np.nan_to_num(np.asarray(X_train, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_train_array = np.nan_to_num(np.asarray(y_train, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    X_real_val_array = np.nan_to_num(np.asarray(X_real_val, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_real_val_array = np.nan_to_num(np.asarray(y_real_val, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    X_real_test_array = np.nan_to_num(np.asarray(X_real_test, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y_real_test_array = np.nan_to_num(np.asarray(y_real_test, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    manager = ANFISManager(config)
    if hasattr(X_train, "columns"):
        manager.set_feature_names(X_train.columns)

    results = manager.train_vanilla_model(
        X_train_array,
        y_train_array,
        X_real_val_array,
        y_real_val_array,
    )

    print("\n🧪 Финальное тестирование vanilla на реальном test split...")
    vanilla_predictions = results["model"].predict(X_real_test_array)
    vanilla_predictions = manager._sanitize_predictions(
        vanilla_predictions,
        reference_shape=y_real_test_array.shape,
        context="vanilla_real_test",
    )
    vanilla_predictions = np.clip(vanilla_predictions, a_min=0.0, a_max=None)

    metrics = manager._calculate_metrics(y_real_test_array, vanilla_predictions)
    band_metrics = _compute_band_metrics(y_real_test_array, vanilla_predictions, ENERGY_BANDS)

    y_pred_denorm = None
    y_test_denorm = None
    metrics_denorm = None
    if normalize_sum and SUM_real_test is not None:
        y_pred_denorm = denormalize_predictions(vanilla_predictions, SUM_real_test)
        y_test_denorm = denormalize_predictions(y_real_test_array, SUM_real_test)
        y_pred_denorm = np.nan_to_num(np.asarray(y_pred_denorm, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        y_test_denorm = np.nan_to_num(np.asarray(y_test_denorm, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        metrics_denorm = manager._calculate_metrics(y_test_denorm, y_pred_denorm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_base = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = f"{timestamp_base}_{args.tag}" if args.tag else timestamp_base

    model_state_path = output_dir / f"anfis_model_state_{timestamp}.pt"
    print(f"\n💾 Сохранение модели: {model_state_path}")
    torch.save(results["model"].network.state_dict(), model_state_path)

    saved_files = {}

    predictions_path = output_dir / f"predictions_{timestamp}.npy"
    targets_path = output_dir / f"targets_test_{timestamp}.npy"
    np.save(predictions_path, np.asarray(vanilla_predictions, dtype=float))
    np.save(targets_path, np.asarray(y_real_test_array, dtype=float))
    saved_files["predictions"] = predictions_path.name
    saved_files["targets_test"] = targets_path.name

    if y_pred_denorm is not None:
        predictions_denorm_path = output_dir / f"predictions_denorm_{timestamp}.npy"
        targets_denorm_path = output_dir / f"targets_test_denorm_{timestamp}.npy"
        np.save(predictions_denorm_path, np.asarray(y_pred_denorm, dtype=float))
        np.save(targets_denorm_path, np.asarray(y_test_denorm, dtype=float))
        saved_files["predictions_denorm"] = predictions_denorm_path.name
        saved_files["targets_denorm"] = targets_denorm_path.name

    metrics_csv_path = output_dir / f"metrics_{timestamp}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
    saved_files["metrics_csv"] = metrics_csv_path.name

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"X{i+1}" for i in range(X_train_array.shape[1])]
    fi = _prepare_feature_importance(results["feature_importance"], feature_names, normalize=False)
    fi_path = output_dir / f"feature_importance_{timestamp}.csv"
    fi.to_csv(fi_path, header=["importance"])
    saved_files["feature_importance"] = fi_path.name

    prediction_stats = {
        "mean": float(np.mean(vanilla_predictions)),
        "std": float(np.std(vanilla_predictions)),
        "min": float(np.min(vanilla_predictions)),
        "max": float(np.max(vanilla_predictions)),
        "zero_fraction": float(np.mean(np.isclose(vanilla_predictions, 0.0))),
    }
    target_stats = {
        "mean": float(np.mean(y_real_test_array)),
        "std": float(np.std(y_real_test_array)),
        "min": float(np.min(y_real_test_array)),
        "max": float(np.max(y_real_test_array)),
    }

    summary = {
        "timestamp": timestamp,
        "tag": args.tag,
        "config_path": os.path.abspath(args.config),
        "model_state": model_state_path.name,
        "model_state_path": str(model_state_path),
        "train_size": int(X_train_array.shape[0]),
        "test_size": int(X_real_test_array.shape[0]),
        "vanilla_train_count": int(X_train_array.shape[0]),
        "real_test_count": int(X_real_test_array.shape[0]),
        "normalize_sum": normalize_sum,
        "metrics": metrics,
        "band_metrics": band_metrics,
        "metrics_source": "vanilla_real_only",
        "shap_config_enabled": False,
        "shap_applied": False,
        "training_time_total": results.get("training_time"),
        "saved_files": saved_files,
        "diagnostics": {
            "prediction_stats": prediction_stats,
            "target_stats": target_stats,
            "nonfinite_parameters": _to_serializable(results.get("nonfinite_report", {})),
            "regularization": {
                "active_components": [],
                "shap_contribution": {"mean": 0.0, "last": 0.0, "max": 0.0},
                "tikhonov_contribution": {"mean": 0.0, "last": 0.0, "max": 0.0},
                "regularization_share": {"mean": 0.0, "last": 0.0, "max": 0.0},
                "dominant_regularizer": "none",
            },
        },
        "dataset_settings": {
            "train_limit": dataset_config.get("train_limit"),
            "train_fraction": dataset_config.get("train_fraction"),
            "synthetic_test_size": dataset_config.get("test_size", 0.25),
            "test_size": REAL_TEST_FRACTION,
            "real_data_split": REAL_DATA_SPLIT,
            "random_state": random_state,
            "shap_uses_real_data_only": False,
            "test_uses_real_data_only": True,
        },
    }
    if metrics_denorm is not None:
        summary["metrics_denorm"] = metrics_denorm

    summary_path = output_dir / f"training_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(_to_serializable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"📄 Vanilla summary saved: {summary_path}")
    print("\n✅ Vanilla baseline завершён.")


if __name__ == "__main__":
    main()
