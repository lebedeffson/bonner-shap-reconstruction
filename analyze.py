#!/usr/bin/env python3
"""Анализ результатов обучения ANFIS"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Просмотр результатов обучения ANFIS")
    parser.add_argument(
        "--summary",
        help="Путь к файлу training_summary_*.json (если не указано, выбирается последний)"
    )
    parser.add_argument(
        "--state",
        help="Путь к сохранённой модели anfis_model_state_*.pt (timestamp используется для поиска сводки)"
    )
    return parser.parse_args()


def _format_metrics(title, metrics):
    if not metrics:
        return
    print(f"\n📈 {title}:")
    for key, value in metrics.items():
        try:
            numeric = float(value)
            print(f"   {key.upper()}: {numeric:.6f}")
        except (TypeError, ValueError):
            print(f"   {key.upper()}: {value}")


def _choose_latest_summary():
    candidates = []
    for directory in Path(".").glob("results*"):
        if directory.is_dir():
            candidates.extend(directory.glob("training_summary_*.json"))
    if not candidates:
        raise FileNotFoundError("Не найдено training_summary_*.json ни в одной папке results*")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def analyze(state_path=None, summary_path=None):
    print("=" * 80)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
    print("=" * 80)

    summary_path = Path(summary_path) if summary_path else None
    state_path = Path(state_path) if state_path else None

    if summary_path and not summary_path.exists():
        raise FileNotFoundError(f"Файл сводки не найден: {summary_path}")

    if summary_path is None:
        if state_path is not None:
            timestamp = state_path.stem.split("_")[-1]
            summary_guess = state_path.parent / f"training_summary_{timestamp}.json"
            if summary_guess.exists():
                summary_path = summary_guess
            else:
                raise FileNotFoundError(f"Не удалось найти сводку для модели {state_path}")
        else:
            summary_path = _choose_latest_summary()

    results_dir = summary_path.parent

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    if state_path is None:
        model_state_path = summary.get("model_state_path")
        if model_state_path:
            model_state_path = Path(model_state_path)
            if not model_state_path.is_absolute():
                model_state_path = results_dir / model_state_path
            state_path = model_state_path if model_state_path.exists() else None
        else:
            model_state = summary.get("model_state")
            if model_state:
                candidate = results_dir / model_state
                state_path = candidate if candidate.exists() else None

    print(f"\n📄 Сводка: {summary_path}")
    if state_path:
        print(f"💾 Модель: {state_path}")

    config_path = summary.get("config_path")
    if config_path:
        print(f"⚙️  Конфигурация: {config_path}")
    if summary.get("tag"):
        print(f"🏷️  Тег запуска: {summary['tag']}")

    dataset_settings = summary.get("dataset_settings", {})
    if dataset_settings:
        print("\n🧪 Настройки датасета:")
        for key, value in dataset_settings.items():
            print(f"   {key}: {value}")

    if summary.get("training_time_total") is not None:
        print("\n⏱️  Время обучения:")
        print(f"   total: {summary['training_time_total']:.2f} c")
        if summary.get("training_time_vanilla") is not None:
            print(f"   vanilla: {summary['training_time_vanilla']:.2f} c")
        if summary.get("training_time_shap") is not None:
            print(f"   shap: {summary['training_time_shap']:.2f} c")

    metrics_source = summary.get("metrics_source", "vanilla")
    title = "Метрики (SHAP)" if metrics_source == "shap" else "Метрики (Vanilla)"
    _format_metrics(title, summary.get("metrics"))

    if summary.get("metrics_vanilla") and metrics_source == "shap":
        _format_metrics("Метрики (Vanilla до SHAP)", summary.get("metrics_vanilla"))

    if summary.get("metrics_shap"):
        _format_metrics("Метрики (SHAP)", summary.get("metrics_shap"))

    if summary.get("metrics_denorm"):
        _format_metrics("Метрики (денормализованные)", summary.get("metrics_denorm"))

    band_metrics = summary.get("band_metrics")
    if band_metrics:
        print("\n🔬 Метрики по диапазонам (нормализованные):")
        for band_name, values in band_metrics.items():
            print(f"   {band_name}:")
            for key, value in values.items():
                try:
                    numeric = float(value)
                    print(f"      {key.upper()}: {numeric:.6f}")
                except (TypeError, ValueError):
                    print(f"      {key.upper()}: {value}")

    band_metrics_denorm = summary.get("band_metrics_denorm")
    if band_metrics_denorm:
        print("\n🔬 Метрики по диапазонам (денормализованные):")
        for band_name, values in band_metrics_denorm.items():
            print(f"   {band_name}:")
            for key, value in values.items():
                try:
                    numeric = float(value)
                    print(f"      {key.upper()}: {numeric:.6f}")
                except (TypeError, ValueError):
                    print(f"      {key.upper()}: {value}")

    saved_files = summary.get("saved_files", {})
    if saved_files:
        print("\n📦 Сохранённые артефакты:")
        for key, value in saved_files.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")

    shap_config_enabled = summary.get("shap_config_enabled", False)
    shap_applied = summary.get("shap_applied", False)
    if shap_config_enabled:
        status = "активирован" if shap_applied else "включён, но не применён"
        print(f"\n🧭 SHAP: {status}")
        shap_files = summary.get("shap_files", {})
        if shap_files:
            for key, value in shap_files.items():
                print(f"   {key}: {value}")

    diagnostics = summary.get("diagnostics")
    if diagnostics:
        print("\n🔎 Диагностика:")
        for key, stats in diagnostics.items():
            print(f"   {key}:")
            for stat_name, stat_value in stats.items():
                print(f"      {stat_name}: {stat_value}")

    feature_file = results_dir / f"feature_importance_{summary['timestamp']}.csv"
    if feature_file.exists():
        print(f"\n📄 Важность признаков (vanilla): {feature_file}")

    metrics_csv = results_dir / f"metrics_{summary['timestamp']}.csv"
    if metrics_csv.exists():
        print(f"📄 Таблица метрик: {metrics_csv}")

    print("\n✅ Анализ завершён.")


if __name__ == "__main__":
    args = parse_args()
    analyze(state_path=args.state, summary_path=args.summary)

