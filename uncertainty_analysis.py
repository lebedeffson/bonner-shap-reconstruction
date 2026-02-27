#!/usr/bin/env python3
"""
Monte Carlo анализ устойчивости: добавляем гауссов шум к Q и
получаем распределение восстановленных спектров.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config
from src.utils.data_loader import (
    load_data,
    resolve_feature_columns,
    resolve_feature_columns_from_config,
    resolve_target_columns_from_config,
    resolve_target_count,
)
from src.utils.uncertainty_estimation import UncertaintyEstimator
from constants import Ebins_float_IAEA_Comp


def parse_args():
    parser = argparse.ArgumentParser(description="Monte Carlo устойчивость (ошибка измерений)")
    parser.add_argument("--config", required=True, help="Путь к конфигурации YAML")
    parser.add_argument("--model", required=True, help="Путь к .pt модели")
    parser.add_argument("--input", help="Строка значений признаков через запятую")
    parser.add_argument("--input-csv", help="CSV файл с признаками")
    parser.add_argument("--n-samples", type=int, default=1000, help="Число MC прогонов")
    parser.add_argument("--error-percent", type=float, default=1.0, help="Стд ошибки в процентах")
    parser.add_argument("--error-percent-list", help="Список ошибок через запятую (например: 0.5,1,2,5)")
    parser.add_argument("--error-percent-range", help="Диапазон ошибок start:stop:step (например: 0.5:5:0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--output-dir", default="results/uncertainty", help="Папка для результатов")
    parser.add_argument("--plot-each", action="store_true", help="Сохранять графики для каждого уровня ошибки")
    return parser.parse_args()


def _parse_input_values(raw: str):
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Пустой список значений для --input")
    return values


def _resolve_x_bins(n_bins: int):
    if len(Ebins_float_IAEA_Comp) == n_bins + 1:
        return np.asarray(Ebins_float_IAEA_Comp[:-1], dtype=float)
    if len(Ebins_float_IAEA_Comp) == n_bins:
        return np.asarray(Ebins_float_IAEA_Comp, dtype=float)
    return np.arange(n_bins)


def _plot_uncertainty(base, mean, std, p5, p95, output_path, title="Uncertainty"):
    n_bins = base.shape[-1]
    x_bins = _resolve_x_bins(n_bins)

    plt.figure(figsize=(10, 5))
    plt.step(x_bins, base, where="post", label="Base", linewidth=2)
    plt.step(x_bins, mean, where="post", label="Mean", linewidth=2, linestyle="--")
    plt.fill_between(x_bins, p5, p95, alpha=0.2, label="P5-P95")
    plt.fill_between(x_bins, mean - std, mean + std, alpha=0.2, label="±1σ")

    if len(x_bins) == n_bins and np.all(x_bins > 0):
        plt.xscale("log")
    plt.xlabel("Energy bin")
    plt.ylabel("Flux density")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def _parse_error_percent_list(args):
    if args.error_percent_list:
        return [float(x.strip()) for x in args.error_percent_list.split(",") if x.strip()]
    if args.error_percent_range:
        parts = [p.strip() for p in args.error_percent_range.split(":")]
        if len(parts) != 3:
            raise ValueError("Формат --error-percent-range: start:stop:step")
        start, stop, step = map(float, parts)
        if step <= 0:
            raise ValueError("step должен быть > 0")
        values = []
        current = start
        while current <= stop + 1e-12:
            values.append(round(current, 6))
            current += step
        return values
    return [args.error_percent]


def main():
    args = parse_args()
    config = load_config(args.config)
    dataset_config = config.get("dataset", {})
    normalize_sum = dataset_config.get("normalize_sum", False)

    if not args.input and not args.input_csv:
        raise ValueError("Нужно указать --input или --input-csv")
    if args.input and args.input_csv:
        raise ValueError("Укажите только один из параметров: --input или --input-csv")

    if args.input_csv:
        data = load_data(args.input_csv, drop_index=True)
        feature_columns = resolve_feature_columns(data, dataset_config)
        X = data[feature_columns].copy()
    else:
        feature_columns = resolve_feature_columns_from_config(dataset_config)
        if not feature_columns:
            raise ValueError("Не удалось определить порядок признаков из конфигурации. Укажите feature_columns или feature_prefix/feature_count.")
        values = _parse_input_values(args.input)
        if len(values) != len(feature_columns):
            raise ValueError(f"Ожидалось {len(feature_columns)} значений, получено {len(values)}")
        X = pd.DataFrame([values], columns=feature_columns)

    X_array = np.asarray(X, dtype=float)

    output_dim = resolve_target_count(dataset_config)
    if output_dim is None:
        target_cols = resolve_target_columns_from_config(dataset_config)
        output_dim = len(target_cols) if target_cols else 60

    manager = ANFISManager(config)
    model = manager.create_model(
        input_dim=X_array.shape[1],
        output_dim=output_dim,
        verbose=False
    )

    state_dict = torch.load(args.model, map_location="cpu")
    model.network.load_state_dict(state_dict, strict=False)
    model.network.eval()

    estimator = UncertaintyEstimator(model, device=next(model.network.parameters()).device, verbose=True)
    error_levels = _parse_error_percent_list(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_rows = []
    for error_percent in error_levels:
        results = estimator.estimate_uncertainty(
            X_array,
            n_samples=args.n_samples,
            error_percent=error_percent,
            seed=args.seed,
            normalize_sum=normalize_sum
        )
        metrics = estimator.compute_uncertainty_metrics(results)
        # Нормализуем имена для сводного графика
        metrics_standard = {
            "mean_std": metrics.get("mean_uncertainty"),
            "max_std": metrics.get("max_uncertainty"),
            "cv": metrics.get("coefficient_of_variation"),
            "ci_width": metrics.get("mean_ci_width"),
        }
        metrics_standard.update(metrics)
        metrics_standard["error_percent"] = float(error_percent)
        summary_rows.append(metrics_standard)

        # Сохраняем результаты в поддиректорию, чтобы не перезаписывать
        subdir = output_dir / f"error_{error_percent:.3g}"
        subdir.mkdir(parents=True, exist_ok=True)

        np.save(subdir / f"all_predictions_{timestamp}.npy", results["all_predictions"])
        np.save(subdir / f"mean_{timestamp}.npy", results["mean"])
        np.save(subdir / f"std_{timestamp}.npy", results["std"])
        np.save(subdir / f"base_{timestamp}.npy", results["base_prediction"])

        percentiles = results["percentiles"]
        np.savez(
            subdir / f"percentiles_{timestamp}.npz",
            p5=percentiles[5],
            p25=percentiles[25],
            p50=percentiles[50],
            p75=percentiles[75],
            p95=percentiles[95],
        )

        with open(subdir / f"metrics_{timestamp}.txt", "w", encoding="utf-8") as f:
            for k, v in metrics_standard.items():
                f.write(f"{k}: {v}\n")

        if args.plot_each:
            base = results["base_prediction"][0]
            mean = results["mean"][0]
            std = results["std"][0]
            p5 = percentiles[5][0]
            p95 = percentiles[95][0]
            _plot_uncertainty(
                base,
                mean,
                std,
                p5,
                p95,
                subdir / f"uncertainty_{timestamp}.png",
                title=f"Uncertainty (error={error_percent}%, N={args.n_samples})"
            )

    # Сводная таблица и график зависимости разброса от ошибки
    summary_df = pd.DataFrame(summary_rows).sort_values("error_percent")
    summary_path = output_dir / f"uncertainty_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)

    # График: mean_std, max_std, cv, ci_width vs error
    plt.figure(figsize=(10, 6))
    plt.plot(summary_df["error_percent"], summary_df["mean_std"], marker="o", label="mean_std")
    plt.plot(summary_df["error_percent"], summary_df["max_std"], marker="o", label="max_std")
    plt.plot(summary_df["error_percent"], summary_df["cv"], marker="o", label="cv")
    plt.plot(summary_df["error_percent"], summary_df["ci_width"], marker="o", label="ci_width")
    plt.xlabel("Error percent (%)")
    plt.ylabel("Uncertainty metric")
    plt.title(f"Uncertainty vs error (N={args.n_samples})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"uncertainty_summary_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Устойчивость рассчитана. Сводка в {summary_path}")


if __name__ == "__main__":
    main()
