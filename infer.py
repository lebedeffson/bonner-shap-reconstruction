#!/usr/bin/env python3
"""
Инференс для обученной ANFIS модели:
загрузка модели, подача измерений, сохранение спектра (массив + картинка).
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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
from src.visualization.mpl_style import (
    COLORS,
    apply_axis_style,
    finalize_figure,
    resolve_energy_axis,
    setup_publication_style,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Инференс ANFIS модели")
    parser.add_argument("--config", required=True, help="Путь к конфигурации YAML")
    parser.add_argument("--model", required=True, help="Путь к .pt модели")
    parser.add_argument("--input", help="Строка значений признаков через запятую (в порядке колонок)")
    parser.add_argument("--input-csv", help="CSV файл с признаками")
    parser.add_argument("--output-dir", default="results/inference", help="Папка для результатов")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Устройство для инференса")
    parser.add_argument("--no-plot", action="store_true", help="Не сохранять графики")
    return parser.parse_args()


def _parse_input_values(raw: str):
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Пустой список значений для --input")
    return values


def _resolve_x_bins(n_bins: int):
    return resolve_energy_axis(n_bins)


def _plot_spectrum(pred, output_path, title="Predicted spectrum"):
    n_bins = pred.shape[-1]
    x_bins = _resolve_x_bins(n_bins)
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    ax.step(x_bins, pred, where="mid", linewidth=2.4, color=COLORS["pred"])
    ax.fill_between(x_bins, 0.0, pred, step="mid", alpha=0.16, color=COLORS["fill_pred"])
    ax.set_xlabel("Энергия, эВ")
    ax.set_ylabel("Плотность потока")
    ax.set_title(title)
    apply_axis_style(ax, log_x=bool(len(x_bins) == n_bins and np.all(x_bins > 0)))
    finalize_figure(fig, output_path)


def main():
    setup_publication_style()
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
    sums = None
    if normalize_sum:
        sums = np.sum(X_array, axis=1, keepdims=True)
        sums = np.where(sums == 0, 1e-12, sums)
        X_array = X_array / sums

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

    if args.device == "cuda" and torch.cuda.is_available():
        model.network = model.network.cuda()

    device = next(model.network.parameters()).device
    with torch.no_grad():
        X_tensor = torch.tensor(X_array, dtype=torch.float32, device=device)
        pred = model.network(X_tensor).cpu().numpy()

    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.clip(pred, a_min=0.0, a_max=None)

    pred_denorm = None
    if normalize_sum and sums is not None:
        pred_denorm = pred * sums

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    np.save(output_dir / f"predictions_{timestamp}.npy", pred)
    if pred_denorm is not None:
        np.save(output_dir / f"predictions_denorm_{timestamp}.npy", pred_denorm)

    pred_columns = resolve_target_columns_from_config(dataset_config) or [f"E{i+1}" for i in range(pred.shape[1])]
    pd.DataFrame(pred, columns=pred_columns).to_csv(output_dir / f"predictions_{timestamp}.csv", index=False)
    if pred_denorm is not None:
        pd.DataFrame(pred_denorm, columns=pred_columns).to_csv(output_dir / f"predictions_denorm_{timestamp}.csv", index=False)

    if not args.no_plot:
        for i in range(min(pred.shape[0], 3)):
            title = f"Predicted spectrum #{i+1}"
            _plot_spectrum(pred[i], output_dir / f"spectrum_{timestamp}_idx{i}.png", title=title)
            if pred_denorm is not None:
                _plot_spectrum(pred_denorm[i], output_dir / f"spectrum_{timestamp}_idx{i}_denorm.png", title=title + " (denorm)")

    print(f"✅ Инференс завершен. Результаты в {output_dir}")


if __name__ == "__main__":
    main()
