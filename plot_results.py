#!/usr/bin/env python3
"""Построение аккуратных графиков по сохранённым результатам обучения."""

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.data_loader import (
    resolve_feature_columns_from_config,
    resolve_target_columns_from_config,
)
from src.visualization.mpl_style import (
    COLORS,
    annotate_bar_values,
    apply_axis_style,
    finalize_figure,
    resolve_energy_axis,
    setup_publication_style,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Построение графиков по результатам обучения")
    parser.add_argument(
        "--summary",
        help="Путь к training_summary_*.json (если не указан, берётся последний из results*)"
    )
    parser.add_argument(
        "--output-dir",
        help="Каталог для сохранения графиков (по умолчанию тот же, что и у сводки)"
    )
    parser.add_argument(
        "--spectra-count",
        type=int,
        default=12,
        help="Сколько случайных спектров построить из полного набора (по умолчанию 12)"
    )
    parser.add_argument(
        "--spectra-dir",
        default="spectra",
        help="Подкаталог внутри output-dir для индивидуальных графиков спектров"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для случайного выбора спектров"
    )
    parser.add_argument(
        "--plot-style",
        choices=["line", "step"],
        default="step",
        help="Стиль построения спектров (по умолчанию ступенчатый график)"
    )
    parser.add_argument(
        "--no-log-x",
        action="store_false",
        dest="log_x",
        help="Отключить логарифмическую шкалу по оси энергий"
    )
    parser.set_defaults(log_x=True)
    parser.add_argument(
        "--report-file",
        help="Если указан, сохранить Markdown-отчёт с основными метриками и ссылками на графики"
    )
    return parser.parse_args()


def _find_latest_summary():
    candidates = []
    for directory in Path(".").glob("results*"):
        if directory.is_dir():
            candidates.extend(directory.glob("training_summary_*.json"))
    if not candidates:
        raise FileNotFoundError("Не найдено training_summary_*.json ни в одном results* каталоге")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(results_dir, saved_files):
    samples_info = saved_files.get("samples")
    if not samples_info:
        return None

    def _load(key):
        filename = samples_info.get(key)
        if not filename:
            return None
        path = results_dir / filename
        return np.load(path) if path.exists() else None

    return {
        "indices": samples_info.get("indices"),
        "X": _load("X"),
        "y": _load("y"),
        "pred": _load("pred"),
        "sum": _load("sum"),
    }


def load_predictions(results_dir, saved_files):
    predictions = saved_files.get("predictions")
    targets = saved_files.get("targets_test")
    predictions_denorm = saved_files.get("predictions_denorm")
    targets_denorm = saved_files.get("targets_denorm")

    def _load(filename):
        if not filename:
            return None
        path = results_dir / filename
        return np.load(path) if path.exists() else None

    return {
        "pred": _load(predictions),
        "target": _load(targets),
        "pred_denorm": _load(predictions_denorm),
        "target_denorm": _load(targets_denorm),
    }


def _get_model_label_from_summary(summary):
    src = str(summary.get("metrics_source", "")).lower()
    if src == "vanilla_real_only":
        return "Vanilla (real-only)"
    if src == "shap":
        return "Two-Stage SHAP + Tikhonov"
    return src or "Model"


def _load_plot_metadata(summary):
    metadata = {
        "feature_names": None,
        "target_names": None,
        "dataset_config": {},
    }

    config_path = summary.get("config_path")
    if not config_path:
        return metadata

    config_file = Path(config_path)
    if not config_file.exists():
        return metadata

    try:
        config = load_config(str(config_file))
    except Exception:
        return metadata

    dataset_config = config.get("dataset", {}) or {}
    metadata["dataset_config"] = dataset_config
    metadata["feature_names"] = resolve_feature_columns_from_config(dataset_config)
    metadata["target_names"] = resolve_target_columns_from_config(dataset_config)
    return metadata


def _sanitize_array(arr):
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _resolve_feature_names(metadata, n_features):
    feature_names = metadata.get("feature_names") or []
    if len(feature_names) == n_features:
        return feature_names
    return [f"Q{i+1}" for i in range(n_features)]


def _resolve_energy_axis(length, metadata=None):
    energies = resolve_energy_axis(length)
    if energies.shape[0] == length:
        return energies

    target_names = (metadata or {}).get("target_names") or []
    if len(target_names) == length:
        try:
            return np.asarray([float(name) for name in target_names], dtype=float)
        except (TypeError, ValueError):
            pass
    return np.arange(length, dtype=float)


def _plot_series(ax, x, y, *, label, color, style="step", linewidth=2.2, alpha=1.0, linestyle="-"):
    if style == "step":
        ax.step(x, y, where="mid", label=label, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
    else:
        ax.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)


def _fill_band(ax, x, low, high, *, color, alpha=0.18, label=None):
    ax.fill_between(x, low, high, step="mid", color=color, alpha=alpha, label=label)


def _format_scientific(value):
    if not np.isfinite(value):
        return "nan"
    if value == 0:
        return "0"
    if abs(value) >= 1e-2:
        return f"{value:.4f}"
    return f"{value:.2e}"


def _plot_single_spectrum_pair(truth, pred, energies, output_path, *, title, subtitle=None, style="step", log_x=True):
    truth = _sanitize_array(truth)
    pred = _sanitize_array(pred)
    residual = pred - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.1]},
    )
    ax_top, ax_bottom = axes

    _plot_series(ax_top, energies, truth, label="Истинный спектр", color=COLORS["true"], style=style)
    _plot_series(ax_top, energies, pred, label="Предсказанный спектр", color=COLORS["pred"], style=style, linestyle="--")
    ax_top.set_title(title)
    if subtitle:
        ax_top.text(
            0.01,
            0.97,
            subtitle,
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=COLORS["muted"],
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#E2E8F0", "alpha": 0.95},
        )
    ax_top.set_ylabel("Плотность потока")
    apply_axis_style(ax_top, log_x=log_x)
    ax_top.legend(loc="upper right")

    _plot_series(ax_bottom, energies, residual, label="Ошибка", color=COLORS["error"], style=style, linewidth=1.8)
    ax_bottom.axhline(0.0, color=COLORS["ink"], linewidth=1.0, linestyle=":")
    ax_bottom.set_ylabel("Pred - True")
    ax_bottom.set_xlabel("Энергия, эВ")
    apply_axis_style(ax_bottom, log_x=log_x)
    ax_bottom.text(
        0.01,
        0.92,
        f"MAE={mae:.4f} | RMSE={rmse:.4f}",
        transform=ax_bottom.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=COLORS["ink"],
    )

    finalize_figure(fig, output_path)
    return output_path


def plot_samples(output_dir, timestamp, samples, spectra_dir, metadata, style="step", log_x=True, model_label=None):
    if samples is None or samples["y"] is None or samples["pred"] is None:
        print("⚠️  Нет сохранённых подвыборок для построения спектров.")
        return []

    y_true = _sanitize_array(samples["y"])
    y_pred = _sanitize_array(samples["pred"])
    sum_values = samples.get("sum")
    indices = samples.get("indices") or range(len(y_true))
    energies = _resolve_energy_axis(y_true.shape[1], metadata)

    target_dir = output_dir / spectra_dir / "saved"
    target_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    for idx, (truth, pred, sample_id) in enumerate(zip(y_true, y_pred, indices)):
        title = f"Сохранённый пример спектра #{sample_id}"
        if model_label:
            title += f" — {model_label}"
        subtitle = None
        if sum_values is not None:
            subtitle = f"SUM={float(sum_values[idx]):.3f}"
        output_path = target_dir / f"saved_sample_{timestamp}_{sample_id}.png"
        _plot_single_spectrum_pair(
            truth,
            pred,
            energies,
            output_path,
            title=title,
            subtitle=subtitle,
            style=style,
            log_x=log_x,
        )
        figure_paths.append(output_path)

    return figure_paths


def _plot_importance_dataframe(fi_df, output_path, *, title, color):
    if "importance" not in fi_df.columns:
        return None

    fi = fi_df.copy()
    fi = fi.replace([np.inf, -np.inf], np.nan).dropna(subset=["importance"])
    if fi.empty:
        return None

    fi["importance"] = fi["importance"].astype(float)
    fi = fi.sort_values("importance", ascending=False)
    fi_plot = fi.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.52 * len(fi_plot))))
    ax.barh(fi_plot.index, fi_plot["importance"], color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Вклад")
    ax.set_ylabel("Признак")
    apply_axis_style(ax)
    annotate_bar_values(ax, orientation="horizontal", fmt="{:.4f}")

    total = float(np.sum(np.clip(fi["importance"].values, a_min=0.0, a_max=None)))
    top3 = float(np.sum(np.clip(fi["importance"].values[:3], a_min=0.0, a_max=None)))
    if total > 0:
        ax.text(
            0.99,
            0.03,
            f"Top-3 mass: {top3 / total:.3f}\nСумма: {total:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color=COLORS["muted"],
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#E2E8F0", "alpha": 0.96},
        )

    finalize_figure(fig, output_path)
    return output_path


def plot_feature_importance(results_dir, timestamp, output_dir):
    fi_path = results_dir / f"feature_importance_{timestamp}.csv"
    if not fi_path.exists():
        print("⚠️  Файл важности признаков не найден.")
        return None

    fi_df = pd.read_csv(fi_path, index_col=0)
    return _plot_importance_dataframe(
        fi_df,
        output_dir / f"feature_importance_{timestamp}.png",
        title="Важность признаков (Vanilla ANFIS)",
        color=COLORS["accent"],
    )


def plot_metrics(summary, output_dir):
    metrics = summary.get("metrics", {}) or {}
    vanilla_metrics = summary.get("metrics_vanilla", {}) or {}
    metrics_source = summary.get("metrics_source", "vanilla")

    if not metrics:
        return None

    metric_groups = [
        ("Ошибки", ["mse", "rmse", "mae"], {"mse": "MSE", "rmse": "RMSE", "mae": "MAE"}),
        ("Качество аппроксимации", ["r2_weighted", "r2_mean"], {"r2_weighted": "R² weighted", "r2_mean": "R² mean"}),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.suptitle(f"Метрики модели ({metrics_source})", y=1.03)
    for ax, (title, keys, labels_map) in zip(axes, metric_groups):
        labels = [labels_map[k] for k in keys if k in metrics]
        values = [metrics[k] for k in keys if k in metrics]
        bars = ax.bar(labels, values, color=[COLORS["true"], COLORS["pred"], COLORS["accent"]][: len(values)], alpha=0.92)
        ax.set_title(title)
        apply_axis_style(ax)
        annotate_bar_values(ax, fmt="{:.4f}")
        if title == "Качество аппроксимации":
            ax.set_ylim(0, max(1.0, max(values) * 1.18))
        else:
            ax.set_ylim(0, max(values) * 1.20 if values else 1.0)
    output_path = output_dir / f"metrics_{summary['timestamp']}.png"
    finalize_figure(fig, output_path)

    comparison_path = None
    if metrics_source == "shap" and vanilla_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        fig.suptitle("Сравнение Vanilla vs SHAP + Tikhonov", y=1.03)
        for ax, (title, keys, labels_map) in zip(axes, metric_groups):
            labels = [labels_map[k] for k in keys if k in metrics and k in vanilla_metrics]
            shap_vals = [metrics[k] for k in keys if k in metrics and k in vanilla_metrics]
            vanilla_vals = [vanilla_metrics[k] for k in keys if k in metrics and k in vanilla_metrics]
            x = np.arange(len(labels))
            width = 0.36
            ax.bar(x - width / 2, vanilla_vals, width, label="Vanilla", color=COLORS["muted"], alpha=0.85)
            ax.bar(x + width / 2, shap_vals, width, label="SHAP + Tikhonov", color=COLORS["pred"], alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_title(title)
            apply_axis_style(ax)
            ax.legend(loc="upper right")
            if title == "Качество аппроксимации" and shap_vals + vanilla_vals:
                ax.set_ylim(0, max(1.0, max(shap_vals + vanilla_vals) * 1.18))
        comparison_path = output_dir / f"metrics_comparison_{summary['timestamp']}.png"
        finalize_figure(fig, comparison_path)

    if comparison_path:
        return [output_path, comparison_path]
    return [output_path]


def plot_error_distribution(pred, target, timestamp, output_dir, suffix="", metadata=None):
    if pred is None or target is None:
        return []

    pred = _sanitize_array(pred)
    target = _sanitize_array(target)

    if pred.shape != target.shape:
        print("⚠️  Размерности предсказаний и истинных значений не совпадают, пропускаю графики ошибок.")
        return []

    errors = pred - target
    mae_per_bin = np.mean(np.abs(errors), axis=0)
    rmse_per_bin = np.sqrt(np.mean(errors ** 2, axis=0))
    bias_per_bin = np.median(errors, axis=0)
    energies = _resolve_energy_axis(mae_per_bin.shape[0], metadata)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )
    ax_top, ax_bottom = axes
    _plot_series(ax_top, energies, mae_per_bin, label="MAE", color=COLORS["true"], style="line")
    _plot_series(ax_top, energies, rmse_per_bin, label="RMSE", color=COLORS["pred"], style="line")
    ax_top.set_ylabel("Ошибка")
    ax_top.set_title("Ошибка по энергиям" + (" (денорм.)" if suffix else ""))
    apply_axis_style(ax_top, log_x=np.all(energies > 0))
    ax_top.legend(loc="upper right")

    _plot_series(ax_bottom, energies, bias_per_bin, label="Смещённость", color=COLORS["error"], style="line", linewidth=1.9)
    ax_bottom.axhline(0.0, color=COLORS["ink"], linestyle=":", linewidth=1.0)
    ax_bottom.set_xlabel("Энергия, эВ")
    ax_bottom.set_ylabel("Bias")
    apply_axis_style(ax_bottom, log_x=np.all(energies > 0))
    mae_path = output_dir / f"errors_energy_{timestamp}{suffix}.png"
    finalize_figure(fig, mae_path)

    flat_errors = errors.flatten()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(flat_errors, bins=70, alpha=0.85, color=COLORS["pred"], edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color=COLORS["ink"], linewidth=1.1, linestyle=":", label="0")
    ax.axvline(np.mean(flat_errors), color=COLORS["true"], linewidth=1.6, linestyle="--", label="mean")
    ax.axvline(np.median(flat_errors), color=COLORS["error"], linewidth=1.6, linestyle="-.", label="median")
    ax.set_title("Распределение ошибок" + (" (денорм.)" if suffix else ""))
    ax.set_xlabel("Predicted - True")
    ax.set_ylabel("Количество")
    apply_axis_style(ax)
    ax.legend(loc="upper right")
    hist_path = output_dir / f"errors_hist_{timestamp}{suffix}.png"
    finalize_figure(fig, hist_path)

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    flat_target = target.flatten()
    flat_pred = pred.flatten()
    hb = ax.hexbin(flat_target, flat_pred, gridsize=55, mincnt=1, cmap="YlOrRd")
    min_val = float(min(flat_target.min(), flat_pred.min()))
    max_val = float(max(flat_target.max(), flat_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color=COLORS["ink"], linestyle="--", linewidth=1.2, label="Идеал")
    rmse_flat = float(np.sqrt(np.mean((flat_pred - flat_target) ** 2)))
    mae_flat = float(np.mean(np.abs(flat_pred - flat_target)))
    ax.text(
        0.03,
        0.97,
        f"MAE={mae_flat:.4f}\nRMSE={rmse_flat:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#E2E8F0", "alpha": 0.96},
    )
    ax.set_xlabel("Истинные значения")
    ax.set_ylabel("Предсказания")
    ax.set_title("Плотность: истинные vs предсказанные" + (" (денорм.)" if suffix else ""))
    apply_axis_style(ax)
    ax.legend(loc="lower right")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Число точек")
    scatter_path = output_dir / f"scatter_{timestamp}{suffix}.png"
    finalize_figure(fig, scatter_path)

    return [mae_path, hist_path, scatter_path]


def plot_prediction_samples(pred, target, timestamp, output_dir, metadata, sample_size=5, suffix="", indices=None, seed=42, style="step", log_x=True, model_label=None):
    if pred is None or target is None:
        return [], None
    pred = _sanitize_array(pred)
    target = _sanitize_array(target)
    if pred.shape != target.shape:
        print("⚠️  Размерности предсказаний и истинных значений не совпадают, пропускаю сводные графики.")
        return [], None

    n_samples = pred.shape[0]
    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        indices = indices[(indices >= 0) & (indices < n_samples)]
    else:
        sample_size = min(sample_size, n_samples)
        if sample_size <= 0:
            return [], None
        rng = np.random.default_rng(seed)
        base_idx = np.linspace(0, n_samples - 1, min(sample_size, max(sample_size // 2, 1)), dtype=int)
        rand_idx = rng.choice(n_samples, size=sample_size, replace=False) if n_samples > sample_size else base_idx
        indices = np.unique(np.concatenate([base_idx, rand_idx]))
        if indices.size > sample_size:
            indices = indices[:sample_size]

    if indices.size == 0:
        return [], None

    subset_pred = pred[indices]
    subset_target = target[indices]
    energies = _resolve_energy_axis(pred.shape[1], metadata)

    target_med = np.median(subset_target, axis=0)
    pred_med = np.median(subset_pred, axis=0)
    target_q25, target_q75 = np.quantile(subset_target, [0.25, 0.75], axis=0)
    pred_q25, pred_q75 = np.quantile(subset_pred, [0.25, 0.75], axis=0)
    bias = np.median(subset_pred - subset_target, axis=0)
    abs_error = np.median(np.abs(subset_pred - subset_target), axis=0)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15]},
    )
    ax_top, ax_bottom = axes
    _fill_band(ax_top, energies, target_q25, target_q75, color=COLORS["fill_true"], label="Истинные IQR")
    _fill_band(ax_top, energies, pred_q25, pred_q75, color=COLORS["fill_pred"], label="Предсказания IQR")
    _plot_series(ax_top, energies, target_med, label="Истинные медиана", color=COLORS["true"], style=style)
    _plot_series(ax_top, energies, pred_med, label="Предсказания медиана", color=COLORS["pred"], style=style, linestyle="--")
    title = f"Сводка по {len(indices)} спектрам"
    if model_label:
        title += f" — {model_label}"
    ax_top.set_title(title)
    ax_top.set_ylabel("Плотность потока")
    apply_axis_style(ax_top, log_x=log_x)
    ax_top.legend(loc="upper right", ncol=2)

    _plot_series(ax_bottom, energies, abs_error, label="Медианный |error|", color=COLORS["accent"], style="line", linewidth=2.0)
    _plot_series(ax_bottom, energies, bias, label="Медианный bias", color=COLORS["error"], style="line", linewidth=1.8)
    ax_bottom.axhline(0.0, color=COLORS["ink"], linestyle=":", linewidth=1.0)
    ax_bottom.set_xlabel("Энергия, эВ")
    ax_bottom.set_ylabel("Ошибка")
    apply_axis_style(ax_bottom, log_x=log_x)
    ax_bottom.legend(loc="upper right")
    samples_path = output_dir / f"spectra_samples_{timestamp}{suffix}.png"
    finalize_figure(fig, samples_path)

    mean_true = target.mean(axis=0)
    mean_pred = pred.mean(axis=0)
    std_true = target.std(axis=0)
    std_pred = pred.std(axis=0)
    mean_bias = mean_pred - mean_true

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )
    ax_top, ax_bottom = axes
    _fill_band(ax_top, energies, mean_true - std_true, mean_true + std_true, color=COLORS["fill_true"], label="Истинные ±1σ")
    _fill_band(ax_top, energies, mean_pred - std_pred, mean_pred + std_pred, color=COLORS["fill_pred"], label="Предсказания ±1σ")
    _plot_series(ax_top, energies, mean_true, label="Средний истинный спектр", color=COLORS["true"], style=style)
    _plot_series(ax_top, energies, mean_pred, label="Средний предсказанный спектр", color=COLORS["pred"], style=style, linestyle="--")
    title_mean = "Средние спектры"
    if model_label:
        title_mean += f" — {model_label}"
    ax_top.set_title(title_mean)
    ax_top.set_ylabel("Плотность потока")
    apply_axis_style(ax_top, log_x=log_x)
    ax_top.legend(loc="upper right")

    _plot_series(ax_bottom, energies, mean_bias, label="Средний bias", color=COLORS["error"], style="line", linewidth=1.8)
    ax_bottom.axhline(0.0, color=COLORS["ink"], linestyle=":", linewidth=1.0)
    ax_bottom.set_xlabel("Энергия, эВ")
    ax_bottom.set_ylabel("Bias")
    apply_axis_style(ax_bottom, log_x=log_x)
    mean_path = output_dir / f"spectra_mean_{timestamp}{suffix}.png"
    finalize_figure(fig, mean_path)

    return [samples_path, mean_path], indices


def plot_individual_spectra(pred, target, timestamp, output_dir, metadata, count, suffix="", base_dir="spectra", seed=42, indices=None, style="step", log_x=True, model_label=None):
    if pred is None or target is None:
        return [], np.array([])
    pred = _sanitize_array(pred)
    target = _sanitize_array(target)
    if pred.shape != target.shape:
        print("⚠️  Размерности предсказаний и истинных значений не совпадают, пропускаю индивидуальные графики.")
        return [], np.array([])

    n_samples = pred.shape[0]
    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        indices = indices[(indices >= 0) & (indices < n_samples)]
    else:
        count = min(count, n_samples)
        if count <= 0:
            return [], np.array([])
        rng = np.random.default_rng(seed)
        base_idx = np.linspace(0, n_samples - 1, min(count, max(count // 2, 1)), dtype=int)
        rand_idx = rng.choice(n_samples, size=count, replace=False) if n_samples > count else base_idx
        indices = np.unique(np.concatenate([base_idx, rand_idx]))
        if indices.size > count:
            indices = indices[:count]

    if indices.size == 0:
        return [], np.array([])

    energies = _resolve_energy_axis(pred.shape[1], metadata)
    spectra_dir = output_dir / base_dir / ("denorm" if suffix else "normalized")
    spectra_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    for idx in indices:
        title = f"Спектр #{idx}"
        if model_label:
            title += f" — {model_label}"
        path = spectra_dir / f"spectrum_{timestamp}_idx{idx}{suffix}.png"
        _plot_single_spectrum_pair(
            target[idx],
            pred[idx],
            energies,
            path,
            title=title,
            style=style,
            log_x=log_x,
        )
        figure_paths.append(path)

    return figure_paths, indices


def _format_metric_block(block):
    lines = []
    if not block:
        return lines
    for key, value in block.items():
        try:
            numeric = float(value)
            lines.append(f"- **{key}**: {numeric:.6f}")
        except (TypeError, ValueError):
            lines.append(f"- **{key}**: {value}")
    return lines


def _format_band_block(block):
    lines = []
    if not block:
        return lines
    for band, metrics in block.items():
        lines.append(f"- **{band}**:")
        for key, value in metrics.items():
            try:
                numeric = float(value)
                lines.append(f"  - {key}: {numeric:.6f}")
            except (TypeError, ValueError):
                lines.append(f"  - {key}: {value}")
    return lines


def write_report(summary, figures_map, output_dir, report_file):
    if not report_file:
        return None

    report_path = Path(report_file)
    if not report_path.is_absolute():
        report_path = output_dir / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# Отчёт ANFIS — {summary.get('timestamp')}\n\n")
        f.write(f"- Конфигурация: `{summary.get('config_path')}`\n")
        f.write(f"- Тег запуска: `{summary.get('tag')}`\n")
        f.write(f"- Размер train/test: {summary.get('train_size')} / {summary.get('test_size')}\n")
        f.write(f"- Источник метрик: `{summary.get('metrics_source', 'vanilla')}`\n\n")

        metrics = summary.get("metrics")
        if metrics:
            f.write("## Метрики\n")
            for line in _format_metric_block(metrics):
                f.write(f"{line}\n")
            f.write("\n")

        metrics_denorm = summary.get("metrics_denorm")
        if metrics_denorm:
            f.write("## Метрики (денормализованные)\n")
            for line in _format_metric_block(metrics_denorm):
                f.write(f"{line}\n")
            f.write("\n")

        band_metrics = summary.get("band_metrics")
        if band_metrics:
            f.write("## Метрики по диапазонам (норм.)\n")
            for line in _format_band_block(band_metrics):
                f.write(f"{line}\n")
            f.write("\n")

        band_metrics_denorm = summary.get("band_metrics_denorm")
        if band_metrics_denorm:
            f.write("## Метрики по диапазонам (денорм.)\n")
            for line in _format_band_block(band_metrics_denorm):
                f.write(f"{line}\n")
            f.write("\n")

        diagnostics = summary.get("diagnostics", {})
        if diagnostics:
            f.write("## Диагностика\n")
            for key, stats in diagnostics.items():
                f.write(f"- **{key}**:\n")
                if isinstance(stats, dict):
                    for stat_name, stat_value in stats.items():
                        f.write(f"  - {stat_name}: {stat_value}\n")
                else:
                    f.write(f"  - {stats}\n")
            f.write("\n")

        if figures_map:
            f.write("## Графики\n")
            for category, paths in figures_map.items():
                if not paths:
                    continue
                f.write(f"### {category}\n")
                for path in paths:
                    rel = Path(path)
                    try:
                        rel = rel.relative_to(output_dir)
                    except ValueError:
                        pass
                    f.write(f"- ![{rel}]({rel})\n")
                f.write("\n")

    return report_path


def _extract_history_series(history, names):
    series = {}
    for name in names:
        values = history.get(name)
        if not isinstance(values, (list, tuple)):
            continue
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            continue
        series[name] = arr
    return series


def plot_shap_history(results_dir, shap_files, timestamp, output_dir):
    if not shap_files:
        return None

    history_path = shap_files.get("history")
    if not history_path:
        return None

    history_file = results_dir / history_path
    if not history_file.exists():
        return None

    with open(history_file, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, dict):
        return None

    groups = [
        ("Основные потери", ["total_loss", "main_loss", "shap_loss", "shap_loss_normalized", "tikhonov_loss"], False),
        ("Регуляризационные вклады", ["shap_contribution", "tikhonov_contribution"], False),
        ("Адаптивное расписание", ["adaptive_gamma", "convergence_slowdown", "regularization_share"], False),
        ("SHAP-компоненты", ["shap_consistency", "shap_sparsity", "shap_faithfulness", "shap_stability"], False),
        ("Веса компонент SHAP", ["shap_weight_consistency", "shap_weight_sparsity", "shap_weight_faithfulness", "shap_weight_stability"], True),
        ("Масштабирование SHAP", ["shap_scale_factor"], True),
    ]

    available_groups = []
    for title, keys, use_log_y in groups:
        series = _extract_history_series(history, keys)
        if series:
            available_groups.append((title, series, use_log_y))

    if not available_groups:
        return None

    n_panels = len(available_groups)
    n_cols = 2
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.0 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, series, use_log_y) in zip(axes, available_groups):
        for name, values in series.items():
            label = name.replace("_", " ")
            ax.plot(values, linewidth=1.8, label=label)
        ax.set_title(title)
        ax.set_xlabel("Эпоха")
        ax.set_ylabel("Значение")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        apply_axis_style(ax, log_y=use_log_y)
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        if title == "Регуляризационные вклады":
            ax.ticklabel_format(style="sci", axis="y", scilimits=(-2, 2))

    for ax in axes[n_panels:]:
        ax.axis("off")

    fig.suptitle("История SHAP-регуляризации", y=1.01)
    output_path = output_dir / f"shap_history_{timestamp}.png"
    finalize_figure(fig, output_path)
    return output_path


def plot_regularization_summary(summary, output_dir):
    reg = ((summary.get("diagnostics") or {}).get("regularization") or {})
    if not reg:
        return None

    component_terms = reg.get("component_terms") or {}
    weighted_signal = reg.get("weighted_component_signal") or {}
    shap_contribution = reg.get("shap_contribution") or {}
    tikhonov_contribution = reg.get("tikhonov_contribution") or {}
    regularization_share = reg.get("regularization_share") or {}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("Сводка по регуляризации", y=1.02)

    ax = axes[0, 0]
    contrib_labels = ["SHAP", "Tikhonov"]
    contrib_values = [float(shap_contribution.get("mean", 0.0)), float(tikhonov_contribution.get("mean", 0.0))]
    ax.bar(contrib_labels, contrib_values, color=[COLORS["pred"], COLORS["true"]], alpha=0.9)
    ax.set_title("Средний вклад в loss")
    ax.set_ylabel("Значение")
    apply_axis_style(ax)
    annotate_bar_values(ax, fmt="{:.2e}")

    ax = axes[0, 1]
    share_labels = ["mean", "last", "max"]
    share_values = [float(regularization_share.get(label, 0.0)) for label in share_labels]
    ax.bar(share_labels, share_values, color=COLORS["accent"], alpha=0.9)
    ax.set_title("Доля регуляризации в полном loss")
    ax.set_ylabel("Доля")
    apply_axis_style(ax)
    annotate_bar_values(ax, fmt="{:.3f}")

    ax = axes[1, 0]
    if component_terms:
        names = list(component_terms.keys())
        term_values = [float(component_terms[name].get("mean", 0.0)) for name in names]
        ax.bar(names, term_values, color=[COLORS["true"], COLORS["pred"], COLORS["accent"], COLORS["violet"]][: len(names)], alpha=0.9)
        ax.set_title("Средние raw-компоненты SHAP")
        ax.set_ylabel("Значение")
        apply_axis_style(ax, log_y=True)
        annotate_bar_values(ax, fmt="{:.2e}")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.axis("off")

    ax = axes[1, 1]
    if weighted_signal:
        names = list(weighted_signal.keys())
        signal_values = [float(weighted_signal[name].get("mean", 0.0)) for name in names]
        ax.bar(names, signal_values, color=[COLORS["true"], COLORS["pred"], COLORS["accent"], COLORS["violet"]][: len(names)], alpha=0.9)
        ax.set_title("Эффективный вклад компонент")
        ax.set_ylabel("Значение")
        apply_axis_style(ax)
        annotate_bar_values(ax, fmt="{:.2e}")
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.axis("off")

    active_components = reg.get("active_components") or []
    dominant_regularizer = reg.get("dominant_regularizer", "n/a")
    dominant_component = reg.get("dominant_shap_component", "n/a")
    fig.text(
        0.5,
        0.01,
        " | ".join(
            [
                f"Активные компоненты: {', '.join(active_components) if active_components else 'n/a'}",
                f"Доминирующий регуляризатор: {dominant_regularizer}",
                f"Доминирующий SHAP-компонент: {dominant_component}",
            ]
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLORS["muted"],
    )

    output_path = output_dir / f"regularization_summary_{summary['timestamp']}.png"
    finalize_figure(fig, output_path)
    return output_path


def plot_bonner_counts(output_dir, timestamp, samples, metadata, suffix=""):
    if samples is None or samples.get("X") is None:
        return []

    X = _sanitize_array(samples["X"])
    indices = samples.get("indices", range(len(X)))
    feature_names = _resolve_feature_names(metadata, X.shape[1])

    bonner_dir = output_dir / "bonner_counts"
    bonner_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    for x_row, sample_id in zip(X, indices):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        x_pos = np.arange(len(x_row))
        colors = [COLORS["true"] if value >= np.median(x_row) else COLORS["fill_true"] for value in x_row]
        ax.bar(x_pos, x_row, color=colors, alpha=0.92, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=25)
        ax.set_title(f"Измеренные отклики сфер Боннера — sample {sample_id}")
        ax.set_ylabel("Нормированная скорость счёта")
        ax.set_xlabel("Каналы")
        apply_axis_style(ax)
        annotate_bar_values(ax, fmt="{:.3f}")
        path = bonner_dir / f"bonner_{timestamp}_sample_{sample_id}{suffix}.png"
        finalize_figure(fig, path)
        figure_paths.append(path)

    return figure_paths


def main():
    args = parse_args()
    setup_publication_style()

    summary_path = Path(args.summary) if args.summary else _find_latest_summary()
    summary = load_summary(summary_path)
    metadata = _load_plot_metadata(summary)
    results_dir = summary_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Используем сводку: {summary_path}")
    print(f"📂 Каталог графиков: {output_dir}")

    saved_files = summary.get("saved_files", {})
    samples = load_samples(results_dir, saved_files)
    generated_figures = {}
    plot_config = {"style": args.plot_style, "log_x": args.log_x}
    model_label = _get_model_label_from_summary(summary)

    saved_figures = plot_samples(output_dir, summary["timestamp"], samples, args.spectra_dir, metadata, model_label=model_label, **plot_config)
    if saved_figures:
        print("🖼️  Графики сохранённой подвыборки:")
        for path in saved_figures:
            print(f"   {path}")
    generated_figures["saved_samples"] = saved_figures

    bonner_figs = plot_bonner_counts(output_dir, summary["timestamp"], samples, metadata)
    if bonner_figs:
        print("🖼️  Графики сфер Боннера:")
        for path in bonner_figs:
            print(f"   {path}")
    generated_figures["bonner_counts"] = bonner_figs

    predictions = load_predictions(results_dir, saved_files)
    error_figs = plot_error_distribution(predictions["pred"], predictions["target"], summary["timestamp"], output_dir, suffix="", metadata=metadata)
    if error_figs:
        print("🖼️  Графики ошибок (нормализованные):")
        for path in error_figs:
            print(f"   {path}")
    generated_figures["errors_normalized"] = error_figs

    indiv_norm_figs, selected_indices = plot_individual_spectra(
        predictions["pred"],
        predictions["target"],
        summary["timestamp"],
        output_dir,
        metadata,
        count=args.spectra_count,
        suffix="",
        base_dir=args.spectra_dir,
        seed=args.seed,
        model_label=model_label,
        **plot_config,
    )
    if indiv_norm_figs:
        print("🖼️  Отдельные спектры (нормализованные):")
        for path in indiv_norm_figs:
            print(f"   {path}")
    generated_figures["spectra_normalized"] = indiv_norm_figs

    sample_figs, indices = plot_prediction_samples(
        predictions["pred"],
        predictions["target"],
        summary["timestamp"],
        output_dir,
        metadata,
        sample_size=args.spectra_count,
        suffix="",
        indices=selected_indices if selected_indices.size else None,
        seed=args.seed,
        model_label=model_label,
        **plot_config,
    )
    if not sample_figs:
        indices = np.array([])
    if sample_figs:
        print("🖼️  Сводные спектры (нормализованные):")
        for path in sample_figs:
            print(f"   {path}")
    generated_figures["summary_spectra_normalized"] = sample_figs

    error_denorm_figs = plot_error_distribution(predictions["pred_denorm"], predictions["target_denorm"], summary["timestamp"], output_dir, suffix="_denorm", metadata=metadata)
    if error_denorm_figs:
        print("🖼️  Графики ошибок (денормализованные):")
        for path in error_denorm_figs:
            print(f"   {path}")
    generated_figures["errors_denorm"] = error_denorm_figs

    indiv_denorm_figs, _ = plot_individual_spectra(
        predictions["pred_denorm"],
        predictions["target_denorm"],
        summary["timestamp"],
        output_dir,
        metadata,
        count=args.spectra_count,
        suffix="_denorm",
        base_dir=args.spectra_dir,
        seed=args.seed,
        indices=selected_indices if selected_indices.size else None,
        model_label=model_label,
        **plot_config,
    )
    if indiv_denorm_figs:
        print("🖼️  Отдельные спектры (денормализованные):")
        for path in indiv_denorm_figs:
            print(f"   {path}")
    generated_figures["spectra_denorm"] = indiv_denorm_figs

    sample_denorm_figs, _ = plot_prediction_samples(
        predictions["pred_denorm"],
        predictions["target_denorm"],
        summary["timestamp"],
        output_dir,
        metadata,
        sample_size=args.spectra_count,
        suffix="_denorm",
        indices=selected_indices if selected_indices.size else None,
        seed=args.seed,
        model_label=model_label,
        **plot_config,
    )
    if sample_denorm_figs:
        print("🖼️  Сводные спектры (денормализованные):")
        for path in sample_denorm_figs:
            print(f"   {path}")
    generated_figures["summary_spectra_denorm"] = sample_denorm_figs

    metrics_figs = plot_metrics(summary, output_dir)
    if metrics_figs:
        print("🖼️  Метрики модели:")
        for path in metrics_figs:
            print(f"   {path}")
    generated_figures["metrics"] = metrics_figs

    fi_figure = plot_feature_importance(results_dir, summary["timestamp"], output_dir)
    if fi_figure:
        print(f"🖼️  Диаграмма важности признаков: {fi_figure}")
        generated_figures.setdefault("feature_importance", []).append(fi_figure)

    shap_files = saved_files.get("shap")
    if shap_files and "feature_importance_shap" in shap_files:
        shap_fi_path = results_dir / shap_files["feature_importance_shap"]
        if shap_fi_path.exists():
            fi_df = pd.read_csv(shap_fi_path, index_col=0)
            shap_fig_path = _plot_importance_dataframe(
                fi_df,
                output_dir / f"feature_importance_shap_{summary['timestamp']}.png",
                title="Важность признаков (SHAP)",
                color=COLORS["pred"],
            )
            if shap_fig_path:
                print(f"🖼️  SHAP важность признаков: {shap_fig_path}")
                generated_figures.setdefault("feature_importance_shap", []).append(shap_fig_path)

    shap_history_fig = plot_shap_history(results_dir, shap_files, summary["timestamp"], output_dir)
    if shap_history_fig:
        print(f"🖼️  История SHAP: {shap_history_fig}")
        generated_figures.setdefault("shap_history", []).append(shap_history_fig)

    regularization_fig = plot_regularization_summary(summary, output_dir)
    if regularization_fig:
        print(f"🖼️  Сводка регуляризации: {regularization_fig}")
        generated_figures.setdefault("regularization", []).append(regularization_fig)

    report_path = write_report(summary, generated_figures, output_dir, args.report_file)
    if report_path:
        print(f"\n📝 Markdown-отчёт: {report_path}")

    print("\n✅ Построение графиков завершено.")


if __name__ == "__main__":
    main()
