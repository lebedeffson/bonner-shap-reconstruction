#!/usr/bin/env python3
"""Построение графиков по сохранённым результатам ANFIS"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import Ebins_float_IAEA_Comp


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
        return np.load(results_dir / filename)

    return {
        "indices": samples_info.get("indices"),
        "X": _load("X"),
        "y": _load("y"),
        "pred": _load("pred"),
        "sum": _load("sum")
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
        "target_denorm": _load(targets_denorm)
    }


def plot_samples(output_dir, timestamp, samples, spectra_dir, style="step", log_x=True):
    if samples is None or samples["y"] is None or samples["pred"] is None:
        print("⚠️  Нет сохранённых подвыборок для построения спектров.")
        return []

    y_true = samples["y"]
    y_pred = samples["pred"]
    sum_values = samples.get("sum")
    indices = samples.get("indices") or range(len(y_true))

    energies = _get_energy_axis(y_true.shape[1])

    target_dir = output_dir / spectra_dir / "saved"
    target_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    for idx, (truth, pred, sample_id) in enumerate(zip(y_true, y_pred, indices)):
        fig, ax = plt.subplots(figsize=(10, 5))
        if style == "step":
            ax.step(energies, truth, label="Истинный спектр", linewidth=2, where="mid")
            ax.step(energies, pred, label="Предсказанный спектр", linewidth=2, linestyle="--", where="mid")
        else:
            ax.plot(energies, truth, label="Истинный спектр", linewidth=2)
            ax.plot(energies, pred, label="Предсказанный спектр", linewidth=2, linestyle="--")
        ax.set_xlabel("Energy, eV")
        ax.set_ylabel("phi, neutron cm^-2 s^-1")
        title = f"Сравнение спектров (sample {sample_id})"
        if sum_values is not None:
            title += f", SUM={sum_values[idx]:.3f}"
        ax.set_title(title)
        ax.legend()
        _apply_axis_style(ax, log_x)
        fig.tight_layout()

        output_path = target_dir / f"saved_sample_{timestamp}_{sample_id}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        figure_paths.append(output_path)

    return figure_paths


def plot_feature_importance(results_dir, timestamp):
    fi_path = results_dir / f"feature_importance_{timestamp}.csv"
    if not fi_path.exists():
        print("⚠️  Файл важности признаков не найден.")
        return None

    fi_df = pd.read_csv(fi_path, index_col=0)
    if "importance" not in fi_df.columns:
        print("⚠️  Неверный формат файла важности признаков.")
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    fi_df["importance"].plot(kind="bar", ax=ax)
    ax.set_title("Важность признаков (Vanilla ANFIS)")
    ax.set_ylabel("Важность")
    ax.set_xlabel("Признак")
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = results_dir / f"feature_importance_{timestamp}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_metrics(summary, output_dir):
    metrics = summary.get("metrics", {})
    vanilla_metrics = summary.get("metrics_vanilla", {})
    metrics_source = summary.get("metrics_source", "vanilla")

    if not metrics:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    metric_names = ["mse", "rmse", "mae", "r2_weighted", "r2_mean"]
    display_names = {
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2_weighted": "R² weighted",
        "r2_mean": "R² mean"
    }

    values = [metrics.get(name) for name in metric_names if name in metrics]
    labels = [display_names.get(name, name) for name in metric_names if name in metrics]

    ax.bar(labels, values, color="#5B8DEF")
    ax.set_title(f"Метрики модели ({metrics_source})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"metrics_{summary['timestamp']}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    if metrics_source == "shap" and vanilla_metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        vanilla_vals = [vanilla_metrics.get(name) for name in metric_names if name in vanilla_metrics]
        shap_vals = [metrics.get(name) for name in metric_names if name in metrics]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width / 2, vanilla_vals, width, label="Vanilla")
        ax.bar(x + width / 2, shap_vals, width, label="SHAP")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title("Сравнение Vanilla vs SHAP")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        comparison_path = output_dir / f"metrics_comparison_{summary['timestamp']}.png"
        fig.savefig(comparison_path, dpi=150)
        plt.close(fig)
        return [output_path, comparison_path]

    return [output_path]


def plot_error_distribution(pred, target, timestamp, output_dir, suffix=""):
    if pred is None or target is None:
        return []

    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    if pred.shape != target.shape:
        print("⚠️  Размерности предсказаний и истинных значений не совпадают, пропускаю графики ошибок.")
        return []

    errors = pred - target
    mae_per_bin = np.mean(np.abs(errors), axis=0)
    rmse_per_bin = np.sqrt(np.mean(errors ** 2, axis=0))

    energies = np.array(Ebins_float_IAEA_Comp)
    if energies.shape[0] != mae_per_bin.shape[0]:
        energies = np.arange(mae_per_bin.shape[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(energies, mae_per_bin, label="MAE", linewidth=2)
    ax.plot(energies, rmse_per_bin, label="RMSE", linewidth=2)
    ax.set_xlabel("Энергия (бин)")
    ax.set_ylabel("Ошибка")
    ax.set_title("Ошибка по энергиям" + (" (денорм.)" if suffix else ""))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mae_path = output_dir / f"errors_energy_{timestamp}{suffix}.png"
    fig.savefig(mae_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(errors.flatten(), bins=60, alpha=0.7, color="#F17666")
    ax.set_title("Распределение ошибок" + (" (денорм.)" if suffix else ""))
    ax.set_xlabel("Predicted - True")
    ax.set_ylabel("Количество")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    hist_path = output_dir / f"errors_hist_{timestamp}{suffix}.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target.flatten(), pred.flatten(), s=4, alpha=0.3)
    ax.plot([target.min(), target.max()], [target.min(), target.max()], color="black", linestyle="--")
    ax.set_xlabel("Истинные значения")
    ax.set_ylabel("Предсказания")
    ax.set_title("Scatter истинные vs предсказанные" + (" (денорм.)" if suffix else ""))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_path = output_dir / f"scatter_{timestamp}{suffix}.png"
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)

    return [mae_path, hist_path, scatter_path]


def _get_energy_axis(length):
    energies = np.array(Ebins_float_IAEA_Comp)
    if energies.shape[0] != length:
        energies = np.arange(length)
    return energies


def _get_energy_edges(energies):
    energies = np.asarray(energies, dtype=float)
    if energies.ndim != 1 or energies.size < 2:
        return np.arange(energies.size + 1)
    ratios = energies[1:] / energies[:-1]
    edges = np.empty(energies.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(energies[:-1] * energies[1:])
    edges[0] = energies[0] / np.sqrt(ratios[0])
    edges[-1] = energies[-1] * np.sqrt(ratios[-1])
    return edges


def _apply_axis_style(ax, log_x):
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    try:
        ax.minorticks_on()
    except Exception:
        pass


def plot_prediction_samples(pred, target, timestamp, output_dir, sample_size=5, suffix="", indices=None, seed=42, style="step", log_x=True):
    if pred is None or target is None:
        return []
    if pred.shape != target.shape:
        print("⚠️  Размерности предсказаний и истинных значений не совпадают, пропускаю сводные графики.")
        return []

    n_samples = pred.shape[0]
    if indices is not None:
        indices = np.asarray(indices, dtype=int)
        indices = indices[(indices >= 0) & (indices < n_samples)]
    else:
        sample_size = min(sample_size, n_samples)
        if sample_size <= 0:
            return []
        rng = np.random.default_rng(seed)
        base_idx = np.linspace(0, n_samples - 1, min(sample_size, max(sample_size // 2, 1)), dtype=int)
        rand_idx = rng.choice(n_samples, size=sample_size, replace=False) if n_samples > sample_size else base_idx
        indices = np.unique(np.concatenate([base_idx, rand_idx]))
        if indices.size > sample_size:
            indices = indices[:sample_size]

    if indices.size == 0:
        return []

    energies = _get_energy_axis(pred.shape[1])

    fig, ax = plt.subplots(figsize=(10, 5))
    if style == "step":
        for idx in indices:
            ax.step(energies, target[idx], color="#999999", alpha=0.35, where="mid")
        for idx in indices:
            ax.step(energies, pred[idx], color="#FF6B6B", alpha=0.25, linestyle="--", where="mid")
        ax.step(energies, target[indices[0]], color="#333333", linewidth=2, label="Истинные (пример)", where="mid")
        ax.step(energies, pred[indices[0]], color="#FF6B6B", linewidth=2, linestyle="--", label="Предсказания (пример)", where="mid")
    else:
        for idx in indices:
            ax.plot(energies, target[idx], color="#999999", alpha=0.35)
        for idx in indices:
            ax.plot(energies, pred[idx], color="#FF6B6B", alpha=0.25, linestyle="--")
        ax.plot(energies, target[indices[0]], color="#333333", linewidth=2, label="Истинные (пример)")
        ax.plot(energies, pred[indices[0]], color="#FF6B6B", linewidth=2, linestyle="--", label="Предсказания (пример)")
    ax.set_title("Несколько спектров (истинные/предсказанные)")
    ax.set_xlabel("Energy, eV")
    ax.set_ylabel("phi, neutron cm^-2 s^-1")
    _apply_axis_style(ax, log_x)
    ax.legend()
    plt.tight_layout()
    samples_path = output_dir / f"spectra_samples_{timestamp}{suffix}.png"
    fig.savefig(samples_path, dpi=150)
    plt.close(fig)

    mean_true = target.mean(axis=0)
    mean_pred = pred.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    if style == "step":
        ax.step(energies, mean_true, label="Средний истинный спектр", linewidth=2, where="mid")
        ax.step(energies, mean_pred, label="Средний предсказанный спектр", linewidth=2, linestyle="--", where="mid")
    else:
        ax.plot(energies, mean_true, label="Средний истинный спектр", linewidth=2)
        ax.plot(energies, mean_pred, label="Средний предсказанный спектр", linewidth=2, linestyle="--")
    ax.set_title("Средние спектры")
    ax.set_xlabel("Energy, eV")
    ax.set_ylabel("phi, neutron cm^-2 s^-1")
    _apply_axis_style(ax, log_x)
    ax.legend()
    plt.tight_layout()
    mean_path = output_dir / f"spectra_mean_{timestamp}{suffix}.png"
    fig.savefig(mean_path, dpi=150)
    plt.close(fig)

    return [samples_path, mean_path], indices


def plot_individual_spectra(pred, target, timestamp, output_dir, count, suffix="", base_dir="spectra", seed=42, indices=None, style="step", log_x=True):
    if pred is None or target is None:
        return [], np.array([])
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

    energies = _get_energy_axis(pred.shape[1])
    spectra_dir = output_dir / base_dir / ("denorm" if suffix else "normalized")
    spectra_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    for idx in indices:
        fig, ax = plt.subplots(figsize=(10, 5))
        if style == "step":
            ax.step(energies, target[idx], label="Истинный спектр", linewidth=2, where="mid")
            ax.step(energies, pred[idx], label="Предсказанный спектр", linewidth=2, linestyle="--", where="mid")
        else:
            ax.plot(energies, target[idx], label="Истинный спектр", linewidth=2)
            ax.plot(energies, pred[idx], label="Предсказанный спектр", linewidth=2, linestyle="--")
        ax.set_title(f"Спектр #{idx}")
        ax.set_xlabel("Energy, eV")
        ax.set_ylabel("phi, neutron cm^-2 s^-1")
        _apply_axis_style(ax, log_x)
        ax.legend()
        plt.tight_layout()
        path = spectra_dir / f"spectrum_{timestamp}_idx{idx}{suffix}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figure_paths.append(path)

    return figure_paths, indices


def write_report(summary, figures_map, output_dir, report_file):
    if not report_file:
        return None

    report_path = Path(report_file)
    if not report_path.is_absolute():
        report_path = output_dir / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

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
                        rel = rel
                    f.write(f"- ![{rel}]({rel})\n")
                f.write("\n")

    return report_path


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

    fig, ax = plt.subplots(figsize=(8, 4))
    for key, values in history.items():
        if not isinstance(values, (list, tuple)):
            continue
        finite_values = [v for v in values if np.isfinite(v)]
        if not finite_values:
            continue
        ax.plot(finite_values, label=key.replace("_", " "))
    ax.set_title("История потерь SHAP")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Значение")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / f"shap_history_{timestamp}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()

    summary_path = Path(args.summary) if args.summary else _find_latest_summary()
    summary = load_summary(summary_path)
    results_dir = summary_path.parent
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📄 Используем сводку: {summary_path}")
    print(f"📂 Каталог графиков: {output_dir}")

    saved_files = summary.get("saved_files", {})
    samples = load_samples(results_dir, saved_files)
    generated_figures = {}
    plot_config = {"style": args.plot_style, "log_x": args.log_x}

    saved_figures = plot_samples(output_dir, summary["timestamp"], samples, args.spectra_dir, **plot_config)
    if saved_figures:
        print("🖼️  Графики сохранённой подвыборки:")
        for path in saved_figures:
            print(f"   {path}")
    generated_figures["saved_samples"] = saved_figures

    predictions = load_predictions(results_dir, saved_files)
    error_figs = plot_error_distribution(
        predictions["pred"],
        predictions["target"],
        summary["timestamp"],
        output_dir,
        suffix=""
    )
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
        count=args.spectra_count,
        suffix="",
        base_dir=args.spectra_dir,
        seed=args.seed,
        **plot_config
    )
    if indiv_norm_figs:
        print("🖼️  Отдельные спектры (нормализованные):")
        for path in indiv_norm_figs:
            print(f"   {path}")
    generated_figures["spectra_normalized"] = indiv_norm_figs

    sample_figs, _ = plot_prediction_samples(
        predictions["pred"],
        predictions["target"],
        summary["timestamp"],
        output_dir,
        sample_size=args.spectra_count,
        suffix="",
        indices=selected_indices if selected_indices.size else None,
        seed=args.seed,
        **plot_config
    )
    if sample_figs:
        print("🖼️  Сводные спектры (нормализованные):")
        for path in sample_figs:
            print(f"   {path}")
    generated_figures["summary_spectra_normalized"] = sample_figs

    error_denorm_figs = plot_error_distribution(
        predictions["pred_denorm"],
        predictions["target_denorm"],
        summary["timestamp"],
        output_dir,
        suffix="_denorm"
    )
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
        count=args.spectra_count,
        suffix="_denorm",
        base_dir=args.spectra_dir,
        seed=args.seed,
        indices=selected_indices if selected_indices.size else None,
        **plot_config
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
        sample_size=args.spectra_count,
        suffix="_denorm",
        indices=selected_indices if selected_indices.size else None,
        seed=args.seed,
        **plot_config
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

    fi_figure = plot_feature_importance(results_dir, summary["timestamp"])
    if fi_figure:
        target_path = output_dir / fi_figure.name
        if target_path != fi_figure:
            target_path.write_bytes(fi_figure.read_bytes())
            print(f"🖼️  Диаграмма важности признаков: {target_path}")
        else:
            print(f"🖼️  Диаграмма важности признаков: {fi_figure}")
        generated_figures.setdefault("feature_importance", []).append(
            target_path if target_path.exists() else fi_figure
        )

    shap_files = saved_files.get("shap")
    if shap_files and "feature_importance_shap" in shap_files:
        shap_fi_path = results_dir / shap_files["feature_importance_shap"]
        if shap_fi_path.exists():
            fi_df = pd.read_csv(shap_fi_path, index_col=0)
            fig, ax = plt.subplots(figsize=(8, 4))
            fi_df["importance"].plot(kind="bar", color="orange", ax=ax)
            ax.set_title("Важность признаков (SHAP)")
            ax.set_ylabel("Важность")
            ax.set_xlabel("Признак")
            plt.xticks(rotation=45)
            plt.tight_layout()
            shap_fig_path = output_dir / f"feature_importance_shap_{summary['timestamp']}.png"
            fig.savefig(shap_fig_path, dpi=150)
            plt.close(fig)
            print(f"🖼️  SHAP важность признаков: {shap_fig_path}")
            generated_figures.setdefault("feature_importance_shap", []).append(shap_fig_path)

    shap_history_fig = plot_shap_history(results_dir, shap_files, summary["timestamp"], output_dir)
    if shap_history_fig:
        print(f"🖼️  История SHAP: {shap_history_fig}")
        generated_figures.setdefault("shap_history", []).append(shap_history_fig)

    report_path = write_report(summary, generated_figures, output_dir, args.report_file)
    if report_path:
        print(f"\n📝 Markdown-отчёт: {report_path}")

    print("\n✅ Построение графиков завершено.")


if __name__ == "__main__":
    main()

