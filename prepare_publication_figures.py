#!/usr/bin/env python3
"""Собирает финальный набор фигур для статьи/тезисов."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.mpl_style import (
    COLORS,
    apply_axis_style,
    finalize_figure,
    resolve_energy_axis,
    setup_publication_style,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Подготовка финального набора фигур")
    parser.add_argument("--shap-summary", required=True, help="Summary основного SHAP+Tikhonov run")
    parser.add_argument("--vanilla-summary", required=True, help="Summary vanilla baseline")
    parser.add_argument("--uncertainty-dir", required=True, help="Папка с Monte Carlo range run")
    parser.add_argument("--output-dir", default="results/final_figures", help="Куда сохранить финальные фигуры")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_array(summary_path, relpath):
    path = Path(summary_path).parent / relpath
    return np.load(path)


def load_predictions_from_summary(summary_path):
    summary = load_json(summary_path)
    saved = summary.get("saved_files", {})
    return {
        "summary": summary,
        "pred": load_array(summary_path, saved["predictions"]),
        "target": load_array(summary_path, saved["targets_test"]),
        "pred_denorm": load_array(summary_path, saved["predictions_denorm"]) if saved.get("predictions_denorm") else None,
        "target_denorm": load_array(summary_path, saved["targets_denorm"]) if saved.get("targets_denorm") else None,
    }


def find_latest(pattern):
    files = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"Ничего не найдено по шаблону: {pattern}")
    return files[-1]


def plot_metrics_comparison(vanilla_summary, shap_summary, output_path):
    metric_groups = [
        ("Ошибки", ["mse", "rmse", "mae"], {"mse": "MSE", "rmse": "RMSE", "mae": "MAE"}),
        ("Качество", ["r2_weighted", "r2_mean"], {"r2_weighted": "R² weighted", "r2_mean": "R² mean"}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.suptitle("Сравнение Vanilla vs SHAP + Tikhonov", y=1.03)
    for ax, (title, keys, labels_map) in zip(axes, metric_groups):
        labels = [labels_map[k] for k in keys]
        vanilla_vals = [vanilla_summary["metrics"][k] for k in keys]
        shap_vals = [shap_summary["metrics"][k] for k in keys]
        x = np.arange(len(labels))
        width = 0.36
        ax.bar(x - width / 2, vanilla_vals, width, label="Vanilla", color=COLORS["muted"], alpha=0.88)
        ax.bar(x + width / 2, shap_vals, width, label="SHAP + Tikhonov", color=COLORS["pred"], alpha=0.92)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        apply_axis_style(ax)
        ax.legend(loc="upper right")
        if title == "Качество":
            ax.set_ylim(0, max(1.0, max(vanilla_vals + shap_vals) * 1.15))
    finalize_figure(fig, output_path)


def plot_mean_spectra_comparison(vanilla, shap, output_path):
    target = np.asarray(shap["target"], dtype=float)
    pred_vanilla = np.asarray(vanilla["pred"], dtype=float)
    pred_shap = np.asarray(shap["pred"], dtype=float)
    energies = resolve_energy_axis(target.shape[1])

    mean_true = target.mean(axis=0)
    mean_vanilla = pred_vanilla.mean(axis=0)
    mean_shap = pred_shap.mean(axis=0)
    std_true = target.std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.8), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.1]})
    ax_top, ax_bottom = axes
    ax_top.fill_between(energies, mean_true - std_true, mean_true + std_true, step="mid", alpha=0.18, color=COLORS["fill_true"], label="Истинный спектр ±1σ")
    ax_top.step(energies, mean_true, where="mid", color=COLORS["true"], linewidth=2.2, label="Истинный средний")
    ax_top.step(energies, mean_vanilla, where="mid", color=COLORS["muted"], linewidth=2.0, linestyle="-.", label="Vanilla")
    ax_top.step(energies, mean_shap, where="mid", color=COLORS["pred"], linewidth=2.2, linestyle="--", label="SHAP + Tikhonov")
    ax_top.set_title("Средние восстановленные спектры")
    ax_top.set_ylabel("Плотность потока")
    apply_axis_style(ax_top, log_x=True)
    ax_top.legend(loc="upper right")

    ax_bottom.step(energies, mean_vanilla - mean_true, where="mid", color=COLORS["muted"], linewidth=1.9, linestyle="-.", label="Bias Vanilla")
    ax_bottom.step(energies, mean_shap - mean_true, where="mid", color=COLORS["error"], linewidth=1.9, label="Bias SHAP + Tikhonov")
    ax_bottom.axhline(0.0, color=COLORS["ink"], linewidth=1.0, linestyle=":")
    ax_bottom.set_xlabel("Энергия, эВ")
    ax_bottom.set_ylabel("Bias")
    apply_axis_style(ax_bottom, log_x=True)
    ax_bottom.legend(loc="upper right")
    finalize_figure(fig, output_path)


def plot_examples_grid(vanilla, shap, output_path, n_examples=4):
    target = np.asarray(shap["target"], dtype=float)
    pred_vanilla = np.asarray(vanilla["pred"], dtype=float)
    pred_shap = np.asarray(shap["pred"], dtype=float)
    energies = resolve_energy_axis(target.shape[1])

    log_e = np.log10(energies + 1e-30)
    centroid = (target * log_e[None, :]).sum(axis=1) / np.clip(target.sum(axis=1), 1e-12, None)
    shap_mae = np.mean(np.abs(pred_shap - target), axis=1)
    vanilla_mae = np.mean(np.abs(pred_vanilla - target), axis=1)
    improvement = vanilla_mae - shap_mae

    quantiles = np.quantile(centroid, np.linspace(0.0, 1.0, n_examples + 1))
    selected = []
    for start, stop in zip(quantiles[:-1], quantiles[1:]):
        mask = (centroid >= start) & (centroid <= stop if stop == quantiles[-1] else centroid < stop)
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue
        best_local = idxs[np.argmax(improvement[idxs] - 0.2 * shap_mae[idxs])]
        selected.append(int(best_local))
    if len(selected) < n_examples:
        fallback = list(np.argsort(-improvement))
        for idx in fallback:
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) == n_examples:
                break
    selected = selected[:n_examples]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), sharex=True, sharey=True)
    fig.suptitle("Четыре характерных примера восстановления спектра", y=1.01)
    flat_axes = axes.ravel()
    handles = None
    labels = None
    for ax, idx in zip(flat_axes, selected):
        h1 = ax.step(energies, target[idx], where="mid", color=COLORS["true"], linewidth=2.2, label="Истинный спектр")
        h2 = ax.step(energies, pred_vanilla[idx], where="mid", color=COLORS["muted"], linewidth=1.9, linestyle="-.", label="Vanilla")
        h3 = ax.step(energies, pred_shap[idx], where="mid", color=COLORS["pred"], linewidth=2.1, linestyle="--", label="V2.1")
        ax.set_title(f"Образец #{idx}: ΔMAE = {improvement[idx]:.3f}")
        apply_axis_style(ax, log_x=True)
        handles = [h1[0], h2[0], h3[0]]
        labels = [h.get_label() for h in handles]
    for ax in flat_axes[2:]:
        ax.set_xlabel("Энергия, эВ")
    flat_axes[0].set_ylabel("Плотность потока")
    flat_axes[2].set_ylabel("Плотность потока")
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.98))
    finalize_figure(fig, output_path)


def plot_shap_importance_from_csv(summary_path, output_path):
    summary = load_json(summary_path)
    saved = summary.get("saved_files", {}).get("shap", {})
    csv_name = saved.get("feature_importance_shap")
    if not csv_name:
        raise FileNotFoundError("Не найден CSV с SHAP-важностями в summary")
    csv_path = Path(summary_path).parent / csv_name
    df = pd.read_csv(csv_path)
    if "importance" not in df.columns:
        raise ValueError("Ожидался столбец importance в SHAP CSV")
    if "feature" not in df.columns:
        feature_col = next((c for c in df.columns if c.lower().startswith("unnamed")), None)
        if feature_col is None:
            raise ValueError("Ожидался столбец feature либо индексный столбец с названиями признаков")
        df = df.rename(columns={feature_col: "feature"})
    df = df.sort_values("importance", ascending=True)
    top3_mass = float(df["importance"].sort_values(ascending=False).head(3).sum())

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.barh(df["feature"], df["importance"], color=COLORS["pred"], alpha=0.9)
    for bar, value in zip(bars, df["importance"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color=COLORS["ink"],
        )
    ax.set_title(f"Нормализованная SHAP-важность входных измерений (top-3 mass = {top3_mass:.2f})")
    ax.set_xlabel("Нормализованная важность")
    ax.set_ylabel("Измерение")
    ax.set_xlim(0, max(0.3, float(df["importance"].max()) * 1.2))
    apply_axis_style(ax)
    finalize_figure(fig, output_path)


def plot_regularization_comparison(vanilla_summary, shap_summary, output_path):
    reg_v = vanilla_summary.get("diagnostics", {}).get("regularization", {})
    reg_s = shap_summary.get("diagnostics", {}).get("regularization", {})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("Вклад регуляризации", y=1.02)

    ax = axes[0, 0]
    labels = ["SHAP", "Tikhonov"]
    vanilla_vals = [reg_v.get("shap_contribution", {}).get("mean", 0.0), reg_v.get("tikhonov_contribution", {}).get("mean", 0.0)]
    shap_vals = [reg_s.get("shap_contribution", {}).get("mean", 0.0), reg_s.get("tikhonov_contribution", {}).get("mean", 0.0)]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, vanilla_vals, width, label="Vanilla", color=COLORS["muted"], alpha=0.85)
    ax.bar(x + width / 2, shap_vals, width, label="SHAP + Tikhonov", color=COLORS["pred"], alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Средний вклад в loss")
    apply_axis_style(ax)
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    share_labels = ["mean", "last", "max"]
    vanilla_share = [reg_v.get("regularization_share", {}).get(k, 0.0) for k in share_labels]
    shap_share = [reg_s.get("regularization_share", {}).get(k, 0.0) for k in share_labels]
    x = np.arange(len(share_labels))
    ax.bar(x - width / 2, vanilla_share, width, label="Vanilla", color=COLORS["muted"], alpha=0.85)
    ax.bar(x + width / 2, shap_share, width, label="SHAP + Tikhonov", color=COLORS["accent"], alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(share_labels)
    ax.set_title("Доля регуляризации")
    apply_axis_style(ax)

    ax = axes[1, 0]
    component_signal = reg_s.get("weighted_component_signal", {})
    if component_signal:
        names = list(component_signal.keys())
        values = [component_signal[name]["mean"] for name in names]
        ax.bar(names, values, color=[COLORS["true"], COLORS["pred"], COLORS["accent"], COLORS["violet"]][:len(names)], alpha=0.92)
        ax.set_title("Эффективный вклад компонент SHAP")
        apply_axis_style(ax)
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.axis("off")

    ax = axes[1, 1]
    train_labels = ["Время train", "MSE", "R² weighted"]
    train_v = [vanilla_summary.get("training_time_total", 0.0), vanilla_summary["metrics"]["mse"], vanilla_summary["metrics"]["r2_weighted"]]
    train_s = [shap_summary.get("training_time_total", 0.0), shap_summary["metrics"]["mse"], shap_summary["metrics"]["r2_weighted"]]
    x = np.arange(len(train_labels))
    ax.bar(x - width / 2, train_v, width, label="Vanilla", color=COLORS["muted"], alpha=0.85)
    ax.bar(x + width / 2, train_s, width, label="SHAP + Tikhonov", color=COLORS["pred"], alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(train_labels, rotation=15)
    ax.set_title("Цена и качество")
    apply_axis_style(ax)
    ax.legend(loc="upper left")

    finalize_figure(fig, output_path)


def copy_shap_importance(shap_summary, output_path):
    shap_csv = shap_summary.get("saved_files", {}).get("shap", {}).get("feature_importance_shap")
    if not shap_csv:
        raise FileNotFoundError("В SHAP summary не найден feature_importance_shap")
    source_png = Path(output_path).parent.parent / f"feature_importance_shap_{shap_summary['timestamp']}.png"
    if not source_png.exists():
        source_png = Path(args.shap_summary).parent / f"feature_importance_shap_{shap_summary['timestamp']}.png"
    shutil.copy2(source_png, output_path)


def latest_in_dir(directory, pattern):
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"Не найдено {pattern} в {directory}")
    return files[-1]


def plot_uncertainty_composite(uncertainty_dir, output_path):
    uncertainty_dir = Path(uncertainty_dir)
    summary_csv = latest_in_dir(uncertainty_dir, "uncertainty_summary_*.csv")
    df = pd.read_csv(summary_csv).sort_values("error_percent")

    error10_dir = uncertainty_dir / "error_10"
    if not error10_dir.exists():
        error10_dir = uncertainty_dir / "error_10.0"
    if not error10_dir.exists():
        raise FileNotFoundError("Не найдена папка error_10 для uncertainty")

    base = np.load(latest_in_dir(error10_dir, "base_*.npy"))[0]
    mean = np.load(latest_in_dir(error10_dir, "mean_*.npy"))[0]
    std = np.load(latest_in_dir(error10_dir, "std_*.npy"))[0]
    percentiles = np.load(latest_in_dir(error10_dir, "percentiles_*.npz"))
    p5 = percentiles["p5"][0]
    p95 = percentiles["p95"][0]
    energies = resolve_energy_axis(base.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5))
    ax_left, ax_right = axes

    ax_left.plot(df["error_percent"], df["mean_std"], marker="o", linewidth=2.0, color=COLORS["true"], label="mean_std")
    ax_left.plot(df["error_percent"], df["max_std"], marker="o", linewidth=2.0, color=COLORS["pred"], label="max_std")
    ax_left.plot(df["error_percent"], df["cv"], marker="o", linewidth=2.0, color=COLORS["accent"], label="cv")
    ax_left.plot(df["error_percent"], df["ci_width"], marker="o", linewidth=2.0, color=COLORS["violet"], label="ci_width")
    ax_left.set_title("Рост неопределённости")
    ax_left.set_xlabel("Ошибка измерений, %")
    ax_left.set_ylabel("Метрика неопределённости")
    apply_axis_style(ax_left)
    ax_left.legend(loc="upper left")

    ax_right.fill_between(energies, p5, p95, step="mid", alpha=0.18, color=COLORS["fill_pred"], label="P5-P95")
    ax_right.fill_between(energies, mean - std, mean + std, step="mid", alpha=0.20, color=COLORS["fill_true"], label="mean ± 1σ")
    ax_right.step(energies, base, where="mid", color=COLORS["true"], linewidth=2.1, label="Базовый спектр")
    ax_right.step(energies, mean, where="mid", color=COLORS["pred"], linewidth=2.1, linestyle="--", label="Средний спектр")
    ax_right.set_title("Monte Carlo band при 10% шуме")
    ax_right.set_xlabel("Энергия, эВ")
    ax_right.set_ylabel("Плотность потока")
    apply_axis_style(ax_right, log_x=True)
    ax_right.legend(loc="upper right")

    finalize_figure(fig, output_path)


def write_manifest(output_dir):
    output_dir = Path(output_dir)
    manifest = output_dir / "manifest.md"
    lines = [
        "# Final Figures",
        "",
        "1. `fig_01_metrics_comparison.png` — сравнение метрик Vanilla vs SHAP + Tikhonov.",
        "2. `fig_02_mean_spectra_comparison.png` — средние восстановленные спектры и bias.",
        "3. `fig_03_representative_spectrum.png` — четыре характерных примера восстановления спектров.",
        "4. `fig_04_regularization_comparison.png` — вклад регуляризации и цена по времени/качеству.",
        "5. `fig_05_shap_importance.png` — итоговая SHAP-важность признаков с округлёнными подписями.",
        "6. `fig_06_uncertainty_monte_carlo.png` — рост неопределённости и доверительный коридор при 10% шуме.",
    ]
    manifest.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    setup_publication_style()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vanilla = load_predictions_from_summary(args.vanilla_summary)
    shap = load_predictions_from_summary(args.shap_summary)

    if vanilla["target"].shape != shap["target"].shape:
        raise ValueError("Vanilla и SHAP summaries используют разные test shapes")

    plot_metrics_comparison(vanilla["summary"], shap["summary"], out / "fig_01_metrics_comparison.png")
    plot_mean_spectra_comparison(vanilla, shap, out / "fig_02_mean_spectra_comparison.png")
    plot_examples_grid(vanilla, shap, out / "fig_03_representative_spectrum.png")
    plot_regularization_comparison(vanilla["summary"], shap["summary"], out / "fig_04_regularization_comparison.png")
    plot_shap_importance_from_csv(args.shap_summary, out / "fig_05_shap_importance.png")

    plot_uncertainty_composite(args.uncertainty_dir, out / "fig_06_uncertainty_monte_carlo.png")
    write_manifest(out)
    print(f"✅ Финальные фигуры сохранены в {out}")
