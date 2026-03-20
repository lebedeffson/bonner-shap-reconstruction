#!/usr/bin/env python3
"""Сравнение методов регуляризации по качеству, гладкости и Monte Carlo-устойчивости."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization.mpl_style import COLORS, annotate_bar_values, apply_axis_style, finalize_figure, setup_publication_style


def parse_args():
    parser = argparse.ArgumentParser(description="Сравнение Vanilla / SHAP / Tikhonov / SHAP+Tikhonov")
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Метод в формате label=path/to/training_summary.json",
    )
    parser.add_argument(
        "--uncertainty",
        action="append",
        default=[],
        help="Monte Carlo директория в формате label=path/to/uncertainty_dir",
    )
    parser.add_argument("--output-dir", default="results/method_comparison", help="Куда сохранить результаты")
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(summary_path: Path, summary: dict):
    saved = summary.get("saved_files", {})
    pred_rel = saved.get("predictions")
    true_rel = saved.get("targets_test")
    if not pred_rel or not true_rel:
        return None, None

    pred_path = summary_path.parent / pred_rel
    true_path = summary_path.parent / true_rel
    if not pred_path.exists() or not true_path.exists():
        return None, None

    pred = np.load(pred_path)
    true = np.load(true_path)
    return np.asarray(pred, dtype=float), np.asarray(true, dtype=float)


def compute_smoothness(predictions: np.ndarray, targets: np.ndarray | None = None):
    if predictions is None:
        return {
            "d1_mean_sq": float("nan"),
            "d2_mean_sq": float("nan"),
            "d1_error_sq": float("nan"),
            "d2_error_sq": float("nan"),
        }
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] < 3:
        return {
            "d1_mean_sq": float("nan"),
            "d2_mean_sq": float("nan"),
            "d1_error_sq": float("nan"),
            "d2_error_sq": float("nan"),
        }
    d1 = predictions[:, 1:] - predictions[:, :-1]
    d2 = predictions[:, 2:] - 2.0 * predictions[:, 1:-1] + predictions[:, :-2]
    result = {
        "d1_mean_sq": float(np.mean(d1 ** 2)),
        "d2_mean_sq": float(np.mean(d2 ** 2)),
    }
    if targets is not None:
        targets = np.asarray(targets, dtype=float)
        if targets.shape == predictions.shape and targets.shape[1] >= 3:
            d1_true = targets[:, 1:] - targets[:, :-1]
            d2_true = targets[:, 2:] - 2.0 * targets[:, 1:-1] + targets[:, :-2]
            result["d1_error_sq"] = float(np.mean((d1 - d1_true) ** 2))
            result["d2_error_sq"] = float(np.mean((d2 - d2_true) ** 2))
        else:
            result["d1_error_sq"] = float("nan")
            result["d2_error_sq"] = float("nan")
    else:
        result["d1_error_sq"] = float("nan")
        result["d2_error_sq"] = float("nan")
    return result


def collect_methods(method_args):
    methods = []
    for raw in method_args:
        if "=" not in raw:
            raise ValueError(f"Ожидался формат label=path, получено: {raw}")
        label, path_str = raw.split("=", 1)
        path = Path(path_str).resolve()
        summary = load_json(path)
        pred, true = load_predictions(path, summary)
        smoothness = compute_smoothness(pred, true)
        reg = summary.get("diagnostics", {}).get("regularization", {})
        methods.append(
            {
                "label": label,
                "summary_path": str(path),
                "summary": summary,
                "pred": pred,
                "true": true,
                "smoothness": smoothness,
                "training_time_total": float(summary.get("training_time_total") or 0.0),
                "training_time_shap": float(summary.get("training_time_shap") or 0.0),
                "mse": float(summary["metrics"]["mse"]),
                "rmse": float(summary["metrics"]["rmse"]),
                "mae": float(summary["metrics"]["mae"]),
                "r2_weighted": float(summary["metrics"]["r2_weighted"]),
                "r2_mean": float(summary["metrics"].get("r2_mean", np.nan)),
                "band_0_19_r2": float(summary.get("band_metrics", {}).get("band_0_19", {}).get("r2", np.nan)),
                "band_20_39_r2": float(summary.get("band_metrics", {}).get("band_20_39", {}).get("r2", np.nan)),
                "band_40_59_r2": float(summary.get("band_metrics", {}).get("band_40_59", {}).get("r2", np.nan)),
                "regularization_share_mean": float(reg.get("regularization_share", {}).get("mean", 0.0)),
                "shap_contribution_mean": float(reg.get("shap_contribution", {}).get("mean", 0.0)),
                "tikhonov_contribution_mean": float(reg.get("tikhonov_contribution", {}).get("mean", 0.0)),
                "nonnegativity_contribution_mean": float(reg.get("nonnegativity_contribution", {}).get("mean", 0.0)),
                "negative_fraction": float(summary.get("diagnostics", {}).get("prediction_stats", {}).get("negative_fraction", np.nan)),
                "negative_count": float(summary.get("diagnostics", {}).get("prediction_stats", {}).get("negative_count", np.nan)),
                "dominant_regularizer": reg.get("dominant_regularizer", "none"),
                "dominant_shap_component": reg.get("dominant_shap_component", "none"),
            }
        )
    return methods


def collect_uncertainty(uncertainty_args):
    result = {}
    for raw in uncertainty_args:
        if "=" not in raw:
            raise ValueError(f"Ожидался формат label=path, получено: {raw}")
        label, path_str = raw.split("=", 1)
        directory = Path(path_str).resolve()
        summaries = sorted(directory.glob("uncertainty_summary_*.csv"), key=lambda p: p.stat().st_mtime)
        if not summaries:
            raise FileNotFoundError(f"В {directory} не найден uncertainty_summary_*.csv")
        df = pd.read_csv(summaries[-1]).sort_values("error_percent")
        result[label] = df
    return result


def plot_method_tradeoff(df: pd.DataFrame, output_path: Path):
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    fig.suptitle("Сравнение методов: качество, гладкость и вычислительная цена", y=1.02)

    panels = [
        (axes[0, 0], "mse", "MSE", COLORS["error"]),
        (axes[0, 1], "r2_weighted", "R² weighted", COLORS["true"]),
        (axes[1, 0], "negative_fraction", "Доля отрицательных бинов", COLORS["accent"]),
        (axes[1, 1], "training_time_total", "Время обучения, с", COLORS["pred"]),
    ]
    for ax, key, title, color in panels:
        vals = df[key].to_numpy(dtype=float)
        bars = ax.bar(x, vals, color=color, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_title(title)
        apply_axis_style(ax)
        annotate_bar_values(ax, fmt="{:.4f}" if key != "training_time_total" else "{:.1f}")

    finalize_figure(fig, output_path)


def plot_band_quality(df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    labels = ["0-19", "20-39", "40-59"]
    x = np.arange(len(labels))
    width = 0.18 if len(df) >= 4 else 0.22
    palette = [COLORS["muted"], COLORS["accent"], COLORS["violet"], COLORS["pred"], COLORS["true"]]

    for idx, row in df.reset_index(drop=True).iterrows():
        vals = [row["band_0_19_r2"], row["band_20_39_r2"], row["band_40_59_r2"]]
        ax.bar(x + (idx - (len(df) - 1) / 2) * width, vals, width, label=row["label"], color=palette[idx % len(palette)], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R²")
    ax.set_title("Качество по энергетическим диапазонам")
    apply_axis_style(ax)
    ax.legend(loc="best")
    finalize_figure(fig, output_path)


def plot_uncertainty_comparison(uncertainty_map: dict[str, pd.DataFrame], output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.2))
    ax_left, ax_right = axes
    palette = [COLORS["muted"], COLORS["accent"], COLORS["violet"], COLORS["pred"], COLORS["true"]]

    for idx, (label, df) in enumerate(uncertainty_map.items()):
        color = palette[idx % len(palette)]
        ax_left.plot(df["error_percent"], df["mean_std"], marker="o", linewidth=2.0, color=color, label=label)
        ax_right.plot(df["error_percent"], df["max_std"], marker="o", linewidth=2.0, color=color, label=label)

    ax_left.set_title("Средняя неопределённость vs шум")
    ax_left.set_xlabel("Ошибка измерений, %")
    ax_left.set_ylabel("mean_std")
    apply_axis_style(ax_left)
    ax_left.legend(loc="upper left")

    ax_right.set_title("Максимальная неопределённость vs шум")
    ax_right.set_xlabel("Ошибка измерений, %")
    ax_right.set_ylabel("max_std")
    apply_axis_style(ax_right)
    ax_right.legend(loc="upper left")

    finalize_figure(fig, output_path)


def write_summary(df: pd.DataFrame, output_dir: Path):
    summary_csv = output_dir / "method_comparison_summary.csv"
    df.to_csv(summary_csv, index=False)

    table_df = df.copy()
    numeric_cols = table_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        table_df[col] = table_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")

    lines = [
        "# Method Comparison",
        "",
        "Сравнение методов по качеству, гладкости и регуляризации.",
        "",
    ]
    best_mse = df.loc[df["mse"].idxmin(), "label"]
    best_r2 = df.loc[df["r2_weighted"].idxmax(), "label"]
    best_nonneg = df.loc[df["negative_fraction"].idxmin(), "label"]
    lines.extend(
        [
            f"- Лучший по MSE: `{best_mse}`",
            f"- Лучший по R² weighted: `{best_r2}`",
            f"- Лучший по доле неотрицательных бинов: `{best_nonneg}`",
            "",
            "## Таблица",
            "",
            table_df.to_csv(index=False),
        ]
    )
    (output_dir / "method_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    setup_publication_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = collect_methods(args.method)
    df = pd.DataFrame(methods)
    df["d1_mean_sq"] = df["smoothness"].apply(lambda x: x["d1_mean_sq"])
    df["d2_mean_sq"] = df["smoothness"].apply(lambda x: x["d2_mean_sq"])
    df["d1_error_sq"] = df["smoothness"].apply(lambda x: x["d1_error_sq"])
    df["d2_error_sq"] = df["smoothness"].apply(lambda x: x["d2_error_sq"])
    df = df.drop(columns=["summary", "pred", "true", "smoothness"])
    df = df.sort_values(["r2_weighted", "mse"], ascending=[False, True]).reset_index(drop=True)

    plot_method_tradeoff(df, output_dir / "fig_method_tradeoff.png")
    plot_band_quality(df, output_dir / "fig_band_quality.png")

    uncertainty_map = collect_uncertainty(args.uncertainty) if args.uncertainty else {}
    if uncertainty_map:
        plot_uncertainty_comparison(uncertainty_map, output_dir / "fig_uncertainty_methods.png")

    write_summary(df, output_dir)
    print(f"✅ Сравнение методов сохранено в {output_dir}")
