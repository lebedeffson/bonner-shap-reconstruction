#!/usr/bin/env python3
"""
Генерация инженерного отчёта по результатам обучения ANFIS.
Формирует многостраничный PDF с ключевыми метриками и графиками.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from constants import Ebins_float_IAEA_Comp


def parse_args():
    parser = argparse.ArgumentParser(description="Инженерный PDF-отчёт по результатам ANFIS")
    parser.add_argument(
        "--summary",
        required=True,
        help="Путь к training_summary_*.json"
    )
    parser.add_argument(
        "--output",
        help="Путь к итоговому PDF (по умолчанию рядом с summary)"
    )
    parser.add_argument(
        "--spectra-count",
        type=int,
        default=6,
        help="Сколько отдельных спектров показать (по умолчанию 6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для случайного выбора спектров"
    )
    return parser.parse_args()


def _load_np(path):
    return np.load(path) if path and Path(path).exists() else None


def _prepare_energies(length):
    energies = np.array(Ebins_float_IAEA_Comp, dtype=float)
    if energies.shape[0] != length:
        energies = np.arange(length, dtype=float) + 1.0
    return energies


def _step(ax, energies, values, *args, **kwargs):
    ax.step(energies, values, where="mid", *args, **kwargs)


def _text_block(ax, title, items, start_y=0.95, line_height=0.045):
    ax.text(0.02, start_y, title, fontsize=12, fontweight='bold', va='top')
    y = start_y - line_height
    for line in items:
        ax.text(0.03, y, line, fontsize=10, va='top')
        y -= line_height


def _format_metrics(metrics):
    lines = []
    if not metrics:
        return lines
    for key, value in metrics.items():
        try:
            numeric = float(value)
            lines.append(f"{key.upper():15s} : {numeric:.6f}")
        except (TypeError, ValueError):
            lines.append(f"{key.upper():15s} : {value}")
    return lines


def _format_band_metrics(band_metrics):
    lines = []
    if not band_metrics:
        return lines
    for band, stats in band_metrics.items():
        lines.append(f"{band}:")
        for key, value in stats.items():
            try:
                numeric = float(value)
                lines.append(f"  {key.upper():12s} = {numeric:.6f}")
            except (TypeError, ValueError):
                lines.append(f"  {key.upper():12s} = {value}")
    return lines


def main():
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    results_dir = summary_path.parent
    timestamp = summary.get("timestamp", summary_path.stem.replace("training_summary_", ""))

    output_path = Path(args.output) if args.output else results_dir / f"engineering_report_{timestamp}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    saved_files = summary.get("saved_files", {})
    predictions = _load_np(results_dir / saved_files.get("predictions", ""))
    targets = _load_np(results_dir / saved_files.get("targets_test", ""))
    predictions_denorm = _load_np(results_dir / saved_files.get("predictions_denorm", ""))
    targets_denorm = _load_np(results_dir / saved_files.get("targets_denorm", ""))

    samples_info = saved_files.get("samples", {})
    samples_pred = _load_np(results_dir / samples_info.get("pred", "")) if samples_info else None
    samples_true = _load_np(results_dir / samples_info.get("y", "")) if samples_info else None
    if samples_pred is None:
        samples_pred = predictions
    if samples_true is None:
        samples_true = targets
    sample_indices = samples_info.get("indices")

    energies = _prepare_energies(predictions.shape[1] if predictions is not None else (samples_pred.shape[1] if samples_pred is not None else 0))
    energy_edges = np.sqrt(energies[:-1] * energies[1:]) if energies.size > 1 else energies

    rng = np.random.default_rng(args.seed)
    n_spectra = args.spectra_count
    total_samples = predictions.shape[0] if predictions is not None else (samples_pred.shape[0] if samples_pred is not None else 0)
    chosen_indices = np.linspace(0, total_samples - 1, num=min(n_spectra, total_samples), dtype=int) if total_samples else np.array([], dtype=int)
    if total_samples > n_spectra:
        random_idx = rng.choice(total_samples, size=n_spectra, replace=False)
        chosen_indices = np.unique(np.concatenate([chosen_indices, random_idx]))[:n_spectra]

    scatter_mask = None
    if predictions is not None and predictions.size > 0:
        scatter_count = min(5000, predictions.shape[0])
        scatter_mask = rng.choice(predictions.shape[0], size=scatter_count, replace=False)

    with PdfPages(output_path) as pdf:
        # Page 1: title and overall metrics
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        _text_block(ax, f"ANFIS Engineering Report — {timestamp}", [
            f"Summary path : {summary_path}",
            f"Model state  : {results_dir / summary.get('model_state', '')}",
            f"Metrics source: {summary.get('metrics_source', 'unknown')}",
            f"Train/Test    : {summary.get('train_size')} / {summary.get('test_size')}"
        ])
        ax.text(0.02, 0.65, "Dataset settings", fontsize=12, fontweight='bold', va='top')
        y = 0.60
        for key, value in summary.get("dataset_settings", {}).items():
            ax.text(0.03, y, f"{key}: {value}", fontsize=10, va='top')
            y -= 0.035

        ax.text(0.02, 0.50, "Training time (sec)", fontsize=12, fontweight='bold', va='top')
        ax.text(0.03, 0.46, f"Total   : {summary.get('training_time_total', 0):.2f}", fontsize=10, va='top')
        ax.text(0.03, 0.425, f"Vanilla : {summary.get('training_time_vanilla', 0):.2f}", fontsize=10, va='top')
        if summary.get("training_time_shap") is not None:
            ax.text(0.03, 0.39, f"SHAP    : {summary.get('training_time_shap', 0):.2f}", fontsize=10, va='top')

        ax.text(0.02, 0.34, "Metrics (normalized)", fontsize=12, fontweight='bold', va='top')
        for i, line in enumerate(_format_metrics(summary.get("metrics"))):
            ax.text(0.03, 0.30 - i * 0.035, line, fontsize=10, va='top')

        ax.text(0.55, 0.34, "Metrics (denormalized)", fontsize=12, fontweight='bold', va='top')
        for i, line in enumerate(_format_metrics(summary.get("metrics_denorm"))):
            ax.text(0.56, 0.30 - i * 0.035, line, fontsize=10, va='top')

        ax.text(0.02, 0.14, "Diagnostics", fontsize=12, fontweight='bold', va='top')
        diag = summary.get("diagnostics", {})
        y_diag = 0.10
        for section, stats in diag.items():
            ax.text(0.03, y_diag, f"{section}:", fontsize=10, va='top', fontweight='bold')
            y_diag -= 0.035
            if isinstance(stats, dict):
                for key, value in stats.items():
                    ax.text(0.05, y_diag, f"{key}: {value}", fontsize=9, va='top')
                    y_diag -= 0.03
            else:
                ax.text(0.05, y_diag, str(stats), fontsize=9, va='top')
                y_diag -= 0.03
            y_diag -= 0.02

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: band metrics
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        _text_block(ax, "Band metrics (normalized)", _format_band_metrics(summary.get("band_metrics")), start_y=0.95)
        _text_block(ax, "Band metrics (denormalized)", _format_band_metrics(summary.get("band_metrics_denorm")), start_y=0.5)
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: mean spectra (normalized and denorm)
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69), sharex=True)
        for ax, data_pred, data_true, title in [
            (axes[0], predictions, targets, "Mean spectra (normalized)"),
            (axes[1], predictions_denorm, targets_denorm, "Mean spectra (denormalized)")
        ]:
            ax.axis("tight")
            if data_pred is not None and data_true is not None:
                mean_true = np.nanmean(data_true, axis=0)
                mean_pred = np.nanmean(data_pred, axis=0)
                _step(ax, energies, mean_true, label="Ground truth", linewidth=2)
                _step(ax, energies, mean_pred, label="Prediction", linewidth=2, linestyle="--")
                ax.set_title(title)
                ax.set_xlabel("Energy, eV")
                ax.set_ylabel("phi, neutron cm$^{-2}$ s$^{-1}$")
                ax.set_xscale("log")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: sample spectra normalized
        if predictions is not None and targets is not None and chosen_indices.size:
            rows = int(np.ceil(chosen_indices.size / 2))
            fig, axes = plt.subplots(rows, 2, figsize=(8.27, 11.69), sharex=True)
            axes = np.atleast_2d(axes).reshape(-1)
            for ax, idx in zip(axes, chosen_indices):
                _step(ax, energies, targets[idx], label="True", linewidth=1.5)
                _step(ax, energies, predictions[idx], label="Pred", linewidth=1.5, linestyle="--")
                ax.set_title(f"Spectrum #{idx} (normalized)")
                ax.set_xscale("log")
                ax.set_xlabel("Energy, eV")
                ax.set_ylabel("phi")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(fontsize=8)
            for ax in axes[len(chosen_indices):]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Page 5: sample spectra denormalized
        if predictions_denorm is not None and targets_denorm is not None and chosen_indices.size:
            rows = int(np.ceil(chosen_indices.size / 2))
            fig, axes = plt.subplots(rows, 2, figsize=(8.27, 11.69), sharex=True)
            axes = np.atleast_2d(axes).reshape(-1)
            for ax, idx in zip(axes, chosen_indices):
                _step(ax, energies, targets_denorm[idx], label="True", linewidth=1.5)
                _step(ax, energies, predictions_denorm[idx], label="Pred", linewidth=1.5, linestyle="--")
                ax.set_title(f"Spectrum #{idx} (denorm)")
                ax.set_xscale("log")
                ax.set_xlabel("Energy, eV")
                ax.set_ylabel("phi")
                ax.grid(True, which="both", alpha=0.3)
                ax.legend(fontsize=8)
            for ax in axes[len(chosen_indices):]:
                ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Page 6: errors
        if predictions is not None and targets is not None:
            errors = predictions - targets
            mae_per_bin = np.mean(np.abs(errors), axis=0)
            rmse_per_bin = np.sqrt(np.mean(errors ** 2, axis=0))

            fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
            ax = axes[0]
            ax.plot(energies, mae_per_bin, label="MAE", linewidth=1.5)
            ax.plot(energies, rmse_per_bin, label="RMSE", linewidth=1.5)
            ax.set_xscale("log")
            ax.set_ylabel("Error")
            ax.set_title("Errors per energy bin (normalized)")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

            ax = axes[1]
            ax.hist(errors.flatten(), bins=80, alpha=0.8, color="#F17666")
            ax.set_title("Histogram of errors (Prediction - Truth)")
            ax.set_xlabel("Error")
            ax.set_ylabel("Count")
            ax.grid(axis="y", alpha=0.3)

            ax = axes[2]
            if scatter_mask is not None:
                ax.scatter(targets[scatter_mask].flatten(), predictions[scatter_mask].flatten(), s=4, alpha=0.3)
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),
                    np.max([ax.get_xlim(), ax.get_ylim()])
                ]
                ax.plot(lims, lims, 'k--')
            ax.set_title("Scatter plot (normalized)")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Page 7: feature importance
        fi_path = results_dir / summary['saved_files'].get('feature_importance', '')
        shap_fi_path = results_dir / summary.get('shap_files', {}).get('feature_importance_shap', '')
        if fi_path.exists():
            vanilla_fi = pd.read_csv(fi_path, index_col=0)
        else:
            vanilla_fi = None
        shap_fi = pd.read_csv(shap_fi_path, index_col=0) if shap_fi_path and shap_fi_path.exists() else None

        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        if vanilla_fi is not None:
            vanilla_fi.plot(kind="bar", ax=axes[0], legend=False)
            axes[0].set_title("Feature importance (Vanilla)")
            axes[0].set_ylabel("Importance")
            axes[0].grid(axis="y", alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, "No vanilla feature importance", ha='center', va='center')
        if shap_fi is not None:
            shap_fi.plot(kind="bar", color="orange", ax=axes[1], legend=False)
            axes[1].set_title("Feature importance (SHAP)")
            axes[1].set_ylabel("Importance")
            axes[1].grid(axis="y", alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No SHAP feature importance", ha='center', va='center')
        for ax in axes:
            ax.set_xlabel("")
            ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 8: SHAP history
        shap_history_path = results_dir / summary.get('shap_files', {}).get('history', '')
        if shap_history_path.exists():
            with shap_history_path.open("r", encoding="utf-8") as f:
                shap_history = json.load(f)
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            for key, values in shap_history.items():
                if isinstance(values, (list, tuple)):
                    finite = [v for v in values if np.isfinite(v)]
                    if finite:
                        ax.plot(finite, label=key)
            ax.set_title("SHAP training losses")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ Отчёт сохранён: {output_path}")


if __name__ == "__main__":
    main()

