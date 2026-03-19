"""Общие утилиты оформления matplotlib-графиков."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from constants import Ebins_float_IAEA_Comp


COLORS = {
    "true": "#1D4ED8",
    "pred": "#D97706",
    "accent": "#0F766E",
    "error": "#DC2626",
    "muted": "#64748B",
    "ink": "#0F172A",
    "grid": "#CBD5E1",
    "fill_true": "#93C5FD",
    "fill_pred": "#FDBA74",
    "success": "#16A34A",
    "violet": "#7C3AED",
}


def setup_publication_style():
    """Настройка единого аккуратного стиля для графиков проекта."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": True,
        }
    )


def resolve_energy_axis(n_bins: int) -> np.ndarray:
    """Возвращает физическую ось энергий, если она согласована с числом бинов."""
    if len(Ebins_float_IAEA_Comp) == n_bins + 1:
        return np.asarray(Ebins_float_IAEA_Comp[:-1], dtype=float)
    if len(Ebins_float_IAEA_Comp) == n_bins:
        return np.asarray(Ebins_float_IAEA_Comp, dtype=float)
    return np.arange(n_bins, dtype=float)


def apply_axis_style(ax, *, log_x: bool = False, log_y: bool = False):
    """Применяет единый стиль осей и сетки."""
    if log_x:
        try:
            ax.set_xscale("log")
        except ValueError:
            pass
    if log_y:
        try:
            ax.set_yscale("log")
        except ValueError:
            pass

    ax.grid(True, which="major", color=COLORS["grid"], alpha=0.45, linewidth=0.8)
    ax.grid(True, which="minor", color=COLORS["grid"], alpha=0.18, linewidth=0.6)
    ax.tick_params(colors=COLORS["ink"])
    for spine_name in ("left", "bottom"):
        spine = ax.spines.get(spine_name)
        if spine is not None:
            spine.set_color(COLORS["grid"])
            spine.set_linewidth(0.9)

    try:
        ax.minorticks_on()
    except Exception:
        pass


def finalize_figure(fig, output_path, *, dpi: int = 220):
    """Сохраняет и закрывает фигуру."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_bar_values(ax, *, orientation: str = "vertical", fmt: str = "{:.3f}", fontsize: int = 9):
    """Подписывает значения на столбцах."""
    for patch in ax.patches:
        value = patch.get_height() if orientation == "vertical" else patch.get_width()
        if not np.isfinite(value):
            continue
        if orientation == "vertical":
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_y() + patch.get_height()
            ax.text(x, y, fmt.format(value), ha="center", va="bottom", fontsize=fontsize, color=COLORS["ink"])
        else:
            x = patch.get_x() + patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            ax.text(x, y, f" {fmt.format(value)}", ha="left", va="center", fontsize=fontsize, color=COLORS["ink"])
