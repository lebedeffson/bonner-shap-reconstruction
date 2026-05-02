#!/usr/bin/env python3
"""Скачивание и валидация датасета UCI Energy Efficiency."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_URLS = [
    "https://raw.githubusercontent.com/DarrenCook/h2o/bk/datasets/ENB2012_data.csv",
    "https://raw.githubusercontent.com/JamshedAli18/Energy-Consumption-Regression/main/ENB2012_data.csv",
]

EXPECTED_COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "Y1", "Y2"]
RENAME_MAP = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
    "Y1": "heating_load",
    "Y2": "cooling_load",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download Energy Efficiency dataset")
    parser.add_argument(
        "--output",
        default="data/energy_efficiency.csv",
        help="Путь для сохранения CSV",
    )
    return parser.parse_args()


def _try_load(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    # Иногда встречаются хвостовые пустые колонки.
    df = df.dropna(axis=1, how="all")
    return df


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Нет ожидаемых колонок: {missing}")

    df = df[EXPECTED_COLUMNS].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any").reset_index(drop=True)

    if len(df) < 700:
        raise ValueError(f"Слишком мало строк после очистки: {len(df)}")
    return df


def main():
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    dataset = None
    used_url = None
    for url in DEFAULT_URLS:
        try:
            dataset = _validate(_try_load(url))
            used_url = url
            break
        except Exception as exc:  # noqa: PERF203
            last_error = exc

    if dataset is None:
        raise RuntimeError(f"Не удалось скачать датасет. Последняя ошибка: {last_error}")

    dataset = dataset.rename(columns=RENAME_MAP)
    dataset.to_csv(output_path, index=False)

    print(f"OK: {output_path}")
    print(f"source: {used_url}")
    print(f"shape: {dataset.shape}")
    print(f"columns: {', '.join(dataset.columns)}")


if __name__ == "__main__":
    main()
