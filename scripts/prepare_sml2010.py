#!/usr/bin/env python3
"""Скачивание и подготовка UCI SML2010 в единый CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/274/sml2010.zip"

COLUMNS = [
    "date",
    "time",
    "temperature_comedor_sensor",
    "temperature_habitacion_sensor",
    "weather_temperature",
    "co2_comedor_sensor",
    "co2_habitacion_sensor",
    "humedad_comedor_sensor",
    "humedad_habitacion_sensor",
    "lighting_comedor_sensor",
    "lighting_habitacion_sensor",
    "precipitacion",
    "meteo_exterior_crepusculo",
    "meteo_exterior_viento",
    "meteo_exterior_sol_oest",
    "meteo_exterior_sol_est",
    "meteo_exterior_sol_sud",
    "meteo_exterior_piranometro",
    "exterior_entalpic_1",
    "exterior_entalpic_2",
    "exterior_entalpic_turbo",
    "temperature_exterior_sensor",
    "humedad_exterior_sensor",
    "day_of_week",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SML2010 dataset")
    parser.add_argument("--output", default="data/sml2010.csv", help="Output CSV")
    parser.add_argument("--cache-dir", default="data/sml2010_raw", help="Cache dir")
    return parser.parse_args()


def _read_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="python")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected {len(COLUMNS)} cols, got {df.shape[1]}")
    df.columns = COLUMNS
    return df


def main():
    args = parse_args()
    out_path = Path(args.output).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "sml2010.zip"
    # Качаем архив один раз.
    if not zip_path.exists():
        import urllib.request

        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    df1 = _read_txt(cache_dir / "NEW-DATA-1.T15.txt")
    df2 = _read_txt(cache_dir / "NEW-DATA-2.T15.txt")
    df = pd.concat([df1, df2], ignore_index=True)

    # Временные признаки из даты/времени.
    dt = pd.to_datetime(df["date"] + " " + df["time"], dayfirst=True, errors="coerce")
    df["hour"] = dt.dt.hour.fillna(0).astype(int)
    df["minute"] = dt.dt.minute.fillna(0).astype(int)

    # Убираем исходные строковые поля.
    df = df.drop(columns=["date", "time"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"OK: {out_path}")
    print(f"shape: {df.shape}")
    print(f"columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
