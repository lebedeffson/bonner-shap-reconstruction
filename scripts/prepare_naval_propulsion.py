#!/usr/bin/env python3
"""Скачивание и подготовка UCI Naval Propulsion в единый CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip"

COLUMNS = [
    "lever_position",
    "ship_speed",
    "gas_turbine_shaft_torque",
    "gas_turbine_rate_of_revolutions",
    "gas_generator_rate_of_revolutions",
    "starboard_propeller_torque",
    "port_propeller_torque",
    "hp_turbine_exit_temperature",
    "gt_compressor_inlet_air_temperature",
    "gt_compressor_outlet_air_temperature",
    "hp_turbine_exit_pressure",
    "gt_compressor_inlet_air_pressure",
    "gt_compressor_outlet_air_pressure",
    "gas_turbine_exhaust_gas_pressure",
    "turbine_injection_control",
    "fuel_flow",
    "gt_compressor_decay_state_coefficient",
    "gt_turbine_decay_state_coefficient",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Naval Propulsion dataset")
    parser.add_argument("--output", default="data/naval_propulsion.csv", help="Output CSV")
    parser.add_argument("--cache-dir", default="data/naval_raw", help="Cache dir")
    return parser.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.output).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "naval_propulsion.zip"
    if not zip_path.exists():
        import urllib.request

        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)

    with ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)

    data_file = cache_dir / "UCI CBM Dataset" / "data.txt"
    if not data_file.exists():
        raise FileNotFoundError(f"Не найден файл датасета: {data_file}")

    df = pd.read_csv(data_file, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"Ожидалось {len(COLUMNS)} столбцов, получено {df.shape[1]}")

    df.columns = COLUMNS
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True)

    target_cols = [
        "gt_compressor_decay_state_coefficient",
        "gt_turbine_decay_state_coefficient",
    ]
    feature_cols = [c for c in df.columns if c not in target_cols]

    # 1) Константные признаки
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    # 2) Точные дубликаты признаков
    duplicate_cols = []
    seen = []
    for col in feature_cols:
        is_dup = False
        for ref in seen:
            if df[col].equals(df[ref]):
                duplicate_cols.append(col)
                is_dup = True
                break
        if not is_dup:
            seen.append(col)
    if duplicate_cols:
        df = df.drop(columns=duplicate_cols)
        feature_cols = [c for c in feature_cols if c not in duplicate_cols]

    # 3) Почти идеальная корреляция между признаками
    corr = df[feature_cols].corr().abs()
    high_corr_drop = set()
    for i, left in enumerate(feature_cols):
        if left in high_corr_drop:
            continue
        for right in feature_cols[i + 1 :]:
            if right in high_corr_drop:
                continue
            if corr.loc[left, right] > 0.9999:
                high_corr_drop.add(right)
    if high_corr_drop:
        df = df.drop(columns=sorted(high_corr_drop))
        feature_cols = [c for c in feature_cols if c not in high_corr_drop]

    df.to_csv(out_path, index=False)

    print(f"OK: {out_path}")
    print(f"shape: {df.shape}")
    print(f"features: {len(feature_cols)}")
    if constant_cols:
        print(f"dropped constant: {', '.join(constant_cols)}")
    if duplicate_cols:
        print(f"dropped duplicates: {', '.join(duplicate_cols)}")
    if high_corr_drop:
        print(f"dropped high corr: {', '.join(sorted(high_corr_drop))}")
    print(f"targets: gt_compressor_decay_state_coefficient, gt_turbine_decay_state_coefficient")


if __name__ == "__main__":
    main()
