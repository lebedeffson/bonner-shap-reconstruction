#!/usr/bin/env python3
"""Лёгкий sweep для vanilla ANFIS по конфигу датасета."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.anfis_manager import ANFISManager
from src.utils.data_loader import load_validation_data
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Sweep vanilla ANFIS config")
    p.add_argument("--config", required=True)
    p.add_argument("--rules", default="8,12,16,20")
    p.add_argument("--reg", default="0.001,0.01,0.05")
    p.add_argument("--seeds", default="41,42,43")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _parse_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main():
    args = parse_args()
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds = base["dataset"]

    X, y, _ = load_validation_data(
        ds.get("train_data") or ds.get("validation_data"),
        normalize_sum=ds.get("normalize_sum", False),
        dataset_config=ds,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ds.get("test_size", 0.2), random_state=ds.get("random_state", 42)
    )
    X_train = np.nan_to_num(np.asarray(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(np.asarray(y_train), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(np.asarray(X_test), nan=0.0, posinf=0.0, neginf=0.0)
    y_test = np.nan_to_num(np.asarray(y_test), nan=0.0, posinf=0.0, neginf=0.0)

    rules = _parse_ints(args.rules)
    regs = _parse_floats(args.reg)
    seeds = _parse_ints(args.seeds)

    rows = []
    for n_rules in rules:
        for reg in regs:
            for seed in seeds:
                cfg = copy.deepcopy(base)
                cfg["model"]["num_rules"] = n_rules
                cfg["model"]["reg_lambda"] = reg
                cfg["model"]["seed"] = seed
                mgr = ANFISManager(cfg)
                if hasattr(X, "columns"):
                    mgr.set_feature_names(X.columns)
                try:
                    res = mgr.train_vanilla_model(X_train, y_train, X_test, y_test)
                    m = res["metrics"]
                    rows.append(
                        {
                            "num_rules": n_rules,
                            "reg_lambda": reg,
                            "seed": seed,
                            "mse": float(m["mse"]),
                            "rmse": float(m["rmse"]),
                            "mae": float(m["mae"]),
                            "r2": float(m["r2"]),
                            "ok": True,
                        }
                    )
                except Exception as e:
                    rows.append(
                        {
                            "num_rules": n_rules,
                            "reg_lambda": reg,
                            "seed": seed,
                            "mse": np.nan,
                            "rmse": np.nan,
                            "mae": np.nan,
                            "r2": np.nan,
                            "ok": False,
                            "error": str(e),
                        }
                    )

    out_df = pd.DataFrame(rows).sort_values("r2", ascending=False, na_position="last")
    out_dir = Path("results/sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.config).stem
    out_path = Path(args.out) if args.out else out_dir / f"{stem}_sweep.csv"
    out_df.to_csv(out_path, index=False)
    print(f"saved: {out_path}")
    print(out_df.head(15).to_string(index=False))

    best = out_df[out_df["ok"]].head(1)
    if len(best):
        print("\nbest:")
        print(best.to_string(index=False))
        print("\nbest_json:")
        print(json.dumps(best.iloc[0].to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
