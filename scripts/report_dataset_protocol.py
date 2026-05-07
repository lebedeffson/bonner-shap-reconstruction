#!/usr/bin/env python3
"""
Generate dataset/protocol table for manuscript reproducibility.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Allow direct execution: python scripts/...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.data_loader import load_data, resolve_feature_columns, resolve_target_columns


REAL_TEST_FRACTION = 0.2
REAL_VALIDATION_FRACTION_OF_TEMP = 0.25


def parse_args():
    p = argparse.ArgumentParser(description="Dataset protocol report")
    p.add_argument("--config", default="configs/config_integrated_shap.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default="")
    p.add_argument("--output-md", default="")
    return p.parse_args()


def _split_counts(n, seed):
    idx = np.arange(n)
    temp, test = train_test_split(idx, test_size=REAL_TEST_FRACTION, random_state=seed)
    train, val = train_test_split(temp, test_size=REAL_VALIDATION_FRACTION_OF_TEMP, random_state=seed)
    return int(train.size), int(val.size), int(test.size)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    data_path = dcfg["validation_data"]
    df = load_data(data_path)

    feature_cols = resolve_feature_columns(df, dcfg)
    target_cols = resolve_target_columns(df, dcfg)
    feature_set = set(feature_cols)
    target_set = set(target_cols)
    meta_cols = [c for c in df.columns if c not in feature_set and c not in target_set]
    n = int(len(df))
    n_train, n_val, n_test = _split_counts(n, args.seed)

    y = df[target_cols].to_numpy(dtype=float)
    y_clean = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y_round = np.round(y_clean, 10)
    unique_spectra = int(np.unique(y_round, axis=0).shape[0])
    duplicate_spectra = int(n - unique_spectra)

    payload = {
        "config_path": os.path.abspath(args.config),
        "data_path": os.path.abspath(data_path),
        "n_samples": n,
        "n_features": int(len(feature_cols)),
        "n_targets": int(len(target_cols)),
        "split": {
            "train": n_train,
            "validation": n_val,
            "test": n_test,
            "train_frac": float(n_train / n),
            "validation_frac": float(n_val / n),
            "test_frac": float(n_test / n),
        },
        "meta_columns": meta_cols,
        "meta_column_count": int(len(meta_cols)),
        "unique_spectra": unique_spectra,
        "duplicate_spectra": duplicate_spectra,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = cfg.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = args.output_json or os.path.join(results_dir, f"dataset_protocol_{ts}.json")
    out_md = args.output_md or os.path.join(results_dir, f"dataset_protocol_{ts}.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# Dataset Protocol Report",
        "",
        f"- Config: `{os.path.abspath(args.config)}`",
        f"- Data: `{os.path.abspath(data_path)}`",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| N samples | {n} |",
        f"| Features | {len(feature_cols)} |",
        f"| Spectrum bins (targets) | {len(target_cols)} |",
        f"| Train / Val / Test | {n_train} / {n_val} / {n_test} |",
        f"| Unique spectra (rounded 1e-10) | {unique_spectra} |",
        f"| Duplicate spectra | {duplicate_spectra} |",
        f"| Metadata columns outside feature/target set | {len(meta_cols)} |",
    ]
    if meta_cols:
        lines.extend(["", "Metadata columns:", ""])
        lines.extend([f"- `{c}`" for c in meta_cols])

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()

