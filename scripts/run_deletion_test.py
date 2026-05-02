#!/usr/bin/env python3
"""Deletion test for trained ANFIS checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config
from src.utils.data_loader import load_validation_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True)
    p.add_argument("--importance-kind", choices=["shap", "vanilla"], default="shap")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--mask", choices=["permute", "mean", "noise"], default="permute")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_mask(X: np.ndarray, cols: list[int], mode: str, rng: np.random.Generator) -> np.ndarray:
    X2 = X.copy()
    for j in cols:
        if mode == "permute":
            idx = rng.permutation(X2.shape[0])
            X2[:, j] = X2[idx, j]
        elif mode == "mean":
            X2[:, j] = float(np.mean(X2[:, j]))
        else:  # noise
            std = float(np.std(X2[:, j]))
            X2[:, j] = X2[:, j] + rng.normal(0.0, 0.1 * std + 1e-8, size=X2.shape[0])
    return X2


def predict(manager: ANFISManager, state_path: str, X: np.ndarray, y_dim: int) -> np.ndarray:
    model = manager.create_model(input_dim=X.shape[1], output_dim=y_dim, verbose=False)
    state = torch.load(state_path, map_location="cpu")
    model.network.load_state_dict(state, strict=False)
    model.network.eval()
    with torch.no_grad():
        return model.network(torch.tensor(X, dtype=torch.float32)).cpu().numpy()


def main():
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    config = load_config(summary["config_path"])
    ds = config["dataset"]
    random_state = int(ds.get("random_state", 42))

    X_real, y_real, _ = load_validation_data(
        ds.get("validation_data") or ds.get("train_data"),
        normalize_sum=bool(ds.get("normalize_sum", False)),
        dataset_config=ds,
    )
    X_real = np.asarray(X_real, dtype=float)
    y_real = np.asarray(y_real, dtype=float)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=random_state
    )
    # same second split as train.py (not used directly, but preserves test identity)
    _ = train_test_split(X_temp, y_temp, test_size=0.25, random_state=random_state)

    manager = ANFISManager(config)
    if isinstance(X_real, pd.DataFrame):
        manager.set_feature_names(X_real.columns)
    y_dim = y_test.shape[1]

    pred0 = predict(manager, summary["model_state_path"], X_test, y_dim)
    base_mse = float(mean_squared_error(y_test, pred0))

    results_dir = Path(args.summary).resolve().parent
    if args.importance_kind == "shap":
        fi_name = summary["saved_files"]["shap"]["feature_importance_shap"]
    else:
        fi_name = summary["saved_files"]["feature_importance"]
    fi = pd.read_csv(results_dir / fi_name)
    values = fi.iloc[:, -1].to_numpy(dtype=float)
    order = np.argsort(values)[::-1]
    k = max(1, min(args.k, values.size // 2))
    top = order[:k].tolist()
    bottom = order[-k:].tolist()
    rng = np.random.default_rng(args.seed)

    pred_top = predict(manager, summary["model_state_path"], apply_mask(X_test, top, args.mask, rng), y_dim)
    pred_bottom = predict(manager, summary["model_state_path"], apply_mask(X_test, bottom, args.mask, rng), y_dim)
    mse_top = float(mean_squared_error(y_test, pred_top))
    mse_bottom = float(mean_squared_error(y_test, pred_bottom))
    d_top = mse_top - base_mse
    d_bottom = mse_bottom - base_mse
    ratio = d_top / (d_bottom + 1e-12)

    out = {
        "summary": str(Path(args.summary).resolve()),
        "importance_kind": args.importance_kind,
        "mask": args.mask,
        "k": k,
        "base_mse": base_mse,
        "delta_mse_top": d_top,
        "delta_mse_bottom": d_bottom,
        "ratio_top_bottom": ratio,
        "top_idx": top,
        "bottom_idx": bottom,
    }
    out_path = results_dir / f"deletion_{Path(args.summary).stem}_{args.importance_kind}_{args.mask}_k{k}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
