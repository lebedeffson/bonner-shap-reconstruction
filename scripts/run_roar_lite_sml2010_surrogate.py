#!/usr/bin/env python3
"""ROAR-lite (surrogate retrain) for SML2010 using ANFIS-derived rankings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config
from src.utils.data_loader import load_validation_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed", required=True)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--k-list", default="1,2")
    p.add_argument("--mask", choices=["permute", "mean", "noise"], default="permute")
    p.add_argument("--out", default="results/roar_lite_sml2010_surrogate_3seed_20260504.json")
    return p.parse_args()


def _apply_mask(X: np.ndarray, cols: list[int], mode: str, rng: np.random.Generator) -> np.ndarray:
    X2 = X.copy()
    for j in cols:
        if mode == "permute":
            idx = rng.permutation(X2.shape[0])
            X2[:, j] = X2[idx, j]
        elif mode == "mean":
            X2[:, j] = float(np.mean(X2[:, j]))
        else:
            std = float(np.std(X2[:, j]))
            X2[:, j] = X2[:, j] + rng.normal(0.0, 0.1 * std + 1e-8, size=X2.shape[0])
    return X2


def _predict_anfis(manager: ANFISManager, state_path: str, X: np.ndarray, y_dim: int) -> np.ndarray:
    model = manager.create_model(input_dim=X.shape[1], output_dim=y_dim, verbose=False)
    state = torch.load(state_path, map_location="cpu")
    model.network.load_state_dict(state, strict=False)
    model.network.eval()
    with torch.no_grad():
        return model.network(torch.tensor(X, dtype=torch.float32)).cpu().numpy()


def _load_importance(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = "importance" if "importance" in df.columns else df.columns[-1]
    vals = np.asarray(df[col], dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, 0.0)
    s = float(vals.sum())
    return vals / s if s > 1e-12 else np.full_like(vals, 1.0 / max(1, vals.size))


def _permutation_importance(
    manager: ANFISManager,
    state_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_dim: int,
    mask_mode: str,
    seed: int,
) -> np.ndarray:
    pred0 = _predict_anfis(manager, state_path, X_test, y_dim)
    mse0 = float(mean_squared_error(y_test, pred0))
    vals = np.zeros(X_test.shape[1], dtype=float)
    for j in range(X_test.shape[1]):
        rng = np.random.default_rng(seed + 9973 * (j + 1))
        Xm = _apply_mask(X_test, [j], mask_mode, rng)
        predm = _predict_anfis(manager, state_path, Xm, y_dim)
        vals[j] = max(0.0, float(mean_squared_error(y_test, predm) - mse0))
    s = float(vals.sum())
    return vals / s if s > 1e-12 else np.full_like(vals, 1.0 / max(1, vals.size))


def _split(base_cfg: dict, seed: int):
    ds = dict(base_cfg["dataset"])
    ds["random_state"] = int(seed)
    X_real, y_real, _ = load_validation_data(
        ds.get("validation_data") or ds.get("train_data"),
        normalize_sum=bool(ds.get("normalize_sum", False)),
        dataset_config=ds,
    )
    X_real = np.asarray(X_real, dtype=float)
    y_real = np.asarray(y_real, dtype=float)
    split_strategy = str(ds.get("split_strategy", "random")).strip().lower()
    if split_strategy in {"time_block", "time", "temporal"}:
        n = X_real.shape[0]
        n_test = max(1, int(round(n * 0.2)))
        n_test = min(n_test, n - 2)
        n_temp = n - n_test
        n_val = max(1, int(round(n_temp * 0.25)))
        n_train = n_temp - n_val
        X_train, y_train = X_real[:n_train], y_real[:n_train]
        X_test, y_test = X_real[n_temp:], y_real[n_temp:]
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_real, y_real, test_size=0.2, random_state=seed
        )
        X_train, _, y_train, _ = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)
    return np.nan_to_num(X_train), np.nan_to_num(y_train), np.nan_to_num(X_test), np.nan_to_num(y_test)


def _fit_r2(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, seed: int) -> float:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=600,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        ))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return float(r2_score(y_test, pred, multioutput="uniform_average"))


def main():
    args = parse_args()
    ms_path = Path(args.multiseed).resolve()
    ms = json.loads(ms_path.read_text(encoding="utf-8"))
    repo_root = ms_path.parent.parent
    seed_set = {int(x.strip()) for x in args.seeds.split(",") if x.strip()}
    k_list = sorted({int(x.strip()) for x in args.k_list.split(",") if x.strip()})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in ms["runs"]:
        seed = int(run["seed"])
        if seed not in seed_set:
            continue
        summary_path = Path(run["summary_path"])
        if not summary_path.is_absolute():
            summary_path = (repo_root / summary_path).resolve()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        cfg_path = Path(summary.get("config_path", ""))
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()
        if not cfg_path.exists():
            cfg_path = (repo_root / ms["config"]).resolve()
        cfg = load_config(str(cfg_path))
        manager = ANFISManager(cfg)

        X_train, y_train, X_test, y_test = _split(cfg, seed)
        y_dim = y_test.shape[1]
        n_features = X_train.shape[1]

        sdir = summary_path.parent
        imp_grad = _load_importance(sdir / summary["saved_files"]["feature_importance"])
        imp_ea = _load_importance(sdir / summary["saved_files"]["shap"]["feature_importance_shap"])
        imp_perm = _permutation_importance(manager, summary["model_state_path"], X_test, y_test, y_dim, args.mask, seed)
        methods = {"vanilla_gradient": imp_grad, "eaar_internal": imp_ea, "permutation": imp_perm}

        base_r2 = _fit_r2(X_train, y_train, X_test, y_test, seed)
        for method, imp in methods.items():
            order_desc = np.argsort(imp)[::-1]
            order_asc = np.argsort(imp)
            for k in k_list:
                kk = max(1, min(k, n_features // 2))
                for mode, idx in [("top", order_desc[:kk]), ("bottom", order_asc[:kk])]:
                    keep_idx = [i for i in range(n_features) if i not in set(idx.tolist())]
                    r2_masked = _fit_r2(X_train[:, keep_idx], y_train, X_test[:, keep_idx], y_test, seed)
                    rows.append({
                        "seed": seed,
                        "method": method,
                        "k": kk,
                        "mode": mode,
                        "base_r2": base_r2,
                        "masked_r2": r2_masked,
                        "drop_r2": float(base_r2 - r2_masked),
                    })

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index=["seed", "method", "k"], columns="mode", values="drop_r2", aggfunc="mean").reset_index()
    piv["roar_gap_top_minus_bottom"] = piv["top"] - piv["bottom"]
    agg = piv.groupby(["method", "k"])[["top", "bottom", "roar_gap_top_minus_bottom"]].agg(["mean", "std"]).reset_index()

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "multiseed": str(ms_path),
        "seeds_used": sorted(seed_set),
        "k_list": k_list,
        "rows": json.loads(df.to_json(orient="records")),
        "pivot": json.loads(piv.to_json(orient="records")),
        "aggregate": json.loads(agg.to_json(orient="records")),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    md_path = out_path.with_suffix(".md")
    piv.to_csv(csv_path, index=False)
    md_path.write_text(
        "# ROAR-lite SML2010 (surrogate)\n\n"
        f"JSON: `{out_path}`\n"
        f"CSV: `{csv_path}`\n\n"
        "## Seed-level\n\n"
        + piv.to_csv(index=False)
        + "\n## Aggregate\n\n"
        + agg.to_csv(index=False),
        encoding="utf-8",
    )
    print(out_path)
    print(csv_path)
    print(md_path)


if __name__ == "__main__":
    main()
