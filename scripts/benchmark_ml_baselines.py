#!/usr/bin/env python3
"""
Benchmark classical/modern tabular ML baselines on the same real-data split.
Outputs compact JSON + Markdown report with R2_weighted / R2_mean / RMSE / MAE.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from src.utils.config_loader import load_config
from src.utils.data_loader import load_validation_data


REAL_TEST_FRACTION = 0.2
REAL_VALIDATION_FRACTION_OF_TEMP = 0.25


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark tabular ML baselines")
    p.add_argument("--config", default="configs/config_integrated_shap.yaml")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-md", default="")
    return p.parse_args()


def _metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_weighted": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "r2_mean": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }


def _split_real(X, y, random_state):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=REAL_TEST_FRACTION, random_state=random_state
    )
    X_train, _X_val, y_train, _y_val = train_test_split(
        X_temp, y_temp, test_size=REAL_VALIDATION_FRACTION_OF_TEMP, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def _build_models(seed):
    return {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, random_state=seed, n_jobs=-1
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, random_state=seed, n_jobs=-1
        ),
        "HGB": MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=seed, max_depth=8, learning_rate=0.05)
        ),
        "MLPRegressor": MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=700,
            random_state=seed,
        ),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dcfg = cfg["dataset"]
    normalize_sum = bool(dcfg.get("normalize_sum", False))

    X, y, _sum = load_validation_data(
        dcfg["validation_data"],
        normalize_sum=normalize_sum,
        dataset_config=dcfg,
    )
    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(np.asarray(y, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    records = []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = _split_real(X, y, random_state=seed)
        models = _build_models(seed)
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            pred = np.nan_to_num(np.asarray(model.predict(X_te), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            m = _metrics(y_te, pred)
            records.append({"seed": seed, "model": name, **m})

    # Aggregate
    by_model = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)

    agg = {}
    for model, vals in by_model.items():
        agg[model] = {}
        for k in ["r2_weighted", "r2_mean", "rmse", "mae"]:
            arr = [v[k] for v in vals]
            agg[model][f"{k}_mean"] = float(np.mean(arr))
            agg[model][f"{k}_std"] = float(np.std(arr))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = cfg.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = args.output_json or os.path.join(results_dir, f"ml_baselines_benchmark_{ts}.json")
    out_md = args.output_md or os.path.join(results_dir, f"ml_baselines_benchmark_{ts}.md")

    payload = {
        "config_path": os.path.abspath(args.config),
        "seeds": seeds,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "records": records,
        "aggregate": agg,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# ML Baselines Benchmark",
        "",
        f"- Config: `{os.path.abspath(args.config)}`",
        f"- Seeds: `{seeds}`",
        f"- Samples: `{X.shape[0]}`; Features: `{X.shape[1]}`",
        "",
        "| Model | R2_w mean±std | R2_mean mean±std | RMSE mean±std | MAE mean±std |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in sorted(agg.keys()):
        a = agg[model]
        lines.append(
            f"| {model} | "
            f"{a['r2_weighted_mean']:.6f} ± {a['r2_weighted_std']:.6f} | "
            f"{a['r2_mean_mean']:.6f} ± {a['r2_mean_std']:.6f} | "
            f"{a['rmse_mean']:.6f} ± {a['rmse_std']:.6f} | "
            f"{a['mae_mean']:.6f} ± {a['mae_std']:.6f} |"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
