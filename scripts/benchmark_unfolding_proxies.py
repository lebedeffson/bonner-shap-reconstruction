#!/usr/bin/env python3
"""
Benchmark unfolding-style proxy baselines on Bonner data.

Proxies:
1) Direct linear inverse with Tikhonov smoothing.
2) Non-negative Tikhonov (NNLS on augmented system).
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Allow direct execution: python scripts/...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.utils.data_loader import load_validation_data


REAL_TEST_FRACTION = 0.2
REAL_VALIDATION_FRACTION_OF_TEMP = 0.25


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark unfolding-style proxy baselines")
    p.add_argument("--config", default="configs/config_integrated_shap.yaml")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--lambdas", default="1e-5,1e-4,1e-3,1e-2,1e-1")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-md", default="")
    return p.parse_args()


def second_diff_matrix(n: int) -> np.ndarray:
    if n < 3:
        return np.zeros((0, n), dtype=float)
    L = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        L[i, i] = 1.0
        L[i, i + 1] = -2.0
        L[i, i + 2] = 1.0
    return L


def split_real(X, y, random_state):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=REAL_TEST_FRACTION, random_state=random_state
    )
    X_train, _X_val, y_train, _y_val = train_test_split(
        X_temp, y_temp, test_size=REAL_VALIDATION_FRACTION_OF_TEMP, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def metrics(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_weighted": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "r2_mean": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
    }


def estimate_response(y_train, X_train):
    # Solve Y * A ~= X  => A: [n_bins, n_det]
    A, *_ = np.linalg.lstsq(y_train, X_train, rcond=None)
    # Forward model R y ~= x where R: [n_det, n_bins]
    return A.T


def solve_tikhonov_batch(X_test, R, L, lam, nonnegative=False, normalize_sum=False):
    n_det, n_bins = R.shape
    RtR = R.T @ R
    LtL = L.T @ L if L.size else np.zeros((n_bins, n_bins), dtype=float)
    M = RtR + lam * LtL
    Rt = R.T

    if not nonnegative:
        # Precompute linear map K = (RtR + lam LtL)^-1 Rt
        K = np.linalg.solve(M + 1e-10 * np.eye(n_bins), Rt)
        Y = (K @ X_test.T).T
    else:
        # Augmented NNLS system: [R; sqrt(lam)L] y ~= [x; 0]
        A_aug = np.vstack([R, np.sqrt(lam) * L]) if L.size else R.copy()
        Y = np.zeros((X_test.shape[0], n_bins), dtype=float)
        z = np.zeros(A_aug.shape[0], dtype=float)
        for i, x in enumerate(X_test):
            z[:n_det] = x
            y_i, _ = nnls(A_aug, z)
            Y[i] = y_i

    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    if nonnegative:
        Y = np.maximum(Y, 0.0)
    if normalize_sum:
        s = np.sum(Y, axis=1, keepdims=True)
        Y = np.divide(Y, np.maximum(s, 1e-12))
    return Y


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
    n_bins = y.shape[1]
    L = second_diff_matrix(n_bins)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    lambdas = [float(v.strip()) for v in args.lambdas.split(",") if v.strip()]
    records = []

    for seed in seeds:
        X_train, X_test, y_train, y_test = split_real(X, y, random_state=seed)
        R = estimate_response(y_train, X_train)
        for lam in lambdas:
            for method in ["tikhonov_linear", "tikhonov_nnls"]:
                pred = solve_tikhonov_batch(
                    X_test,
                    R=R,
                    L=L,
                    lam=lam,
                    nonnegative=(method == "tikhonov_nnls"),
                    normalize_sum=normalize_sum,
                )
                m = metrics(y_test, pred)
                records.append(
                    {
                        "seed": seed,
                        "lambda": lam,
                        "method": method,
                        **m,
                    }
                )

    # best-by-seed-and-method (by r2_weighted)
    best = {}
    for r in records:
        key = (r["seed"], r["method"])
        if key not in best or r["r2_weighted"] > best[key]["r2_weighted"]:
            best[key] = r

    best_records = list(best.values())
    agg = {}
    for method in sorted(set(r["method"] for r in best_records)):
        vals = [r for r in best_records if r["method"] == method]
        agg[method] = {
            "r2_weighted_mean": float(np.mean([v["r2_weighted"] for v in vals])),
            "r2_weighted_std": float(np.std([v["r2_weighted"] for v in vals])),
            "r2_mean_mean": float(np.mean([v["r2_mean"] for v in vals])),
            "r2_mean_std": float(np.std([v["r2_mean"] for v in vals])),
            "rmse_mean": float(np.mean([v["rmse"] for v in vals])),
            "rmse_std": float(np.std([v["rmse"] for v in vals])),
            "mae_mean": float(np.mean([v["mae"] for v in vals])),
            "mae_std": float(np.std([v["mae"] for v in vals])),
            "lambda_selected_per_seed": [float(v["lambda"]) for v in vals],
        }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = cfg.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = args.output_json or os.path.join(results_dir, f"unfolding_proxies_benchmark_{ts}.json")
    out_md = args.output_md or os.path.join(results_dir, f"unfolding_proxies_benchmark_{ts}.md")

    payload = {
        "config_path": os.path.abspath(args.config),
        "seeds": seeds,
        "lambdas": lambdas,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_bins": int(y.shape[1]),
        "records": records,
        "best_records": best_records,
        "aggregate_best_per_seed": agg,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# Unfolding Proxy Benchmark",
        "",
        f"- Config: `{os.path.abspath(args.config)}`",
        f"- Seeds: `{seeds}`",
        f"- Lambdas: `{lambdas}`",
        f"- Samples: `{X.shape[0]}`, Features: `{X.shape[1]}`, Bins: `{y.shape[1]}`",
        "",
        "Best-per-seed aggregation (criterion: `r2_weighted`):",
        "",
        "| Method | R2_w mean±std | R2_mean mean±std | RMSE mean±std | MAE mean±std |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in sorted(agg.keys()):
        a = agg[method]
        lines.append(
            f"| {method} | "
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
