#!/usr/bin/env python3
"""
Dose-oriented metrics from saved prediction/target arrays in training summary.

If no conversion coefficients are provided, reports flux-integral proxy metrics
(all bins weight=1). With coefficients, computes weighted dose-like scalar.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Dose/flux scalar metrics from summary")
    p.add_argument("--summary", required=True, help="Path to training_summary_*.json")
    p.add_argument("--coeffs", default="", help="Path to coefficients (.npy or .csv/.txt)")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-md", default="")
    return p.parse_args()


def _load_coeffs(path, n_bins):
    if not path:
        return np.ones(n_bins, dtype=float), "flat_ones_proxy"
    if path.lower().endswith(".npy"):
        w = np.load(path).astype(float).reshape(-1)
    else:
        w = np.loadtxt(path, delimiter=",", dtype=float).reshape(-1)
    if w.size != n_bins:
        raise ValueError(f"coeffs size mismatch: got {w.size}, expected {n_bins}")
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    return w, os.path.abspath(path)


def _pearson(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    a = a - np.mean(a)
    b = b - np.mean(b)
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / den)


def main():
    args = parse_args()
    with open(args.summary, "r", encoding="utf-8") as f:
        s = json.load(f)

    res_dir = os.path.dirname(os.path.abspath(args.summary))
    saved = s.get("saved_files", {})
    pred_name = saved.get("predictions_denorm") or saved.get("predictions")
    true_name = saved.get("targets_denorm") or saved.get("targets_test")
    if not pred_name or not true_name:
        raise ValueError("summary does not contain predictions/targets file names")

    pred = np.load(os.path.join(res_dir, pred_name)).astype(float)
    true = np.load(os.path.join(res_dir, true_name)).astype(float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    true = np.nan_to_num(true, nan=0.0, posinf=0.0, neginf=0.0)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, true {true.shape}")

    n_bins = int(pred.shape[1])
    w, coeffs_source = _load_coeffs(args.coeffs, n_bins)
    dose_true = true @ w
    dose_pred = pred @ w

    abs_err = np.abs(dose_pred - dose_true)
    rel_err = abs_err / np.maximum(np.abs(dose_true), 1e-12)
    bias = (dose_pred - dose_true) / np.maximum(np.abs(dose_true), 1e-12)

    payload = {
        "summary_path": os.path.abspath(args.summary),
        "coeffs_source": coeffs_source,
        "n_samples": int(pred.shape[0]),
        "n_bins": n_bins,
        "dose_mae_abs": float(np.mean(abs_err)),
        "dose_rmse_abs": float(np.sqrt(np.mean((dose_pred - dose_true) ** 2))),
        "dose_mape_mean": float(np.mean(rel_err)),
        "dose_bias_mean": float(np.mean(bias)),
        "dose_bias_std": float(np.std(bias)),
        "dose_corr_pearson": _pearson(dose_true, dose_pred),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.output_json or os.path.join(res_dir, f"dose_metrics_{ts}.json")
    out_md = args.output_md or os.path.join(res_dir, f"dose_metrics_{ts}.md")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# Dose Metrics Report",
        "",
        f"- Summary: `{os.path.abspath(args.summary)}`",
        f"- Coefficients: `{coeffs_source}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Dose MAE (abs) | {payload['dose_mae_abs']:.6e} |",
        f"| Dose RMSE (abs) | {payload['dose_rmse_abs']:.6e} |",
        f"| Dose MAPE mean | {payload['dose_mape_mean']:.6f} |",
        f"| Dose bias mean | {payload['dose_bias_mean']:.6f} |",
        f"| Dose bias std | {payload['dose_bias_std']:.6f} |",
        f"| Dose Pearson corr | {payload['dose_corr_pearson']:.6f} |",
    ]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()

