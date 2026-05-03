#!/usr/bin/env python3
"""Diagnose per-target regression metrics from training summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, help="path to training_summary_*.json")
    p.add_argument("--out", default="", help="optional output json")
    return p.parse_args()


def _per_target_stats(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    rows = []
    n_out = y_true.shape[1]
    for j in range(n_out):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        err = yt - yp
        ss_res = float(np.sum(err * err))
        y_mean = float(np.mean(yt))
        ss_tot = float(np.sum((yt - y_mean) ** 2))
        mse = float(np.mean(err * err))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(mse))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        rows.append(
            {
                "target_idx": j,
                "r2": r2,
                "mse": mse,
                "mae": mae,
                "rmse": rmse,
                "ss_res": ss_res,
                "ss_tot": ss_tot,
                "y_mean": y_mean,
                "y_std": float(np.std(yt)),
                "y_min": float(np.min(yt)),
                "y_max": float(np.max(yt)),
                "pred_mean": float(np.mean(yp)),
                "pred_std": float(np.std(yp)),
                "pred_min": float(np.min(yp)),
                "pred_max": float(np.max(yp)),
                "abs_err_p95": float(np.quantile(np.abs(err), 0.95)),
                "abs_err_p99": float(np.quantile(np.abs(err), 0.99)),
                "abs_err_max": float(np.max(np.abs(err))),
            }
        )
    return rows


def main():
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    results_dir = summary_path.parent

    saved = s.get("saved_files", {})
    pred_rel = saved.get("predictions")
    tgt_rel = saved.get("targets_test")
    if not pred_rel or not tgt_rel:
        raise SystemExit("summary has no saved_files.predictions/targets_test")

    y_pred = np.load(results_dir / pred_rel)
    y_true = np.load(results_dir / tgt_rel)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    if y_true.shape != y_pred.shape:
        min_dim = min(y_true.shape[-1], y_pred.shape[-1])
        y_true = y_true[..., :min_dim]
        y_pred = y_pred[..., :min_dim]

    rows = _per_target_stats(y_true, y_pred)
    r2_mean = float(np.mean([r["r2"] for r in rows])) if rows else float("nan")
    mse_mean = float(np.mean([r["mse"] for r in rows])) if rows else float("nan")
    mae_mean = float(np.mean([r["mae"] for r in rows])) if rows else float("nan")

    out = {
        "summary": str(summary_path),
        "tag": s.get("tag"),
        "timestamp": s.get("timestamp"),
        "metrics_source": s.get("metrics_source"),
        "model_mode": s.get("model_mode"),
        "fallback_used": s.get("fallback_used"),
        "split_hash": s.get("split_hash"),
        "config_sha256": s.get("config_sha256"),
        "effective_config_sha256": s.get("effective_config_sha256"),
        "train_size": s.get("train_size"),
        "test_size": s.get("test_size"),
        "real_test_count": s.get("real_test_count"),
        "r2_summary": (s.get("metrics") or {}).get("r2"),
        "r2_vanilla": (s.get("vanilla_metrics") or {}).get("r2"),
        "r2_ea_raw": (s.get("shap_metrics") or {}).get("r2"),
        "recomputed": {
            "r2_mean": r2_mean,
            "mse_mean": mse_mean,
            "mae_mean": mae_mean,
        },
        "per_target": rows,
    }

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = summary_path.with_name(summary_path.stem + "_diagnostics.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")
    print(json.dumps(out["recomputed"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
