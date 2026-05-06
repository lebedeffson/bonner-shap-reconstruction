#!/usr/bin/env python3
"""
Faithfulness report (top/random/bottom deletion) for ANFIS checkpoints.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.models.anfis_manager import ANFISManager
from src.utils.config_loader import load_config
from src.utils.data_loader import load_validation_data, resolve_feature_columns_from_config


REAL_TEST_FRACTION = 0.2
REAL_VALIDATION_FRACTION_OF_TEMP = 0.25


def _split_real_data_for_eval(X_real, y_real, sum_real, normalize_sum, random_state):
    if normalize_sum and sum_real is not None:
        X_temp, X_test, y_temp, y_test, _, sum_test = train_test_split(
            X_real, y_real, sum_real, test_size=REAL_TEST_FRACTION, random_state=random_state
        )
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_real, y_real, test_size=REAL_TEST_FRACTION, random_state=random_state
        )
        sum_test = None

    _x_shap, _x_val, _y_shap, _y_val = train_test_split(
        X_temp, y_temp, test_size=REAL_VALIDATION_FRACTION_OF_TEMP, random_state=random_state
    )
    return np.asarray(X_test), np.asarray(y_test), sum_test


def _predict(model, X_np):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
        pred = model(X_tensor).detach().cpu().numpy()
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred


def _mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def _mask_columns(X, cols, mode, rng):
    X_masked = X.copy()
    for j in cols:
        if mode == "permute":
            X_masked[:, j] = rng.permutation(X_masked[:, j])
        elif mode == "mean":
            X_masked[:, j] = np.mean(X_masked[:, j])
        else:
            raise ValueError(f"Unknown masking mode: {mode}")
    return X_masked


def _auc_from_curve(values):
    if len(values) == 1:
        return float(values[0])
    x = np.arange(1, len(values) + 1, dtype=float)
    y = np.asarray(values, dtype=float)
    # NumPy >=2.0: trapz удален в пользу trapezoid
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x=x))
    return float(np.trapezoid(y, x=x))


def main():
    parser = argparse.ArgumentParser(description="Faithfulness top/random/bottom report")
    parser.add_argument("--summary", required=True, help="Path to training_summary_*.json")
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--masking", choices=["permute", "mean"], default="permute")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--importance-key", choices=["shap", "feature"], default="shap")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    config_path = summary["config_path"]
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    normalize_sum = bool(dataset_cfg.get("normalize_sum", False))
    random_state = int(dataset_cfg.get("random_state", 42))

    X_real, y_real, sum_real = load_validation_data(
        dataset_cfg["validation_data"],
        normalize_sum=normalize_sum,
        dataset_config=dataset_cfg,
    )
    X_test, y_test, _sum_test = _split_real_data_for_eval(
        X_real, y_real, sum_real, normalize_sum, random_state
    )
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

    manager = ANFISManager(config)
    model = manager.create_model(
        input_dim=X_test.shape[1],
        output_dim=y_test.shape[1],
        verbose=False,
    )
    state_path = summary.get("model_state_path")
    if not state_path:
        state_path = os.path.join(config["output"]["results_dir"], summary["model_state"])
    state_dict = torch.load(state_path, map_location="cpu")
    model.network.load_state_dict(state_dict, strict=False)

    y_base = _predict(model.network, X_test)
    base_mse = _mse(y_test, y_base)

    saved = summary.get("saved_files", {})
    if args.importance_key == "shap":
        fi_name = saved.get("shap", {}).get("feature_importance_shap")
    else:
        fi_name = saved.get("feature_importance")

    if fi_name:
        fi_path = os.path.join(config["output"]["results_dir"], fi_name)
        fi_df = pd.read_csv(fi_path)
        if "importance" not in fi_df.columns:
            raise ValueError(f"'importance' column not found: {fi_path}")
        importance = np.asarray(fi_df["importance"], dtype=float)
        importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
        if importance.size != X_test.shape[1]:
            feature_names = resolve_feature_columns_from_config(dataset_cfg)
            if not (feature_names and len(feature_names) == X_test.shape[1] and len(fi_df) == len(feature_names)):
                raise ValueError("Importance size does not match feature count.")
    else:
        # Fallback для vanilla summary: берём важность из coeffs модели.
        coeffs = model.network.state_dict()["coeffs"].detach().cpu().numpy()
        coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=0.0, neginf=0.0)
        if coeffs.ndim == 3:
            importance = np.sum(np.mean(np.abs(coeffs[:, :-1, :]), axis=2), axis=0)
        else:
            importance = np.sum(np.abs(coeffs[:, :-1, 0]), axis=0)
        importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)

    order_desc = np.argsort(-importance)
    order_asc = np.argsort(importance)
    k_max = min(args.k_max, X_test.shape[1])
    rng = np.random.default_rng(args.seed)

    top_curve, rnd_curve, bot_curve = [], [], []
    rows = []
    for k in range(1, k_max + 1):
        top_idx = order_desc[:k]
        bot_idx = order_asc[:k]

        X_top = _mask_columns(X_test, top_idx, args.masking, rng)
        X_bot = _mask_columns(X_test, bot_idx, args.masking, rng)

        mse_top = _mse(y_test, _predict(model.network, X_top))
        mse_bot = _mse(y_test, _predict(model.network, X_bot))
        d_top = mse_top - base_mse
        d_bot = mse_bot - base_mse

        d_rnd_trials = []
        for _ in range(args.random_trials):
            rnd_idx = rng.choice(X_test.shape[1], size=k, replace=False)
            X_r = _mask_columns(X_test, rnd_idx, args.masking, rng)
            mse_r = _mse(y_test, _predict(model.network, X_r))
            d_rnd_trials.append(mse_r - base_mse)
        d_rnd = float(np.mean(d_rnd_trials))

        top_curve.append(float(d_top))
        rnd_curve.append(float(d_rnd))
        bot_curve.append(float(d_bot))
        rows.append(
            {
                "k": k,
                "delta_mse_top": float(d_top),
                "delta_mse_random": float(d_rnd),
                "delta_mse_bottom": float(d_bot),
            }
        )

    auc_top = _auc_from_curve(top_curve)
    auc_rnd = _auc_from_curve(rnd_curve)
    auc_bot = _auc_from_curve(bot_curve)
    auc_gap = auc_top - auc_bot
    top_random_ratio = float(auc_top / (auc_rnd + 1e-12))

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = config["output"]["results_dir"]
    out_md = args.output_md or os.path.join(results_dir, f"faithfulness_top_random_bottom_{now}.md")
    out_json = args.output_json or os.path.join(results_dir, f"faithfulness_top_random_bottom_{now}.json")

    report = {
        "summary_path": args.summary,
        "model_state_path": state_path,
        "config_path": config_path,
        "masking": args.masking,
        "k_max": int(k_max),
        "random_trials": int(args.random_trials),
        "base_mse": float(base_mse),
        "auc_top": float(auc_top),
        "auc_random": float(auc_rnd),
        "auc_bottom": float(auc_bot),
        "auc_gap_top_minus_bottom": float(auc_gap),
        "top_random_ratio": float(top_random_ratio),
        "curve": rows,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# Faithfulness Report (Top/Random/Bottom)",
        "",
        f"- Summary: `{args.summary}`",
        f"- Config: `{config_path}`",
        f"- Model: `{state_path}`",
        f"- Masking: `{args.masking}`",
        f"- k_max: `{k_max}`",
        f"- random_trials: `{args.random_trials}`",
        "",
        f"- base MSE: `{base_mse:.6f}`",
        f"- AUC top: `{auc_top:.6f}`",
        f"- AUC random: `{auc_rnd:.6f}`",
        f"- AUC bottom: `{auc_bot:.6f}`",
        f"- AUC gap (top-bottom): `{auc_gap:.6f}`",
        f"- Top/random ratio: `{top_random_ratio:.6f}`",
        "",
        "| k | ΔMSE top | ΔMSE random | ΔMSE bottom |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['k']} | {r['delta_mse_top']:.6f} | {r['delta_mse_random']:.6f} | {r['delta_mse_bottom']:.6f} |"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_md}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
