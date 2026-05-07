#!/usr/bin/env python3
"""
Neural unfolding baselines on the same Bonner split.

Models:
1) MLP baseline
2) 1D-CNN baseline
3) MC-Dropout MLP (bayesian-like proxy)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
    p = argparse.ArgumentParser(description="Neural unfolding baselines benchmark")
    p.add_argument("--config", default="configs/config_integrated_shap.yaml")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--mc-samples", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-md", default="")
    return p.parse_args()


def _split_real(X, y, random_state):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=REAL_TEST_FRACTION, random_state=random_state
    )
    X_train, _X_val, y_train, _y_val = train_test_split(
        X_temp, y_temp, test_size=REAL_VALIDATION_FRACTION_OF_TEMP, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def _split_train_val(X, y, random_state):
    return train_test_split(X, y, test_size=0.2, random_state=random_state)


def _metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    dot = np.sum(y_true * y_pred, axis=1)
    n1 = np.linalg.norm(y_true, axis=1)
    n2 = np.linalg.norm(y_pred, axis=1)
    cos = dot / np.maximum(n1 * n2, 1e-12)
    rel_l1 = np.sum(np.abs(y_true - y_pred), axis=1) / np.maximum(np.sum(np.abs(y_true), axis=1), 1e-12)
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2_weighted": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "r2_mean": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "cosine_mean": float(np.mean(cos)),
        "shape_rel_l1_mean": float(np.mean(rel_l1)),
    }


class MLPNet(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        return self.net(x)


class CNN1DNet(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * d_in, 128),
            nn.ReLU(),
            nn.Linear(128, d_out),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B,1,d]
        x = self.features(x)
        return self.head(x)


def _train_one(model, X_train, y_train, X_val, y_val, *, epochs, batch_size, lr, patience, device):
    model.to(device)
    x_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_va = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_va = torch.tensor(y_val, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    bad = 0

    for _epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_tr.size(0), device=device)
        for i in range(0, x_tr.size(0), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_va), y_va).item())
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model, X, device):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        yp = model(xt).detach().cpu().numpy()
    return np.nan_to_num(yp, nan=0.0, posinf=0.0, neginf=0.0)


def _predict_mc_dropout(model, X, device, mc_samples):
    # Keep dropout enabled in eval-style pass by forcing train mode.
    model.train()
    preds = []
    xt = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        for _ in range(mc_samples):
            yp = model(xt).detach().cpu().numpy()
            preds.append(yp)
    mean_pred = np.mean(np.asarray(preds, dtype=float), axis=0)
    return np.nan_to_num(mean_pred, nan=0.0, posinf=0.0, neginf=0.0)


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
    d_in, d_out = X.shape[1], y.shape[1]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    records = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        X_tr0, X_te, y_tr0, y_te = _split_real(X, y, random_state=seed)
        X_tr, X_va, y_tr, y_va = _split_train_val(X_tr0, y_tr0, random_state=seed)

        # MLP
        mlp = MLPNet(d_in=d_in, d_out=d_out, dropout=0.0)
        mlp = _train_one(
            mlp, X_tr, y_tr, X_va, y_va,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience, device=device
        )
        pred = _predict(mlp, X_te, device=device)
        records.append({"seed": seed, "model": "MLP", **_metrics(y_te, pred)})

        # 1D-CNN
        cnn = CNN1DNet(d_in=d_in, d_out=d_out)
        cnn = _train_one(
            cnn, X_tr, y_tr, X_va, y_va,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience, device=device
        )
        pred = _predict(cnn, X_te, device=device)
        records.append({"seed": seed, "model": "CNN1D", **_metrics(y_te, pred)})

        # Bayesian-like proxy: MC Dropout MLP
        mc = MLPNet(d_in=d_in, d_out=d_out, dropout=0.15)
        mc = _train_one(
            mc, X_tr, y_tr, X_va, y_va,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience, device=device
        )
        pred = _predict_mc_dropout(mc, X_te, device=device, mc_samples=args.mc_samples)
        records.append({"seed": seed, "model": "MC_Dropout_MLP", **_metrics(y_te, pred)})

    by_model = {}
    for r in records:
        by_model.setdefault(r["model"], []).append(r)
    agg = {}
    keys = ["r2_weighted", "r2_mean", "rmse", "mae", "cosine_mean", "shape_rel_l1_mean"]
    for model, vals in by_model.items():
        agg[model] = {}
        for k in keys:
            arr = [v[k] for v in vals]
            agg[model][f"{k}_mean"] = float(np.mean(arr))
            agg[model][f"{k}_std"] = float(np.std(arr))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = cfg.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = args.output_json or os.path.join(results_dir, f"neural_unfolding_benchmark_{ts}.json")
    out_md = args.output_md or os.path.join(results_dir, f"neural_unfolding_benchmark_{ts}.md")

    payload = {
        "config_path": os.path.abspath(args.config),
        "seeds": seeds,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "patience": int(args.patience),
        "mc_samples": int(args.mc_samples),
        "device": device,
        "n_samples": int(X.shape[0]),
        "n_features": int(d_in),
        "n_bins": int(d_out),
        "records": records,
        "aggregate": agg,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# Neural Unfolding Benchmark",
        "",
        f"- Config: `{os.path.abspath(args.config)}`",
        f"- Seeds: `{seeds}`",
        f"- Device: `{device}`",
        f"- Samples: `{X.shape[0]}`; Features: `{d_in}`; Bins: `{d_out}`",
        "",
        "| Model | R2_w mean±std | R2_mean mean±std | Cosine mean±std | Rel-L1 mean±std | RMSE mean±std | MAE mean±std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in sorted(agg.keys()):
        a = agg[model]
        lines.append(
            f"| {model} | "
            f"{a['r2_weighted_mean']:.6f} ± {a['r2_weighted_std']:.6f} | "
            f"{a['r2_mean_mean']:.6f} ± {a['r2_mean_std']:.6f} | "
            f"{a['cosine_mean_mean']:.6f} ± {a['cosine_mean_std']:.6f} | "
            f"{a['shape_rel_l1_mean_mean']:.6f} ± {a['shape_rel_l1_mean_std']:.6f} | "
            f"{a['rmse_mean']:.6f} ± {a['rmse_std']:.6f} | "
            f"{a['mae_mean']:.6f} ± {a['mae_std']:.6f} |"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
