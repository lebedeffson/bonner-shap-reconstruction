#!/usr/bin/env python3
"""Run ResMLP vanilla vs EAAR multi-seed and compute deletion faithfulness metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.resmlp_regressor import ResMLPRegressor


def _trapz(y, x):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default="42,43,44,45,46")
    p.add_argument("--python", default="/home/lebedeffson/Code/venv_cuda/bin/python")
    p.add_argument("--tag-prefix", default="resmlp_eaar_energy5")
    p.add_argument("--k-list", default="1,2,3,4")
    p.add_argument("--mask", choices=["permute", "mean", "noise"], default="permute")
    p.add_argument("--random-trials", type=int, default=20)
    p.add_argument("--out", default=None)
    return p.parse_args()


def _mean(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _apply_mask(X, cols, mode, rng):
    X2 = np.asarray(X, dtype=float).copy()
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


def _load_importance(path: Path):
    df = pd.read_csv(path)
    col = "importance" if "importance" in df.columns else df.columns[-1]
    v = np.asarray(df[col], dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.maximum(v, 0.0)
    s = float(np.sum(v))
    if s <= 1e-12:
        return np.full_like(v, 1.0 / max(1, v.size))
    return v / s


def _predict(model_path, X_train, X_test, input_dim, output_dim, hidden_dim, n_blocks, dropout):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    model = ResMLPRegressor(input_dim, output_dim, hidden_dim=hidden_dim, n_blocks=n_blocks, dropout=dropout)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test_sc, dtype=torch.float32)).cpu().numpy()
    return np.asarray(pred, dtype=float)


def _deletion_metrics(model_path, X_train, X_test, y_test, imp, input_dim, output_dim, hidden_dim, n_blocks, dropout, k_list, mask, seed, random_trials):
    def predict_fn(Xm):
        return _predict(model_path, X_train, Xm, input_dim, output_dim, hidden_dim, n_blocks, dropout)

    p0 = predict_fn(X_test)
    mse0 = float(mean_squared_error(y_test, p0))
    n_features = imp.size
    ks = sorted(set(k for k in k_list if 0 < k <= n_features))
    x = []
    topv, botv, randv = [], [], []
    for k in ks:
        top = np.argsort(imp)[::-1][:k].tolist()
        bot = np.argsort(imp)[:k].tolist()
        rng = np.random.default_rng(seed + 111 * (k + 1))
        pt = predict_fn(_apply_mask(X_test, top, mask, rng))
        pb = predict_fn(_apply_mask(X_test, bot, mask, rng))
        dt = float(mean_squared_error(y_test, pt) - mse0)
        db = float(mean_squared_error(y_test, pb) - mse0)
        dr_trials = []
        for t in range(max(1, int(random_trials))):
            rng_t = np.random.default_rng(seed + 1009 * (k + 1) + 7919 * (t + 1))
            cols = rng_t.choice(n_features, size=k, replace=False).tolist()
            pr = predict_fn(_apply_mask(X_test, cols, mask, rng_t))
            dr_trials.append(float(mean_squared_error(y_test, pr) - mse0))
        dr = float(np.mean(dr_trials))
        x.append(float(k) / float(max(ks)))
        topv.append(dt)
        botv.append(db)
        randv.append(dr)
    xa = np.asarray(x, dtype=float)
    topa = np.asarray(topv, dtype=float)
    bota = np.asarray(botv, dtype=float)
    randa = np.asarray(randv, dtype=float)
    auc_top = _trapz(topa, xa) if topa.size > 1 else float(topa[0])
    auc_bottom = _trapz(bota, xa) if bota.size > 1 else float(bota[0])
    auc_random = _trapz(randa, xa) if randa.size > 1 else float(randa[0])
    return {
        "auc_top": auc_top,
        "auc_random": auc_random,
        "auc_bottom": auc_bottom,
        "auc_gap": float(auc_top - auc_bottom),
        "top_random_ratio": float(auc_top / max(auc_random, 1e-12)),
    }


def _write_cfg(base_cfg, seed, enabled, tmp_path):
    cfg = deepcopy(base_cfg)
    cfg.setdefault("dataset", {})["random_state"] = int(seed)
    cfg.setdefault("model", {})["seed"] = int(seed)
    cfg.setdefault("shap_reg", {})["enabled"] = bool(enabled)
    tmp_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _run_train(python_exe, cfg_path, tag, mode):
    cmd = [python_exe, "scripts/train_resmlp_ea.py", "--config", str(cfg_path), "--tag", tag, "--mode", mode]
    res = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
    summary = None
    for ln in lines:
        if "training_summary_" in ln and ln.endswith(".json"):
            summary = Path(ln)
            break
    if summary is None:
        raise RuntimeError(f"Cannot parse summary path from output:\n{res.stdout}\n{res.stderr}")
    return summary.resolve()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    k_list = [int(s.strip()) for s in args.k_list.split(",") if s.strip()]
    out_path = Path(args.out) if args.out else Path("results") / f"resmlp_eaar_multiseed_{cfg_path.stem}_{args.tag_prefix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / "_tmp_resmlp_multiseed"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ds = base_cfg["dataset"]
    mcfg = base_cfg.get("model", {})
    dropout = float(mcfg.get("dropout", 0.1))
    hidden_dim = int(mcfg.get("hidden_dim", 128))
    n_blocks = int(mcfg.get("n_blocks", 2))
    df = pd.read_csv(ds["train_data"])
    X_all = df[ds["feature_columns"]].to_numpy(dtype=float)
    y_all = df[ds["target_columns"]].to_numpy(dtype=float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y_all = np.nan_to_num(y_all, nan=0.0, posinf=0.0, neginf=0.0)

    rows = []
    for seed in seeds:
        cfg_v = tmp_dir / f"{cfg_path.stem}_s{seed}_vanilla.yaml"
        cfg_e = tmp_dir / f"{cfg_path.stem}_s{seed}_ea.yaml"
        _write_cfg(base_cfg, seed, False, cfg_v)
        _write_cfg(base_cfg, seed, True, cfg_e)
        sum_v = _run_train(args.python, cfg_v, f"{args.tag_prefix}_s{seed}_vanilla", "vanilla")
        sum_e = _run_train(args.python, cfg_e, f"{args.tag_prefix}_s{seed}_ea", "ea")
        sv = json.loads(sum_v.read_text(encoding="utf-8"))
        se = json.loads(sum_e.read_text(encoding="utf-8"))

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=float(ds.get("test_size", 0.2)), random_state=seed
        )
        in_dim = X_train.shape[1]
        out_dim = y_train.shape[1]

        out_dir_v = Path(sv["model_state_path"]).resolve().parent
        out_dir_e = Path(se["model_state_path"]).resolve().parent
        imp_v = _load_importance(out_dir_v / sv["saved_files"]["feature_importance"])
        imp_e = _load_importance(out_dir_e / se["saved_files"]["feature_importance"])

        dm_v = _deletion_metrics(
            sv["model_state_path"], X_train, X_test, y_test, imp_v,
            in_dim, out_dim, hidden_dim, n_blocks, dropout, k_list, args.mask, seed, args.random_trials
        )
        dm_e = _deletion_metrics(
            se["model_state_path"], X_train, X_test, y_test, imp_e,
            in_dim, out_dim, hidden_dim, n_blocks, dropout, k_list, args.mask, seed, args.random_trials
        )

        rows.append({
            "seed": seed,
            "vanilla_r2": float(sv["metrics"]["r2"]),
            "ea_r2": float(se["metrics"]["r2"]),
            "delta_r2": float(se["metrics"]["r2"] - sv["metrics"]["r2"]),
            "vanilla_auc_gap": float(dm_v["auc_gap"]),
            "ea_auc_gap": float(dm_e["auc_gap"]),
            "vanilla_top_random": float(dm_v["top_random_ratio"]),
            "ea_top_random": float(dm_e["top_random_ratio"]),
            "vanilla_summary": str(sum_v),
            "ea_summary": str(sum_e),
        })

    out = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(cfg_path),
        "seeds": seeds,
        "k_list": k_list,
        "mask": args.mask,
        "random_trials": int(args.random_trials),
        "aggregate": {
            "n_runs": len(rows),
            "delta_r2_mean": _mean([r["delta_r2"] for r in rows]),
            "delta_r2_std": _std([r["delta_r2"] for r in rows]),
            "wins": int(sum(1 for r in rows if r["delta_r2"] > 0)),
            "losses": int(sum(1 for r in rows if r["delta_r2"] < 0)),
            "vanilla_auc_gap_mean": _mean([r["vanilla_auc_gap"] for r in rows]),
            "ea_auc_gap_mean": _mean([r["ea_auc_gap"] for r in rows]),
            "vanilla_top_random_mean": _mean([r["vanilla_top_random"] for r in rows]),
            "ea_top_random_mean": _mean([r["ea_top_random"] for r in rows]),
        },
        "runs": rows,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
