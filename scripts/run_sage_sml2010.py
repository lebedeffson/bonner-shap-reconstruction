#!/usr/bin/env python3
"""SAGE baseline on SML2010 for ANFIS checkpoints (seed-level + aggregate)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sage
import torch
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
    p.add_argument("--multiseed", required=True)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--bg-size", type=int, default=512)
    p.add_argument("--eval-size", type=int, default=512)
    p.add_argument("--permutations", type=int, default=512)
    p.add_argument("--out", default="results/sage_sml2010_3seed_20260504.json")
    return p.parse_args()


def _load_importance(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    col = "importance" if "importance" in df.columns else df.columns[-1]
    v = np.asarray(df[col], dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.maximum(v, 0.0)
    s = float(v.sum())
    return v / s if s > 1e-12 else np.full_like(v, 1.0 / max(1, v.size))


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _split(cfg: dict, seed: int):
    ds = dict(cfg["dataset"])
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


def _predict_fn(manager: ANFISManager, state_path: str):
    holder = {"net": None}

    def _predict(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if holder["net"] is None:
            y_dim = 2
            m = manager.create_model(input_dim=X.shape[1], output_dim=y_dim, verbose=False)
            state = torch.load(state_path, map_location="cpu")
            m.network.load_state_dict(state, strict=False)
            m.network.eval()
            holder["net"] = m.network
        with torch.no_grad():
            return holder["net"](torch.tensor(X, dtype=torch.float32)).cpu().numpy()

    return _predict


def main():
    args = parse_args()
    ms_path = Path(args.multiseed).resolve()
    ms = json.loads(ms_path.read_text(encoding="utf-8"))
    repo_root = ms_path.parent.parent
    seed_set = {int(x.strip()) for x in args.seeds.split(",") if x.strip()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_names = None
    rows = []
    values_rows = []
    for run in ms["runs"]:
        seed = int(run["seed"])
        if seed not in seed_set:
            continue
        s_path = Path(run["summary_path"])
        if not s_path.is_absolute():
            s_path = (repo_root / s_path).resolve()
        summary = json.loads(s_path.read_text(encoding="utf-8"))

        cfg_path = Path(summary.get("config_path", ""))
        if not cfg_path.is_absolute():
            cfg_path = (repo_root / cfg_path).resolve()
        if not cfg_path.exists():
            cfg_path = (repo_root / ms["config"]).resolve()
        cfg = load_config(str(cfg_path))
        feature_names = list(cfg["dataset"]["feature_columns"])
        X_train, y_train, X_test, y_test = _split(cfg, seed)

        rng = np.random.default_rng(seed)
        bg_idx = rng.choice(X_train.shape[0], size=min(args.bg_size, X_train.shape[0]), replace=False)
        ev_idx = rng.choice(X_test.shape[0], size=min(args.eval_size, X_test.shape[0]), replace=False)
        X_bg = X_train[bg_idx]
        X_ev, y_ev = X_test[ev_idx], y_test[ev_idx]

        manager = ANFISManager(cfg)
        model = _predict_fn(manager, summary["model_state_path"])
        imputer = sage.MarginalImputer(model, X_bg)
        estimator = sage.PermutationEstimator(imputer, loss="mse", n_jobs=1, random_state=seed)
        expl = estimator(
            X_ev,
            y_ev,
            batch_size=256,
            detect_convergence=True,
            thresh=0.025,
            n_permutations=int(args.permutations),
            verbose=False,
            bar=False,
        )
        v = np.asarray(expl.values, dtype=float)
        v = np.maximum(v, 0.0)
        s = float(v.sum())
        sage_norm = v / s if s > 1e-12 else np.full_like(v, 1.0 / max(1, v.size))

        sdir = s_path.parent
        imp_grad = _load_importance(sdir / summary["saved_files"]["feature_importance"])
        imp_ea = _load_importance(sdir / summary["saved_files"]["shap"]["feature_importance_shap"])

        rows.append({
            "seed": seed,
            "corr_sage_vs_vanilla_grad_spearman": _rank_corr(sage_norm, imp_grad),
            "corr_sage_vs_eaar_internal_spearman": _rank_corr(sage_norm, imp_ea),
            "top1_sage": int(np.argmax(sage_norm)),
            "top3_mass_sage": float(np.sum(np.sort(sage_norm)[::-1][:3])),
        })
        for i, fn in enumerate(feature_names):
            values_rows.append({"seed": seed, "feature": fn, "sage": float(sage_norm[i])})

    df_meta = pd.DataFrame(rows)
    df_vals = pd.DataFrame(values_rows)
    agg_vals = df_vals.groupby("feature")["sage"].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=False)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "multiseed": str(ms_path),
        "seeds_used": sorted(seed_set),
        "bg_size": int(args.bg_size),
        "eval_size": int(args.eval_size),
        "permutations": int(args.permutations),
        "meta": json.loads(df_meta.to_json(orient="records")),
        "values": json.loads(df_vals.to_json(orient="records")),
        "aggregate": json.loads(agg_vals.to_json(orient="records")),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_meta = out_path.with_name(out_path.stem + "_meta.csv")
    csv_vals = out_path.with_name(out_path.stem + "_features.csv")
    md_path = out_path.with_suffix(".md")
    df_meta.to_csv(csv_meta, index=False)
    agg_vals.to_csv(csv_vals, index=False)
    md_path.write_text(
        "# SAGE SML2010\n\n"
        f"JSON: `{out_path}`\n"
        f"META CSV: `{csv_meta}`\n"
        f"FEATURE CSV: `{csv_vals}`\n\n"
        "## Seed-level meta\n\n"
        + df_meta.to_csv(index=False)
        + "\n## Aggregate feature SAGE\n\n"
        + agg_vals.to_csv(index=False),
        encoding="utf-8",
    )
    print(out_path)
    print(csv_meta)
    print(csv_vals)
    print(md_path)


if __name__ == "__main__":
    main()
