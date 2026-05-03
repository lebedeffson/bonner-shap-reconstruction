#!/usr/bin/env python3
"""Explainability report for multiseed EA runs."""

from __future__ import annotations

import argparse
import itertools
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


REAL_TEST_FRACTION = 0.2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed", required=True, help="results/multiseed_*.json")
    p.add_argument("--k-list", default="1,2,3,4")
    p.add_argument("--mask", choices=["permute", "mean", "noise"], default="permute")
    p.add_argument("--insertion-baseline", choices=["mean"], default="mean")
    p.add_argument("--seed", type=int, default=42, help="seed для mask/permutation")
    p.add_argument("--random-trials", type=int, default=20, help="число повторов для random deletion")
    p.add_argument(
        "--eval-importance",
        choices=[
            "final",
            "shap",
            "ea_raw",
            "ea-only",
            "shap-only",
            "vanilla",
            "vanilla_gradient",
            "vanilla_permutation",
        ],
        default="final",
    )
    p.add_argument("--out", default=None, help="output json path")
    return p.parse_args()


def _mean_ci95(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(vals))
    if vals.size == 1:
        return m, m
    se = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
    half = 1.96 * se
    return m - half, m + half


def _read_importance(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if "importance" in df.columns:
        vals = df["importance"].to_numpy(dtype=float)
    else:
        vals = df.iloc[:, -1].to_numpy(dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.maximum(vals, 0.0)
    s = float(vals.sum())
    if s <= 1e-12:
        return np.full_like(vals, 1.0 / max(1, vals.size), dtype=float)
    return vals / s


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _gini(p: np.ndarray) -> float:
    x = np.sort(np.asarray(p, dtype=float))
    n = x.size
    if n == 0:
        return 0.0
    s = float(np.sum(x))
    if s <= 1e-12:
        return 0.0
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * x)) / (n * s) - (n + 1.0) / n)


def _n_eff(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    denom = float(np.sum(p * p))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 / denom)


def _mass_at_k(p: np.ndarray, k: int) -> float:
    if p.size == 0:
        return 0.0
    k = int(max(1, min(k, p.size)))
    idx = np.argsort(p)[::-1][:k]
    return float(np.sum(p[idx]))


def _rankdata(a: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(a, dtype=float)).rank(method="average").to_numpy(dtype=float)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _pairwise_stats(vectors: list[np.ndarray], topk: int = 3) -> dict:
    if len(vectors) < 2:
        return {
            "pearson_mean": float("nan"),
            "spearman_mean": float("nan"),
            "topk_overlap_mean": float("nan"),
        }
    ps, ss, os = [], [], []
    for a, b in itertools.combinations(vectors, 2):
        ps.append(_corr(a, b))
        ss.append(_corr(_rankdata(a), _rankdata(b)))
        ka = set(np.argsort(a)[::-1][:topk].tolist())
        kb = set(np.argsort(b)[::-1][:topk].tolist())
        os.append(float(len(ka & kb) / max(1, topk)))
    return {
        "pearson_mean": float(np.mean(ps)),
        "spearman_mean": float(np.mean(ss)),
        "topk_overlap_mean": float(np.mean(os)),
    }


def _auc_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


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


def _make_insertion_base(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        m = np.mean(X, axis=0, keepdims=True)
        return np.repeat(m, X.shape[0], axis=0)
    raise ValueError(f"Unsupported insertion baseline mode: {mode}")


def _predict(manager: ANFISManager, state_path: str, X: np.ndarray, y_dim: int) -> np.ndarray:
    model = manager.create_model(input_dim=X.shape[1], output_dim=y_dim, verbose=False)
    state = torch.load(state_path, map_location="cpu")
    model.network.load_state_dict(state, strict=False)
    model.network.eval()
    with torch.no_grad():
        return model.network(torch.tensor(X, dtype=torch.float32)).cpu().numpy()


def _permutation_importance(
    manager: ANFISManager,
    state_path: str,
    X: np.ndarray,
    y: np.ndarray,
    y_dim: int,
    mask_mode: str,
    seed: int,
) -> np.ndarray:
    pred0 = _predict(manager, state_path, X, y_dim)
    mse0 = float(mean_squared_error(y, pred0))
    vals = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        rng = np.random.default_rng(seed + 9973 * (j + 1))
        Xm = _apply_mask(X, [j], mask_mode, rng)
        predm = _predict(manager, state_path, Xm, y_dim)
        vals[j] = max(0.0, float(mean_squared_error(y, predm) - mse0))
    s = float(vals.sum())
    if s <= 1e-12:
        return np.full(X.shape[1], 1.0 / max(1, X.shape[1]), dtype=float)
    return vals / s


def _get_test_split(base_cfg: dict, seed: int, cache: dict[int, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    if seed in cache:
        return cache[seed]
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
        n_test = max(1, int(round(n * REAL_TEST_FRACTION)))
        n_test = min(n_test, n - 2)
        n_temp = n - n_test
        n_val = max(1, int(round(n_temp * 0.25)))
        n_shap = n_temp - n_val
        X_test, y_test = X_real[n_temp:], y_real[n_temp:]
        _ = (X_real[:n_shap], y_real[:n_shap], X_real[n_shap:n_temp], y_real[n_shap:n_temp])
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_real, y_real, test_size=REAL_TEST_FRACTION, random_state=seed
        )
        _ = train_test_split(X_temp, y_temp, test_size=0.25, random_state=seed)
    cache[seed] = (np.nan_to_num(X_test), np.nan_to_num(y_test))
    return cache[seed]


def _resolve_summary(results_dir: Path, tag: str) -> Path:
    found = sorted(results_dir.glob(f"training_summary_*_{tag}.json"))
    if not found:
        raise FileNotFoundError(f"summary not found for tag={tag} in {results_dir}")
    return found[-1]


def main():
    args = parse_args()
    ms_path = Path(args.multiseed).resolve()
    ms = json.loads(ms_path.read_text(encoding="utf-8"))
    base_cfg = load_config(ms["config"])
    results_dir = Path(base_cfg["output"]["results_dir"]).resolve()
    ks = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    ks = sorted(set(k for k in ks if k > 0))
    if not ks:
        ks = [1, 2, 3]

    test_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    per_run = []
    vecs_shap, vecs_vanilla = [], []
    rng_seed = int(args.seed)
    skipped_no_model = 0

    for run in ms["runs"]:
        seed = int(run["seed"])
        tag = run["tag"]
        summary_path = _resolve_summary(results_dir, tag)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved = summary.get("saved_files", {})
        shap_file = (saved.get("shap") or {}).get("feature_importance_shap") or (summary.get("shap_files") or {}).get("feature_importance_shap")
        vanilla_file = saved.get("feature_importance")
        if not shap_file or not vanilla_file:
            continue
        p_shap = _read_importance(results_dir / shap_file)
        p_van = _read_importance(results_dir / vanilla_file)
        vecs_shap.append(p_shap)
        vecs_vanilla.append(p_van)

        eval_mode = args.eval_importance
        if eval_mode in {"shap", "ea_raw", "ea-only", "shap-only"}:
            p_eval = p_shap
            kind = "ea_raw"
        elif eval_mode in {"vanilla", "vanilla_gradient"}:
            p_eval = p_van
            kind = "vanilla_gradient"
        elif eval_mode == "vanilla_permutation":
            p_eval = None
            kind = "vanilla_permutation"
        else:
            use_shap = summary.get("metrics_source") == "shap"
            p_eval = p_shap if use_shap else p_van
            kind = "ea_raw" if use_shap else "vanilla_gradient"

        row = {
            "seed": seed,
            "tag": tag,
            "summary": str(summary_path),
            "metrics_source": summary.get("metrics_source"),
            "importance_kind_eval": kind,
            "r2_final": float(run.get("r2_final", np.nan)),
            "r2_vanilla": float(run.get("r2_vanilla", np.nan)),
            "r2_shap_raw": float(run.get("r2_shap_raw", np.nan)),
            "entropy_shap": _entropy(p_shap),
            "gini_shap": _gini(p_shap),
            "n_eff_shap": _n_eff(p_shap),
            "mass3_shap": _mass_at_k(p_shap, 3),
            "entropy_vanilla": _entropy(p_van),
            "gini_vanilla": _gini(p_van),
            "n_eff_vanilla": _n_eff(p_van),
            "mass3_vanilla": _mass_at_k(p_van, 3),
        }

        state_path = summary.get("model_state_path")
        if state_path and Path(state_path).exists():
            X_test, y_test = _get_test_split(base_cfg, seed, test_cache)
            manager = ANFISManager(base_cfg)
            y_dim = y_test.shape[1]
            if p_eval is None and kind == "vanilla_permutation":
                p_eval = _permutation_importance(
                    manager=manager,
                    state_path=state_path,
                    X=X_test,
                    y=y_test,
                    y_dim=y_dim,
                    mask_mode=args.mask,
                    seed=rng_seed + seed,
                )
            pred0 = _predict(manager, state_path, X_test, y_dim)
            mse0 = float(mean_squared_error(y_test, pred0))
            row["base_mse"] = mse0

            order = np.argsort(p_eval)[::-1]
            max_k = max(ks)
            max_k = min(max_k, max(1, p_eval.size // 2))
            k_used = [k for k in ks if k <= max_k]
            del_top, del_bottom, del_random, ins_top = [], [], [], []
            del_top_rel, del_bottom_rel, del_random_rel, ins_top_rel = [], [], [], []

            for k in k_used:
                top = order[:k].tolist()
                bottom = order[-k:].tolist()
                rng = np.random.default_rng(rng_seed + seed + 1000 * k)
                random_trials = max(1, int(args.random_trials))

                pred_top = _predict(manager, state_path, _apply_mask(X_test, top, args.mask, rng), y_dim)
                pred_bottom = _predict(manager, state_path, _apply_mask(X_test, bottom, args.mask, rng), y_dim)
                d_random_trials = []
                for t in range(random_trials):
                    rng_t = np.random.default_rng(rng_seed + seed + 1000 * k + 100000 * (t + 1))
                    random_cols = rng_t.choice(np.arange(p_eval.size), size=k, replace=False).tolist()
                    pred_random = _predict(manager, state_path, _apply_mask(X_test, random_cols, args.mask, rng_t), y_dim)
                    d_random_trials.append(float(mean_squared_error(y_test, pred_random) - mse0))
                d_top = float(mean_squared_error(y_test, pred_top) - mse0)
                d_bottom = float(mean_squared_error(y_test, pred_bottom) - mse0)
                d_random = float(np.mean(d_random_trials))
                d_top_rel = float(d_top / (mse0 + 1e-12))
                d_bottom_rel = float(d_bottom / (mse0 + 1e-12))
                d_random_rel = float(d_random / (mse0 + 1e-12))
                del_top.append(d_top)
                del_bottom.append(d_bottom)
                del_random.append(d_random)
                del_top_rel.append(d_top_rel)
                del_bottom_rel.append(d_bottom_rel)
                del_random_rel.append(d_random_rel)
                row[f"del_top_k{k}"] = d_top
                row[f"del_bottom_k{k}"] = d_bottom
                row[f"del_random_k{k}"] = d_random
                row[f"del_top_rel_k{k}"] = d_top_rel
                row[f"del_bottom_rel_k{k}"] = d_bottom_rel
                row[f"del_random_rel_k{k}"] = d_random_rel
                row[f"del_random_trials_k{k}"] = random_trials
                row[f"del_ratio_k{k}"] = float(d_top / (d_bottom + 1e-12))
                row[f"del_top_random_ratio_k{k}"] = float(d_top / (d_random + 1e-12))

                X_base = _make_insertion_base(X_test, args.insertion_baseline)
                pred_base = _predict(manager, state_path, X_base, y_dim)
                mse_base = float(mean_squared_error(y_test, pred_base))
                X_ins = X_base.copy()
                X_ins[:, top] = X_test[:, top]
                pred_ins = _predict(manager, state_path, X_ins, y_dim)
                gain = float(mse_base - mean_squared_error(y_test, pred_ins))
                gain_rel = float(gain / (mse_base + 1e-12))
                ins_top.append(gain)
                ins_top_rel.append(gain_rel)
                row[f"ins_gain_k{k}"] = gain
                row[f"ins_gain_rel_k{k}"] = gain_rel

            if k_used:
                X_base_ref = _make_insertion_base(X_test, args.insertion_baseline)
                pred_base_ref = _predict(manager, state_path, X_base_ref, y_dim)
                mse_base_ref = float(mean_squared_error(y_test, pred_base_ref))
                row["insertion_baseline_mse"] = mse_base_ref
                x = np.asarray(k_used, dtype=float)
                row["auc_deletion_top"] = _auc_trapezoid(np.asarray(del_top, dtype=float), x)
                row["auc_deletion_bottom"] = _auc_trapezoid(np.asarray(del_bottom, dtype=float), x)
                row["auc_deletion_random"] = _auc_trapezoid(np.asarray(del_random, dtype=float), x)
                row["auc_deletion_top_rel"] = _auc_trapezoid(np.asarray(del_top_rel, dtype=float), x)
                row["auc_deletion_bottom_rel"] = _auc_trapezoid(np.asarray(del_bottom_rel, dtype=float), x)
                row["auc_deletion_random_rel"] = _auc_trapezoid(np.asarray(del_random_rel, dtype=float), x)
                row["auc_insertion_gain_top"] = _auc_trapezoid(np.asarray(ins_top, dtype=float), x)
                row["auc_insertion_gain_top_rel"] = _auc_trapezoid(np.asarray(ins_top_rel, dtype=float), x)
                row["auc_deletion_gap_top_bottom"] = float(row["auc_deletion_top"] - row["auc_deletion_bottom"])
                row["auc_deletion_gap_top_random"] = float(row["auc_deletion_top"] - row["auc_deletion_random"])
                row["auc_del_top_bottom_ratio"] = float(row["auc_deletion_top"] / (row["auc_deletion_bottom"] + 1e-12))
                row["auc_del_top_random_ratio"] = float(row["auc_deletion_top"] / (row["auc_deletion_random"] + 1e-12))
        else:
            skipped_no_model += 1

        per_run.append(row)

    df = pd.DataFrame(per_run).sort_values("seed").reset_index(drop=True)
    metrics = {
        "n_runs": int(len(df)),
        "delta_r2_mean": float(df["r2_shap_raw"].sub(df["r2_vanilla"]).mean()) if len(df) else float("nan"),
        "delta_r2_std": float(df["r2_shap_raw"].sub(df["r2_vanilla"]).std(ddof=0)) if len(df) else float("nan"),
        "wins": int((df["r2_shap_raw"] > df["r2_vanilla"]).sum()) if len(df) else 0,
        "losses": int((df["r2_shap_raw"] < df["r2_vanilla"]).sum()) if len(df) else 0,
        "deletion_runs_with_model": int(len(df) - skipped_no_model),
        "deletion_runs_skipped_no_model": int(skipped_no_model),
        "shap_stability": _pairwise_stats(vecs_shap, topk=3),
        "vanilla_stability": _pairwise_stats(vecs_vanilla, topk=3),
    }
    for col in [
        "entropy_shap", "gini_shap", "n_eff_shap", "mass3_shap",
        "entropy_vanilla", "gini_vanilla", "n_eff_vanilla", "mass3_vanilla",
        "auc_deletion_top", "auc_deletion_bottom", "auc_deletion_random",
        "auc_deletion_top_rel", "auc_deletion_bottom_rel", "auc_deletion_random_rel",
        "auc_insertion_gain_top", "auc_insertion_gain_top_rel",
        "auc_deletion_gap_top_bottom", "auc_deletion_gap_top_random",
        "auc_del_top_bottom_ratio", "auc_del_top_random_ratio",
    ]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            metrics[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
            metrics[f"{col}_std"] = float(vals.std(ddof=0)) if len(vals) else float("nan")
            lo, hi = _mean_ci95(vals.to_numpy(dtype=float))
            metrics[f"{col}_ci95_low"] = float(lo)
            metrics[f"{col}_ci95_high"] = float(hi)

    out = {
        "multiseed": str(ms_path),
        "config": ms.get("config"),
        "results_dir": str(results_dir),
        "eval_importance": args.eval_importance,
        "mask": args.mask,
        "insertion_baseline": args.insertion_baseline,
        "k_list": ks,
        "random_trials": int(args.random_trials),
        "aggregate": metrics,
        "runs": df.to_dict(orient="records"),
    }

    out_path = Path(args.out) if args.out else Path("results") / (
        f"explainability_{ms_path.stem}_{args.mask}_{args.eval_importance}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"saved json: {out_path}")
    print(f"saved csv: {csv_path}")
    print(json.dumps(out["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
