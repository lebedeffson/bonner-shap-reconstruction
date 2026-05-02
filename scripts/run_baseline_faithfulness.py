#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description="Faithfulness deletion test for tabular baselines")
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    p.add_argument("--k-list", default="1,2,3,4")
    p.add_argument("--mask", choices=["permute", "mean", "noise"], default="permute")
    p.add_argument("--random-trials", type=int, default=20)
    p.add_argument("--tag", default="")
    return p.parse_args()


def load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_models(seed: int):
    return {
        "rf": RandomForestRegressor(
            n_estimators=400, random_state=seed, n_jobs=-1, min_samples_leaf=1
        ),
        "et": ExtraTreesRegressor(
            n_estimators=600, random_state=seed, n_jobs=-1, min_samples_leaf=1
        ),
        "hgb": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                random_state=seed, max_iter=600, learning_rate=0.05, max_depth=None
            )
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=1e-3,
                        max_iter=800,
                        random_state=seed,
                        early_stopping=True,
                        n_iter_no_change=25,
                    ),
                ),
            ]
        ),
    }


def apply_mask(X: np.ndarray, cols: list[int], mode: str, rng: np.random.Generator) -> np.ndarray:
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


def auc_from_k(k_vals: np.ndarray, y_vals: np.ndarray) -> float:
    if len(k_vals) == 1:
        return float(y_vals[0])
    return float(np.trapezoid(y_vals, k_vals))


def main():
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    k_list = sorted([int(x.strip()) for x in args.k_list.split(",") if x.strip()])

    cfg_path = Path(args.config).resolve()
    cfg = load_cfg(cfg_path)
    ds = cfg["dataset"]
    data_path = Path(ds["train_data"]).resolve()
    df = pd.read_csv(data_path)

    feature_cols = ds["feature_columns"]
    target_cols = ds["target_columns"]
    test_size = float(ds.get("test_size", 0.2))
    split_seed = int(ds.get("random_state", 42))
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_cols].to_numpy(dtype=float)
    n_features = X.shape[1]

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = data_path.stem
    tag = f"_{args.tag}" if args.tag else ""
    out_json = out_dir / f"baseline_faithfulness_{dataset_name}{tag}.json"
    out_csv = out_dir / f"baseline_faithfulness_{dataset_name}{tag}.csv"

    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed + seed
        )
        models = make_models(seed)
        rng = np.random.default_rng(seed)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            pred0 = model.predict(X_test)
            base_mse = float(mean_squared_error(y_test, pred0))

            # importance: single-feature masked delta MSE
            imp = np.zeros(n_features, dtype=float)
            for j in range(n_features):
                pred_j = model.predict(apply_mask(X_test, [j], args.mask, rng))
                imp[j] = float(mean_squared_error(y_test, pred_j) - base_mse)

            order = np.argsort(imp)[::-1]

            del_top = []
            del_bottom = []
            del_random = []

            for k in k_list:
                kk = max(1, min(k, n_features // 2))
                top = order[:kk].tolist()
                bottom = order[-kk:].tolist()

                pred_top = model.predict(apply_mask(X_test, top, args.mask, rng))
                pred_bottom = model.predict(apply_mask(X_test, bottom, args.mask, rng))
                d_top = float(mean_squared_error(y_test, pred_top) - base_mse)
                d_bottom = float(mean_squared_error(y_test, pred_bottom) - base_mse)

                d_rand_trials = []
                for _ in range(args.random_trials):
                    cols = rng.choice(n_features, size=kk, replace=False).tolist()
                    pred_rand = model.predict(apply_mask(X_test, cols, args.mask, rng))
                    d_rand_trials.append(float(mean_squared_error(y_test, pred_rand) - base_mse))
                d_rand = float(np.mean(d_rand_trials))

                del_top.append(d_top)
                del_bottom.append(d_bottom)
                del_random.append(d_rand)

            k_arr = np.array(k_list, dtype=float)
            top_arr = np.array(del_top, dtype=float)
            bottom_arr = np.array(del_bottom, dtype=float)
            rand_arr = np.array(del_random, dtype=float)

            auc_top = auc_from_k(k_arr, top_arr)
            auc_bottom = auc_from_k(k_arr, bottom_arr)
            auc_rand = auc_from_k(k_arr, rand_arr)

            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "base_mse": base_mse,
                    "auc_top": auc_top,
                    "auc_random": auc_rand,
                    "auc_bottom": auc_bottom,
                    "auc_gap_top_bottom": auc_top - auc_bottom,
                    "auc_gap_top_random": auc_top - auc_rand,
                    "auc_ratio_top_random": auc_top / (auc_rand + 1e-12),
                    "auc_ratio_top_bottom": auc_top / (auc_bottom + 1e-12),
                }
            )

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(out_csv, index=False)

    agg = (
        df_rows.groupby("model")[
            [
                "auc_top",
                "auc_random",
                "auc_bottom",
                "auc_gap_top_bottom",
                "auc_gap_top_random",
                "auc_ratio_top_random",
                "auc_ratio_top_bottom",
            ]
        ]
        .agg(["mean", "std", "median"])
        .reset_index()
    )

    payload = {
        "config": str(cfg_path),
        "data_path": str(data_path),
        "seeds": seeds,
        "k_list": k_list,
        "mask": args.mask,
        "random_trials": args.random_trials,
        "runs": rows,
        "summary": json.loads(agg.to_json(orient="records")),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

