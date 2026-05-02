#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser(description="Run classic tabular baselines on dataset from config")
    p.add_argument("--config", required=True, help="Path to YAML config with dataset block")
    p.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    p.add_argument("--tag", default="")
    return p.parse_args()


def load_cfg(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metric_pack(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def make_models(seed: int):
    return {
        "rf": RandomForestRegressor(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
        "et": ExtraTreesRegressor(
            n_estimators=600,
            random_state=seed,
            n_jobs=-1,
            min_samples_leaf=1,
        ),
        "hgb": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                random_state=seed,
                max_iter=600,
                learning_rate=0.05,
                max_depth=None,
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


def main():
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
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

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = data_path.stem
    tag = f"_{args.tag}" if args.tag else ""
    out_json = out_dir / f"baselines_{dataset_name}{tag}.json"
    out_csv = out_dir / f"baselines_{dataset_name}{tag}.csv"

    runs = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed + seed
        )
        models = make_models(seed)
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = metric_pack(y_test, pred)
            runs.append({"seed": seed, "model": name, **metrics})

    df_runs = pd.DataFrame(runs)
    summary = (
        df_runs.groupby("model")[["r2", "rmse", "mae", "mse"]]
        .agg(["mean", "std", "median"])
        .reset_index()
    )

    payload = {
        "config": str(cfg_path),
        "data_path": str(data_path),
        "seeds": seeds,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets": int(y.shape[1]),
        "runs": runs,
        "summary": json.loads(summary.to_json(orient="records")),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    df_runs.to_csv(out_csv, index=False)
    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()

