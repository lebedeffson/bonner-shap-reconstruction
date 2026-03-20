#!/usr/bin/env python3
"""Быстрый stage2-only тюнинг V2-кандидатов поверх общей vanilla-инициализации."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.models.anfis_manager import ANFISManager
from src.models.shap_trainer_improved import ShapAwareANFISTrainerImproved
from src.utils.config_loader import load_config
from src.utils.data_loader import (
    load_training_dataset,
    prepare_features_targets,
    split_data,
)
from train import _split_real_data_for_shap
from tune_v2_against_v1 import better_than_baseline, metric_deltas


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_SUMMARY = REPO_ROOT / "results" / "training_summary_20260320_055350_v2_official_det_20260320.json"
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "config_integrated_shap_v2.yaml"


def candidate_definitions() -> list[dict]:
    return [
        {
            "name": "stage2_ref_eq_target0390",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0050,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.60,
            "band_weights": [1 / 3, 1 / 3, 1 / 3],
            "target_shap_ratio": 0.39,
            "gamma_end": 0.10,
        },
        {
            "name": "stage2_eq_target0392",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0050,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.60,
            "band_weights": [1 / 3, 1 / 3, 1 / 3],
            "target_shap_ratio": 0.392,
            "gamma_end": 0.10,
        },
        {
            "name": "stage2_eq_target0394",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0050,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.60,
            "band_weights": [1 / 3, 1 / 3, 1 / 3],
            "target_shap_ratio": 0.394,
            "gamma_end": 0.10,
        },
        {
            "name": "stage2_eq_target0392_gamma0099",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0050,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.60,
            "band_weights": [1 / 3, 1 / 3, 1 / 3],
            "target_shap_ratio": 0.392,
            "gamma_end": 0.099,
        },
        {
            "name": "stage2_eq_target0394_gamma0099",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0050,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.60,
            "band_weights": [1 / 3, 1 / 3, 1 / 3],
            "target_shap_ratio": 0.394,
            "gamma_end": 0.099,
        },
        {
            "name": "stage2_lowband_target0392_gamma0099",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0050,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.60,
            "band_weights": [0.338, 0.333, 0.329],
            "target_shap_ratio": 0.392,
            "gamma_end": 0.099,
        },
        {
            "name": "stage2_midband_target0392_gamma0099",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0050,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.60,
            "band_weights": [0.332, 0.336, 0.332],
            "target_shap_ratio": 0.392,
            "gamma_end": 0.099,
        },
        {
            "name": "stage2_eq_target0392_nonneg0045_warm65",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0045,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.65,
            "band_weights": [1 / 3, 1 / 3, 1 / 3],
            "target_shap_ratio": 0.392,
            "gamma_end": 0.099,
        },
        {
            "name": "stage2_lowband_target0392_nonneg0045_warm65",
            "tikhonov_lambda": 0.0010,
            "tikhonov_lambda_start": 0.0005,
            "tikhonov_warmup_epochs": 0.35,
            "nonneg_lambda": 0.0045,
            "nonneg_lambda_start": 0.0,
            "nonneg_warmup_epochs": 0.65,
            "band_weights": [0.338, 0.333, 0.329],
            "target_shap_ratio": 0.392,
            "gamma_end": 0.099,
        },
    ]


def build_candidate_config(base_config: dict, candidate: dict) -> dict:
    cfg = deepcopy(base_config)
    shap = cfg["shap_reg"]
    shap["target_shap_ratio"] = float(candidate["target_shap_ratio"])
    shap["gamma"] = float(candidate["gamma_end"])
    shap["gamma_end"] = float(candidate["gamma_end"])
    shap["scalarization"]["mode"] = "band_weighted"
    shap["scalarization"]["band_weights"] = [float(x) for x in candidate["band_weights"]]
    shap["tikhonov"]["enabled"] = True
    shap["tikhonov"]["energy_aware"] = True
    shap["tikhonov"]["lambda"] = float(candidate["tikhonov_lambda"])
    shap["tikhonov"]["lambda_start"] = float(candidate["tikhonov_lambda_start"])
    shap["tikhonov"]["lambda_end"] = float(candidate["tikhonov_lambda"])
    shap["tikhonov"]["warmup_epochs"] = float(candidate["tikhonov_warmup_epochs"])
    shap["nonnegativity"]["enabled"] = True
    shap["nonnegativity"]["mode"] = str(candidate.get("nonneg_mode", "mass_ratio"))
    shap["nonnegativity"]["lambda"] = float(candidate["nonneg_lambda"])
    shap["nonnegativity"]["lambda_start"] = float(candidate["nonneg_lambda_start"])
    shap["nonnegativity"]["lambda_end"] = float(candidate["nonneg_lambda"])
    shap["nonnegativity"]["warmup_epochs"] = float(candidate["nonneg_warmup_epochs"])
    if "nonneg_power" in candidate:
        shap["nonnegativity"]["power"] = int(candidate["nonneg_power"])
    if "nonneg_tolerance" in candidate:
        shap["nonnegativity"]["tolerance"] = float(candidate["nonneg_tolerance"])
    if "soft_count_weight" in candidate:
        shap["nonnegativity"]["soft_count_weight"] = float(candidate["soft_count_weight"])
    if "soft_count_temperature" in candidate:
        shap["nonnegativity"]["soft_count_temperature"] = float(candidate["soft_count_temperature"])
    return cfg


def to_numpy_clean(array_like):
    arr = np.asarray(array_like, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_datasets(config: dict) -> dict:
    dataset_config = config["dataset"]
    normalize_sum = dataset_config.get("normalize_sum", False)

    from src.utils.data_loader import load_validation_data

    real_data_path = dataset_config.get("validation_data")
    X_real, y_real, SUM_real = load_validation_data(
        real_data_path,
        normalize_sum=normalize_sum,
        dataset_config=dataset_config,
    )

    data = load_training_dataset(dataset_config)
    X, y, _ = prepare_features_targets(
        data, normalize_sum=normalize_sum, dataset_config=dataset_config
    )

    X_train, _, y_train, _ = split_data(
        X,
        y,
        test_size=dataset_config.get("test_size", 0.25),
        random_state=dataset_config.get("random_state", 42),
    )

    (
        X_real_shap,
        X_real_val,
        X_real_test,
        y_real_shap,
        y_real_val,
        y_real_test,
        _,
    ) = _split_real_data_for_shap(
        X_real,
        y_real,
        SUM_real,
        normalize_sum=normalize_sum,
        random_state=dataset_config.get("random_state", 42),
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_real_shap": X_real_shap,
        "y_real_shap": y_real_shap,
        "X_real_val": X_real_val,
        "y_real_val": y_real_val,
        "X_real_test": X_real_test,
        "y_real_test": y_real_test,
    }


def train_or_load_vanilla(base_config: dict, datasets: dict, cache_path: Path | None):
    manager = ANFISManager(base_config)
    if hasattr(datasets["X_train"], "columns"):
        manager.set_feature_names(datasets["X_train"].columns)

    input_dim = to_numpy_clean(datasets["X_train"]).shape[1]
    output_dim = to_numpy_clean(datasets["y_train"]).shape[1]

    if cache_path is not None and cache_path.exists():
        model = manager.create_model(verbose=False, input_dim=input_dim, output_dim=output_dim)
        state = torch.load(cache_path, map_location="cpu")
        model.network.load_state_dict(state)
        return manager, model, {"cached": True}

    vanilla_results = manager.train_vanilla_model(
        to_numpy_clean(datasets["X_train"]),
        to_numpy_clean(datasets["y_train"]),
        to_numpy_clean(datasets["X_real_val"]),
        to_numpy_clean(datasets["y_real_val"]),
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vanilla_results["model"].network.state_dict(), cache_path)
    return manager, vanilla_results["model"], {"cached": False, "metrics": vanilla_results["metrics"]}


def clone_model_from_state(manager: ANFISManager, vanilla_model, X_train, y_train):
    try:
        return deepcopy(vanilla_model)
    except Exception:
        input_dim = to_numpy_clean(X_train).shape[1]
        output_dim = to_numpy_clean(y_train).shape[1]
        model = manager.create_model(verbose=False, input_dim=input_dim, output_dim=output_dim)
        state = {
            key: value.detach().cpu().clone()
            for key, value in vanilla_model.network.state_dict().items()
        }
        model.network.load_state_dict(state, strict=True)
        return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Быстрый stage2-only тюнинг V2-кандидатов")
    parser.add_argument("--baseline-summary", default=str(DEFAULT_BASELINE_SUMMARY))
    parser.add_argument("--base-config", default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / f"v2_stage2_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--vanilla-cache", default="")
    parser.add_argument("--stop-on-success", action="store_true")
    args = parser.parse_args()

    baseline_summary = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    baseline_metrics = baseline_summary["metrics"]
    base_config = load_config(args.base_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = prepare_datasets(base_config)
    manager, vanilla_model, vanilla_info = train_or_load_vanilla(
        base_config, datasets, Path(args.vanilla_cache) if args.vanilla_cache else None
    )

    rows = []
    winner = None

    for candidate in candidate_definitions():
        cfg = build_candidate_config(base_config, candidate)
        model = clone_model_from_state(manager, vanilla_model, datasets["X_train"], datasets["y_train"])
        trainer = ShapAwareANFISTrainerImproved(
            model,
            cfg,
            gamma=cfg["shap_reg"].get("gamma", 0.0),
            verbose=False,
        )
        history = trainer.fit(
            to_numpy_clean(datasets["X_real_shap"]),
            to_numpy_clean(datasets["y_real_shap"]),
            epochs=cfg["shap_reg"].get("epochs", 25),
            batch_size=cfg["shap_reg"].get("batch_size", 32),
            lr=cfg["shap_reg"].get("lr", 0.003),
        )
        y_pred = trainer.predict(to_numpy_clean(datasets["X_real_test"]))
        y_true = to_numpy_clean(datasets["y_real_test"])
        metrics = manager._calculate_metrics(y_true, y_pred)
        deltas = metric_deltas(metrics, baseline_metrics)
        success = better_than_baseline(metrics, baseline_metrics)

        pred = np.asarray(y_pred, dtype=float)
        negative_fraction = float(np.mean(pred < 0.0))
        row = {
            "name": candidate["name"],
            **candidate,
            **metrics,
            **deltas,
            "success": success,
            "negative_fraction": negative_fraction,
            "training_time_shap": float(trainer.training_time),
            "final_tikhonov_lambda": float(history["tikhonov_lambda"][-1]),
            "final_nonnegativity_lambda": float(history["nonnegativity_lambda"][-1]),
        }
        rows.append(row)

        candidate_dir = output_dir / candidate["name"]
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "stage2_candidate_summary.json").write_text(
            json.dumps(
                {
                    "candidate": candidate,
                    "metrics": metrics,
                    "deltas": deltas,
                    "negative_fraction": negative_fraction,
                    "training_time_shap": trainer.training_time,
                    "vanilla_cached": vanilla_info.get("cached", False),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        if success and winner is None:
            winner = row
            if args.stop_on_success:
                break

    csv_path = output_dir / "stage2_tuning_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = output_dir / "stage2_tuning_summary.md"
    lines = [
        "# Stage2 Tuning Summary",
        "",
        f"- Baseline summary: `{args.baseline_summary}`",
        f"- Vanilla cached: `{vanilla_info.get('cached', False)}`",
        f"- Output CSV: `{csv_path}`",
        "",
        "| name | mse | rmse | mae | r2_weighted | r2_mean | success | negative_fraction |",
        "| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['mse']:.8f} | {row['rmse']:.8f} | {row['mae']:.8f} | "
            f"{row['r2_weighted']:.8f} | {row['r2_mean']:.8f} | {str(row['success'])} | {row['negative_fraction']:.6f} |"
        )
    if winner:
        lines.extend(["", "## Winner", "", f"- {winner['name']}"])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved stage2 tuning summary: {csv_path}")
    if winner:
        print(f"Winner: {winner['name']}")
        return 0
    print("No candidate beat baseline on all metrics.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
