#!/usr/bin/env python3
"""Абляционный анализ вклада SHAP и Tikhonov-регуляризации."""

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from src.utils.config_loader import load_config
from train import train_and_save


def parse_args():
    parser = argparse.ArgumentParser(description="Запуск абляционного анализа для SHAP/Tikhonov")
    parser.add_argument("--config", default="configs/config_integrated_shap.yaml", help="Базовый YAML конфиг")
    parser.add_argument("--output-dir", default="results/ablation_study", help="Папка для абляционных прогонов")
    parser.add_argument("--tag-prefix", default="ablation", help="Префикс тэга для всех запусков")
    parser.add_argument("--train-limit", type=int, default=128, help="Ограничение числа train-образцов для ускорения")
    parser.add_argument("--pso-epochs", type=int, default=20, help="Число эпох PSO для абляции")
    parser.add_argument("--pso-pop-size", type=int, default=20, help="Размер популяции PSO для абляции")
    parser.add_argument("--shap-epochs", type=int, default=40, help="Число эпох SHAP/Tikhonov fine-tune")
    parser.add_argument("--num-rules", type=int, default=20, help="Число правил ANFIS для абляции")
    parser.add_argument("--include-component-ablation", action="store_true", help="Добавить абляции по SHAP-компонентам")
    parser.add_argument(
        "--component-mode",
        choices=["default", "stronger", "both"],
        default="stronger",
        help="На какой силе SHAP запускать компонентные абляции",
    )
    return parser.parse_args()


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_summary(summary_path):
    return json.loads(Path(summary_path).read_text(encoding="utf-8"))


def _merge_updates(*chunks):
    result = {}
    for chunk in chunks:
        _deep_update(result, deepcopy(chunk))
    return result


def _load_shap_history(summary_path):
    summary = _load_summary(summary_path)
    history_rel = summary.get("shap_files", {}).get("history")
    if not history_rel:
        return summary, {}
    history_path = Path(summary_path).parent / history_rel
    if not history_path.exists():
        return summary, {}
    return summary, json.loads(history_path.read_text(encoding="utf-8"))


def _compute_smoothness(summary, summary_path):
    predictions_rel = summary.get("saved_files", {}).get("predictions")
    if not predictions_rel:
        return {}
    predictions_path = Path(summary_path).parent / predictions_rel
    if not predictions_path.exists():
        return {}
    predictions = np.load(predictions_path)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] < 3:
        return {}

    d1 = predictions[:, 1:] - predictions[:, :-1]
    d2 = predictions[:, 2:] - 2.0 * predictions[:, 1:-1] + predictions[:, :-2]
    return {
        "pred_d1_mean_sq": float(np.mean(d1 ** 2)),
        "pred_d2_mean_sq": float(np.mean(d2 ** 2)),
    }


def _compute_shap_distribution_metrics(summary, summary_path):
    shap_csv_rel = summary.get("shap_files", {}).get("feature_importance_shap")
    if not shap_csv_rel:
        return {}
    shap_csv_path = Path(summary_path).parent / shap_csv_rel
    if not shap_csv_path.exists():
        return {}

    with shap_csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}

    names = [row.get("", "") for row in rows]
    values = np.asarray([float(row["importance"]) for row in rows], dtype=float)
    values = np.abs(values)
    if values.sum() > 0:
        values = values / values.sum()

    sorted_values = np.sort(values)
    n = len(sorted_values)
    indices = np.arange(1, n + 1)
    total = np.sum(sorted_values)
    if total <= 1e-12 or n == 0:
        gini = 0.0
    else:
        gini = (2.0 * np.sum(indices * sorted_values)) / (n * total) - (n + 1) / n
    entropy = -np.sum(values[values > 1e-12] * np.log(values[values > 1e-12] + 1e-12))
    entropy /= np.log(n) + 1e-12

    top_indices = np.argsort(values)[::-1][:3]
    return {
        "shap_gini": float(max(0.0, min(1.0, gini))),
        "shap_entropy": float(entropy),
        "shap_top3": ",".join(names[idx] for idx in top_indices),
        "shap_top3_mass": float(np.sum(values[top_indices])),
    }


def _stronger_shap_updates():
    return {
        "shap_reg": {
            "use_improved_shap": True,
            "gamma": 0.099,
            "gamma_start": 0.02,
            "gamma_end": 0.099,
            "target_shap_ratio": 0.3895,
            "min_convergence_slowdown": 0.25,
            "tikhonov": {"enabled": True, "lambda": 0.001},
        }
    }


def _extract_regularization_stats(summary, history):
    diagnostics = summary.get("diagnostics", {}).get("regularization", {})
    if diagnostics:
        return diagnostics

    result = {}
    for key in [
        "shap_contribution",
        "tikhonov_contribution",
        "regularization_share",
        "shap_scale_factor",
        "shap_loss_normalized",
    ]:
        values = np.asarray(history.get(key, []), dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            result[key] = {
                "mean": float(np.mean(values)),
                "last": float(values[-1]),
                "max": float(np.max(values)),
            }
    shap_mean = result.get("shap_contribution", {}).get("mean", 0.0)
    tikh_mean = result.get("tikhonov_contribution", {}).get("mean", 0.0)
    if shap_mean > tikh_mean:
        result["dominant_regularizer"] = "shap"
    elif tikh_mean > shap_mean:
        result["dominant_regularizer"] = "tikhonov"
    else:
        result["dominant_regularizer"] = "balanced"
    return result


def _variant_definitions(component_mode="stronger"):
    stronger_updates = _stronger_shap_updates()
    base_variants = [
        (
            "vanilla_stage1",
            {
                "shap_reg": {
                    "epochs": 0,
                    "gamma": 0.0,
                    "gamma_start": 0.0,
                    "gamma_end": 0.0,
                    "use_adaptive_gamma": False,
                    "target_shap_ratio": 0.0,
                    "use_improved_shap": False,
                    "tikhonov": {"enabled": False, "lambda": 0.0},
                }
            },
        ),
        (
            "tikhonov_only",
            {
                "shap_reg": {
                    "gamma": 0.0,
                    "gamma_start": 0.0,
                    "gamma_end": 0.0,
                    "use_adaptive_gamma": False,
                    "target_shap_ratio": 0.0,
                    "use_improved_shap": False,
                    "tikhonov": {"enabled": True},
                }
            },
        ),
        (
            "shap_only",
            {
                "shap_reg": {
                    "use_improved_shap": True,
                    "tikhonov": {"enabled": False, "lambda": 0.0},
                }
            },
        ),
        ("shap_tikhonov", {"shap_reg": {"use_improved_shap": True, "tikhonov": {"enabled": True}}}),
        ("shap_tikhonov_stronger", stronger_updates),
    ]

    component_variants = []
    default_component_variants = [
        ("no_consistency", {"shap_reg": {"active_components": ["sparsity", "faithfulness", "stability"]}}),
        ("no_sparsity", {"shap_reg": {"active_components": ["consistency", "faithfulness", "stability"]}}),
        ("no_faithfulness", {"shap_reg": {"active_components": ["consistency", "sparsity", "stability"]}}),
        ("no_stability", {"shap_reg": {"active_components": ["consistency", "sparsity", "faithfulness"]}}),
    ]
    stronger_component_variants = [
        (
            "strong_no_consistency",
            _merge_updates(stronger_updates, {"shap_reg": {"active_components": ["sparsity", "faithfulness", "stability"]}}),
        ),
        (
            "strong_no_sparsity",
            _merge_updates(stronger_updates, {"shap_reg": {"active_components": ["consistency", "faithfulness", "stability"]}}),
        ),
        (
            "strong_no_faithfulness",
            _merge_updates(stronger_updates, {"shap_reg": {"active_components": ["consistency", "sparsity", "stability"]}}),
        ),
        (
            "strong_no_stability",
            _merge_updates(stronger_updates, {"shap_reg": {"active_components": ["consistency", "sparsity", "faithfulness"]}}),
        ),
    ]
    if component_mode in {"default", "both"}:
        component_variants.extend(default_component_variants)
    if component_mode in {"stronger", "both"}:
        component_variants.extend(stronger_component_variants)
    return base_variants, component_variants


def _prepare_variant_config(base_config, args, variant_name, updates):
    cfg = deepcopy(base_config)
    _deep_update(cfg, updates)

    cfg["dataset"]["train_limit"] = args.train_limit
    cfg["model"]["num_rules"] = min(cfg["model"].get("num_rules", args.num_rules), args.num_rules)
    cfg["model"]["n_workers"] = 1
    cfg["model"]["optim_params"]["epoch"] = args.pso_epochs
    cfg["model"]["optim_params"]["pop_size"] = args.pso_pop_size
    cfg["model"]["optim_params"]["verbose"] = False

    cfg["shap_reg"]["epochs"] = updates.get("shap_reg", {}).get("epochs", args.shap_epochs)
    cfg["shap_reg"]["batch_size"] = min(int(cfg["shap_reg"].get("batch_size", 32)), 16)
    cfg["shap_reg"]["use_gpu"] = False

    cfg["output"]["results_dir"] = str(Path(args.output_dir) / variant_name)
    cfg["output"]["save_plots"] = False
    cfg["output"]["save_samples"] = False
    return cfg


def _run_variant(base_config, args, variant_name, updates):
    variant_dir = Path(args.output_dir) / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    cfg = _prepare_variant_config(base_config, args, variant_name, updates)

    config_path = variant_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    run_args = SimpleNamespace(
        config=str(config_path),
        train_limit=args.train_limit,
        train_fraction=None,
        tag=f"{args.tag_prefix}_{variant_name}",
    )
    _, summary_path = train_and_save(run_args)
    summary, history = _load_shap_history(summary_path)

    row = {
        "variant": variant_name,
        "summary_path": str(summary_path),
        "mse": summary["metrics"]["mse"],
        "rmse": summary["metrics"]["rmse"],
        "mae": summary["metrics"]["mae"],
        "r2_weighted": summary["metrics"]["r2_weighted"],
        "r2_mean": summary["metrics"]["r2_mean"],
        "training_time_total": summary.get("training_time_total", 0.0),
        "training_time_shap": summary.get("training_time_shap", 0.0),
    }

    reg = _extract_regularization_stats(summary, history)
    for key, stats in reg.items():
        if isinstance(stats, dict):
            row[f"{key}_mean"] = stats.get("mean")
            row[f"{key}_last"] = stats.get("last")
        else:
            row[key] = stats
    row["dominant_regularizer"] = reg.get("dominant_regularizer", summary.get("diagnostics", {}).get("regularization", {}).get("dominant_regularizer"))

    row.update(_compute_smoothness(summary, summary_path))
    row.update(_compute_shap_distribution_metrics(summary, summary_path))
    return row


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_config(args.config)
    base_variants, component_variants = _variant_definitions(component_mode=args.component_mode)
    variants = list(base_variants)
    if args.include_component_ablation:
        variants.extend(component_variants)

    rows = []
    for variant_name, updates in variants:
        print("=" * 100)
        print(f"▶️  Запуск варианта: {variant_name}")
        rows.append(_run_variant(base_config, args, variant_name, updates))

    if not rows:
        return

    columns = sorted({key for row in rows for key in row.keys()})
    csv_path = output_dir / "ablation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "ablation_summary.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"✅ Абляционный анализ завершён")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
