#!/usr/bin/env python3
"""Мультисид-эксперимент для Energy Efficiency + агрегирование метрик."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiseed ANFIS experiments for Energy Efficiency")
    parser.add_argument(
        "--python",
        default="/home/lebedeffson/Code/venv/bin/python",
        help="Путь к python окружения",
    )
    parser.add_argument(
        "--seeds",
        default="41,42,43,44,45",
        help="Список seed через запятую",
    )
    parser.add_argument(
        "--vanilla-config",
        default="configs/config_energy_vanilla_real_only.yaml",
        help="Базовый конфиг vanilla",
    )
    parser.add_argument(
        "--shap-config",
        default="configs/config_energy_integrated_shap.yaml",
        help="Базовый конфиг SHAP",
    )
    parser.add_argument(
        "--results-dir",
        default="results/energy_efficiency",
        help="Папка результатов обучения",
    )
    parser.add_argument(
        "--out-dir",
        default="results/energy_efficiency/multiseed",
        help="Папка сводных результатов",
    )
    return parser.parse_args()


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _run(cmd: list[str], cwd: Path):
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _latest_summary_for_tag(results_dir: Path, tag: str) -> Path:
    candidates = sorted(results_dir.glob(f"training_summary_*_{tag}.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Не найден training_summary для tag={tag}")
    return candidates[-1]


def _read_importance(results_dir: Path, summary: dict, model_kind: str) -> np.ndarray:
    saved = summary.get("saved_files", {})
    if model_kind == "vanilla":
        rel = saved.get("feature_importance")
    else:
        rel = (
            saved.get("shap", {}).get("feature_importance_shap")
            or summary.get("shap_files", {}).get("feature_importance_shap")
        )
    if not rel:
        raise KeyError(f"Не найден файл важности для {model_kind}")
    path = results_dir / rel
    df = pd.read_csv(path, index_col=0)
    arr = np.asarray(df["importance"], dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    s = float(arr.sum())
    if s <= 1e-12:
        arr = np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=float)
    else:
        arr = arr / s
    return arr


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


def _pairwise_corr(vectors: list[np.ndarray]) -> float:
    if len(vectors) < 2:
        return float("nan")
    vals = []
    for a, b in itertools.combinations(vectors, 2):
        if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            vals.append(1.0)
        else:
            vals.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python_bin = args.python
    results_dir = (repo_root / args.results_dir).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    cfg_dir = out_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    vanilla_base = _load_yaml((repo_root / args.vanilla_config).resolve())
    shap_base = _load_yaml((repo_root / args.shap_config).resolve())

    rows = []
    model_vectors = {"vanilla": [], "shap": []}

    for seed in seeds:
        # Vanilla
        tag_v = f"energy_vanilla_s{seed}"
        cfg_v = json.loads(json.dumps(vanilla_base))
        cfg_v.setdefault("dataset", {})["random_state"] = seed
        cfg_v.setdefault("model", {})["seed"] = seed
        cfg_v_path = cfg_dir / f"config_energy_vanilla_seed_{seed}.yaml"
        _save_yaml(cfg_v_path, cfg_v)
        _run(
            [python_bin, "train_vanilla_real_only.py", "--config", str(cfg_v_path), "--tag", tag_v],
            cwd=repo_root,
        )
        summary_v_path = _latest_summary_for_tag(results_dir, tag_v)
        summary_v = json.loads(summary_v_path.read_text(encoding="utf-8"))
        imp_v = _read_importance(results_dir, summary_v, "vanilla")
        model_vectors["vanilla"].append(imp_v)
        m_v = summary_v.get("metrics", {})
        rows.append(
            {
                "model": "vanilla",
                "seed": seed,
                "summary_file": summary_v_path.name,
                "mse": float(m_v.get("mse", np.nan)),
                "rmse": float(m_v.get("rmse", np.nan)),
                "mae": float(m_v.get("mae", np.nan)),
                "r2": float(m_v.get("r2", np.nan)),
                "r2_mean": float(m_v.get("r2_mean", np.nan)),
                "importance_entropy": _entropy(imp_v),
                "importance_gini": _gini(imp_v),
            }
        )

        # SHAP
        tag_s = f"energy_shap_s{seed}"
        cfg_s = json.loads(json.dumps(shap_base))
        cfg_s.setdefault("dataset", {})["random_state"] = seed
        cfg_s.setdefault("model", {})["seed"] = seed
        cfg_s.setdefault("shap_reg", {})["seed"] = seed
        cfg_s_path = cfg_dir / f"config_energy_shap_seed_{seed}.yaml"
        _save_yaml(cfg_s_path, cfg_s)
        _run(
            [python_bin, "train.py", "--config", str(cfg_s_path), "--tag", tag_s],
            cwd=repo_root,
        )
        summary_s_path = _latest_summary_for_tag(results_dir, tag_s)
        summary_s = json.loads(summary_s_path.read_text(encoding="utf-8"))
        imp_s = _read_importance(results_dir, summary_s, "shap")
        model_vectors["shap"].append(imp_s)
        m_s = summary_s.get("metrics", {})
        rows.append(
            {
                "model": "shap",
                "seed": seed,
                "summary_file": summary_s_path.name,
                "mse": float(m_s.get("mse", np.nan)),
                "rmse": float(m_s.get("rmse", np.nan)),
                "mae": float(m_s.get("mae", np.nan)),
                "r2": float(m_s.get("r2", np.nan)),
                "r2_mean": float(m_s.get("r2_mean", np.nan)),
                "importance_entropy": _entropy(imp_s),
                "importance_gini": _gini(imp_s),
            }
        )

    detail_df = pd.DataFrame(rows).sort_values(["model", "seed"]).reset_index(drop=True)
    detail_path = out_dir / "energy_multiseed_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    summary_rows = []
    for model in ["vanilla", "shap"]:
        part = detail_df[detail_df["model"] == model]
        summary_rows.append(
            {
                "model": model,
                "n_runs": int(len(part)),
                "mse_mean": float(part["mse"].mean()),
                "mse_std": float(part["mse"].std(ddof=0)),
                "rmse_mean": float(part["rmse"].mean()),
                "rmse_std": float(part["rmse"].std(ddof=0)),
                "mae_mean": float(part["mae"].mean()),
                "mae_std": float(part["mae"].std(ddof=0)),
                "r2_mean": float(part["r2"].mean()),
                "r2_std": float(part["r2"].std(ddof=0)),
                "r2_outputs_mean": float(part["r2_mean"].mean()),
                "r2_outputs_std": float(part["r2_mean"].std(ddof=0)),
                "importance_entropy_mean": float(part["importance_entropy"].mean()),
                "importance_entropy_std": float(part["importance_entropy"].std(ddof=0)),
                "importance_gini_mean": float(part["importance_gini"].mean()),
                "importance_gini_std": float(part["importance_gini"].std(ddof=0)),
                "importance_stability_corr": _pairwise_corr(model_vectors[model]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "energy_multiseed_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    md_lines = ["# Energy Multiseed Summary", "", summary_df.to_string(index=False), "", "## Detail", "", detail_df.to_string(index=False), ""]
    (out_dir / "energy_multiseed_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("DONE")
    print(f"detail: {detail_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
