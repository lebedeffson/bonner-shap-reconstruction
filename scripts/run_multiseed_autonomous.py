#!/usr/bin/env python3
"""Запуск мультисид-оценки автономной SHAP-регуляризации."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Базовый YAML конфиг")
    p.add_argument("--seeds", default="42,43,44,45,46", help="Список seed через запятую")
    p.add_argument("--tag-prefix", default="autonomous_ms", help="Префикс tag")
    p.add_argument("--python", default="/home/lebedeffson/Code/venv_cuda/bin/python")
    p.add_argument("--inprocess", action="store_true", help="Запуск seed в одном процессе (быстрее)")
    p.add_argument("--fast", action="store_true", help="Быстрый режим для черновых мультисидов")
    p.add_argument("--pso-epochs", type=int, default=25, help="PSO эпох в fast-режиме")
    p.add_argument("--pso-pop", type=int, default=30, help="PSO pop_size в fast-режиме")
    p.add_argument("--shap-epochs", type=int, default=15, help="SHAP эпох в fast-режиме")
    p.add_argument("--fast-save-model", action="store_true", help="В fast-режиме сохранять model_state (для deletion tests)")
    return p.parse_args()


def _apply_fast_overrides(cfg: dict, args) -> None:
    cfg.setdefault("model", {}).setdefault("optim_params", {})
    cfg["model"]["optim_params"]["epoch"] = int(args.pso_epochs)
    cfg["model"]["optim_params"]["pop_size"] = int(args.pso_pop)

    cfg.setdefault("shap_reg", {})
    cfg["shap_reg"]["epochs"] = int(args.shap_epochs)
    cfg["shap_reg"]["early_stopping_patience"] = min(
        int(cfg["shap_reg"].get("early_stopping_patience", 20)),
        max(5, int(args.shap_epochs) // 2),
    )

    cfg.setdefault("output", {})
    cfg["output"]["save_plots"] = False
    cfg["output"]["save_samples"] = False
    cfg["output"]["save_model"] = bool(args.fast_save_model)


def run_one(base_cfg: dict, seed: int, tag_prefix: str, python_bin: str, args) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("_run_meta", {})
    cfg["_run_meta"]["fast_mode"] = bool(args.fast)
    cfg["_run_meta"]["inprocess_mode"] = bool(args.inprocess)
    cfg["_run_meta"]["tag_prefix"] = tag_prefix
    cfg.setdefault("dataset", {})["random_state"] = int(seed)
    cfg.setdefault("model", {})["seed"] = int(seed)
    cfg.setdefault("shap_reg", {})["seed"] = int(seed)
    if args.fast:
        _apply_fast_overrides(cfg, args)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
        cfg_path = tmp.name

    tag = f"{tag_prefix}_s{seed}"
    if args.inprocess:
        from train import train_and_save
        train_args = SimpleNamespace(
            config=cfg_path,
            train_limit=None,
            train_fraction=None,
            tag=tag,
        )
        train_and_save(train_args)
    else:
        cmd = [python_bin, "train.py", "--config", cfg_path, "--tag", tag]
        subprocess.run(cmd, check=True)

    results_dir = Path(cfg["output"]["results_dir"])
    summaries = sorted(results_dir.glob(f"training_summary_*_{tag}.json"))
    if not summaries:
        raise RuntimeError(f"summary not found for tag={tag}")
    s = json.loads(summaries[-1].read_text(encoding="utf-8"))
    metrics_source = s.get("metrics_source")
    unstable_flag = bool(s.get("unstable_prediction_flag", False))
    if metrics_source == "shap":
        model_mode = "ea_raw"
    elif metrics_source == "vanilla_fallback":
        model_mode = "final_policy"
    elif metrics_source == "unstable_run":
        model_mode = "unstable_run"
    else:
        model_mode = "unknown"
    return {
        "seed": seed,
        "tag": tag,
        "summary_path": str(summaries[-1]),
        "metrics_source": metrics_source,
        "model_mode": model_mode,
        "fallback_used": bool(metrics_source == "vanilla_fallback"),
        "unstable_prediction_flag": unstable_flag,
        "r2_final": float(s["metrics"]["r2"]),
        "r2_vanilla": float((s.get("vanilla_metrics") or {}).get("r2", float("nan"))),
        "r2_shap_raw": float((s.get("shap_metrics") or {}).get("r2", float("nan"))),
        "r2_ea_raw": float((s.get("shap_metrics") or {}).get("r2", float("nan"))),
    }


def _is_finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _aggregate_rows(rows):
    deltas = [
        float(r["r2_shap_raw"]) - float(r["r2_vanilla"])
        for r in rows
        if _is_finite(r.get("r2_vanilla")) and _is_finite(r.get("r2_shap_raw"))
    ]
    out = {"n_runs": len(rows)}
    if deltas:
        out["delta_r2_mean"] = sum(deltas) / len(deltas)
        out["delta_r2_min"] = min(deltas)
        out["delta_r2_max"] = max(deltas)
    out["wins"] = int(sum(1 for d in deltas if d > 0))
    out["losses"] = int(sum(1 for d in deltas if d < 0))
    out["fallback_rate"] = float(sum(1 for r in rows if r.get("fallback_used")) / max(len(rows), 1))
    return out


def main():
    args = parse_args()
    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows = []
    for seed in seeds:
        rows.append(run_one(base_cfg, seed, args.tag_prefix, args.python, args))

    unstable_seeds = [int(r["seed"]) for r in rows if r.get("unstable_prediction_flag")]
    stable_rows = [r for r in rows if not r.get("unstable_prediction_flag")]

    out = {
        "config": str(Path(args.config).resolve()),
        "seeds": seeds,
        "runs": rows,
        "n_unstable_runs": len(unstable_seeds),
        "unstable_seeds": unstable_seeds,
        "aggregate_all_runs": _aggregate_rows(rows),
        "aggregate_stable_only": _aggregate_rows(stable_rows),
    }
    # Backward-compat keys
    deltas = [
        float(r["r2_shap_raw"]) - float(r["r2_vanilla"])
        for r in rows
        if _is_finite(r.get("r2_vanilla")) and _is_finite(r.get("r2_shap_raw"))
    ]
    if deltas:
        out["delta_r2_mean"] = sum(deltas) / len(deltas)
        out["delta_r2_min"] = min(deltas)
        out["delta_r2_max"] = max(deltas)
    out_path = Path("results") / f"multiseed_{Path(args.config).stem}_{args.tag_prefix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
