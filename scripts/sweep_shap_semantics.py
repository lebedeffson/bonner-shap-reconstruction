#!/usr/bin/env python3
"""
Sweep SHAP semantic choices:
- baseline mode: feature_mean / median / zero
- utility scalarization: mean_output / sum_output / l2_output

Runs train.py for each combo and aggregates metrics from training_summary.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Sweep SHAP semantic settings")
    p.add_argument("--base-config", required=True)
    p.add_argument("--out-dir", default="configs/sweeps/shap_semantics")
    p.add_argument("--tag-prefix", default="semantics")
    p.add_argument("--train-cmd", default="python train.py")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-md", default="")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--summary-glob-dir", default="results")
    p.add_argument("--baseline-modes", default="feature_mean,median,zero")
    p.add_argument("--value-functions", default="mean_output,sum_output,l2_output")
    return p.parse_args()


def write_config(base_cfg_path, out_path, baseline_mode, value_fn):
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sr = cfg.setdefault("shap_reg", {})
    sr["shap_baseline_mode"] = baseline_mode
    sr["shap_value_function"] = value_fn
    sr.setdefault("shap_baseline_clip_nonnegative", True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def newest_summary(results_dir: str, tag: str):
    cands = sorted(Path(results_dir).glob(f"training_summary_*_{tag}.json"))
    return str(cands[-1]) if cands else ""


def run_train(train_cmd: str, cfg_path: str, tag: str):
    cmd = f"{train_cmd} --config {cfg_path} --tag {tag}"
    return subprocess.run(cmd, shell=True, check=False)


def main():
    args = parse_args()
    modes = [m.strip() for m in args.baseline_modes.split(",") if m.strip()]
    vals = [v.strip() for v in args.value_functions.split(",") if v.strip()]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    records = []

    for bm in modes:
        for vf in vals:
            tag = f"{args.tag_prefix}_{bm}_{vf}"
            cfg_path = str(Path(args.out_dir) / f"config_{tag}.yaml")
            write_config(args.base_config, cfg_path, bm, vf)
            if not args.skip_train:
                rc = run_train(args.train_cmd, cfg_path, tag)
                if rc.returncode != 0:
                    records.append({
                        "tag": tag,
                        "baseline_mode": bm,
                        "value_function": vf,
                        "status": "train_failed",
                    })
                    continue

            summary_path = newest_summary(args.summary_glob_dir, tag)
            if not summary_path:
                records.append({
                    "tag": tag,
                    "baseline_mode": bm,
                    "value_function": vf,
                    "status": "summary_missing",
                })
                continue

            with open(summary_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            m = s.get("metrics", {})
            gs = ((s.get("diagnostics") or {}).get("gap_selection") or {}).get("best") or {}
            records.append({
                "tag": tag,
                "baseline_mode": bm,
                "value_function": vf,
                "status": "ok",
                "summary_path": summary_path,
                "r2_weighted": m.get("r2_weighted"),
                "r2_mean": m.get("r2_mean"),
                "sel_auc_gap": gs.get("auc_gap"),
                "sel_top_random": gs.get("top_random_ratio"),
            })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.output_json or f"results/shap_semantics_sweep_{ts}.json"
    out_md = args.output_md or f"results/shap_semantics_sweep_{ts}.md"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, ensure_ascii=False, indent=2)

    lines = [
        "# SHAP Semantics Sweep",
        "",
        f"- Base config: `{os.path.abspath(args.base_config)}`",
        "",
        "| Baseline mode | Value function | Status | R2_w | R2_mean | Sel AUC gap | Sel top/random | Summary |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {r.get('baseline_mode')} | {r.get('value_function')} | {r.get('status')} | "
            f"{(r.get('r2_weighted') if r.get('r2_weighted') is not None else float('nan')):.6f} | "
            f"{(r.get('r2_mean') if r.get('r2_mean') is not None else float('nan')):.6f} | "
            f"{(r.get('sel_auc_gap') if r.get('sel_auc_gap') is not None else float('nan')):.6f} | "
            f"{(r.get('sel_top_random') if r.get('sel_top_random') is not None else float('nan')):.6f} | "
            f"{r.get('summary_path','')} |"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()
