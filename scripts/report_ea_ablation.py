#!/usr/bin/env python3
"""Build ablation summary table from ablation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="results/ablation/...json")
    p.add_argument("--out-prefix", default="")
    return p.parse_args()


def _safe_get(dct, *keys, default=np.nan):
    cur = dct
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _ms_stats(ms: dict):
    runs = ms.get("runs", [])
    if not runs:
        return {}
    r2_final = np.array([float(r.get("r2_final", np.nan)) for r in runs], dtype=float)
    r2_v = np.array([float(r.get("r2_vanilla", np.nan)) for r in runs], dtype=float)
    r2_s = np.array([float(r.get("r2_shap_raw", np.nan)) for r in runs], dtype=float)
    src = [str(r.get("metrics_source")) for r in runs]
    return {
        "n_seeds": int(len(runs)),
        "r2_final_mean": float(np.nanmean(r2_final)),
        "r2_final_std": float(np.nanstd(r2_final, ddof=1)) if len(r2_final) > 1 else 0.0,
        "r2_vanilla_mean": float(np.nanmean(r2_v)),
        "r2_shap_raw_mean": float(np.nanmean(r2_s)),
        "delta_r2_mean": float(np.nanmean(r2_s - r2_v)),
        "wins": int(np.sum((r2_s - r2_v) > 0)),
        "losses": int(np.sum((r2_s - r2_v) < 0)),
        "fallback_rate": float(np.mean([s == "vanilla_fallback" for s in src])),
        "shap_rate": float(np.mean([s == "shap" for s in src])),
    }


def _exp_stats(exp: dict):
    agg = exp.get("aggregate", {})
    return {
        "auc_top": float(_safe_get(agg, "auc_deletion_top_mean")),
        "auc_random": float(_safe_get(agg, "auc_deletion_random_mean")),
        "auc_bottom": float(_safe_get(agg, "auc_deletion_bottom_mean")),
        "auc_gap": float(_safe_get(agg, "auc_deletion_gap_top_bottom_mean", default=np.nan)),
        "auc_top_random_ratio": float(_safe_get(agg, "auc_ratio_top_random_mean", default=np.nan)),
        "n_eff_shap": float(_safe_get(agg, "n_eff_shap_mean", default=np.nan)),
        "mass3_shap": float(_safe_get(agg, "mass3_shap_mean", default=np.nan)),
        "entropy_shap": float(_safe_get(agg, "entropy_shap_mean", default=np.nan)),
        "gini_shap": float(_safe_get(agg, "gini_shap_mean", default=np.nan)),
    }


def main():
    args = parse_args()
    man_path = Path(args.manifest).resolve()
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    rows = []
    for name, info in manifest.get("variants", {}).items():
        row = {"variant": name}
        ms_path = Path(info["multiseed"])
        ms = json.loads(ms_path.read_text(encoding="utf-8"))
        row.update(_ms_stats(ms))

        exp_path = info.get("explainability")
        if exp_path:
            exp = json.loads(Path(exp_path).read_text(encoding="utf-8"))
            row.update(_exp_stats(exp))
        else:
            row.update(
                {
                    "auc_top": np.nan,
                    "auc_random": np.nan,
                    "auc_bottom": np.nan,
                    "auc_gap": np.nan,
                    "auc_top_random_ratio": np.nan,
                    "n_eff_shap": np.nan,
                    "mass3_shap": np.nan,
                    "entropy_shap": np.nan,
                    "gini_shap": np.nan,
                }
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No variants in manifest")

    out_prefix = args.out_prefix or f"ablation_summary_{man_path.stem}"
    out_json = Path("results") / f"{out_prefix}.json"
    out_csv = Path("results") / f"{out_prefix}.csv"
    out_md = Path("results") / f"{out_prefix}.md"

    out_json.write_text(
        json.dumps(
            {
                "manifest": str(man_path),
                "rows": df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    df.to_csv(out_csv, index=False)

    cols = [
        "variant",
        "delta_r2_mean",
        "wins",
        "losses",
        "fallback_rate",
        "auc_gap",
        "auc_top_random_ratio",
        "n_eff_shap",
        "mass3_shap",
    ]
    use_cols = [c for c in cols if c in df.columns]
    try:
        table_md = df[use_cols].to_markdown(index=False)
    except Exception:
        # Fallback when optional `tabulate` dependency is unavailable.
        table_md = df[use_cols].to_csv(index=False)
    md = ["# EA Ablation Summary", "", f"Manifest: `{man_path}`", "", table_md]
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
