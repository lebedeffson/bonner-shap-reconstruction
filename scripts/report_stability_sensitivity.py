#!/usr/bin/env python3
"""Build all-runs vs stable-only sensitivity summary from multiseed json."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--multiseed", required=True, help="results/multiseed_*.json")
    p.add_argument("--out-prefix", default="", help="optional results/<name> prefix without extension")
    return p.parse_args()


def _is_finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def _aggregate(rows):
    deltas = [
        float(r["r2_shap_raw"]) - float(r["r2_vanilla"])
        for r in rows
        if _is_finite(r.get("r2_shap_raw")) and _is_finite(r.get("r2_vanilla"))
    ]
    out = {
        "n_runs": len(rows),
        "wins": int(sum(1 for d in deltas if d > 0)),
        "losses": int(sum(1 for d in deltas if d < 0)),
    }
    if deltas:
        out["delta_r2_mean"] = float(sum(deltas) / len(deltas))
        out["delta_r2_min"] = float(min(deltas))
        out["delta_r2_max"] = float(max(deltas))
    fallback = [bool(r.get("fallback_used", r.get("metrics_source") == "vanilla_fallback")) for r in rows]
    out["fallback_rate"] = float(sum(1 for x in fallback if x) / max(len(fallback), 1))
    return out


def main():
    args = parse_args()
    ms_path = Path(args.multiseed).resolve()
    d = json.loads(ms_path.read_text(encoding="utf-8"))
    runs = d.get("runs", [])

    if "aggregate_all_runs" in d and "aggregate_stable_only" in d:
        agg_all = d["aggregate_all_runs"]
        agg_stable = d["aggregate_stable_only"]
        unstable_seeds = d.get("unstable_seeds", [])
    else:
        unstable_seeds = [r.get("seed") for r in runs if bool(r.get("unstable_prediction_flag", False))]
        stable_runs = [r for r in runs if not bool(r.get("unstable_prediction_flag", False))]
        if not stable_runs:
            stable_runs = runs
        agg_all = _aggregate(runs)
        agg_stable = _aggregate(stable_runs)

    ts = datetime.now().strftime("%Y%m%d")
    default_prefix = Path("results") / f"stability_sensitivity_{ms_path.stem}_{ts}"
    out_prefix = Path(args.out_prefix) if args.out_prefix else default_prefix
    if out_prefix.suffix:
        out_prefix = out_prefix.with_suffix("")

    payload = {
        "multiseed": str(ms_path),
        "unstable_seeds": unstable_seeds,
        "n_unstable_runs": len(unstable_seeds),
        "aggregate_all_runs": agg_all,
        "aggregate_stable_only": agg_stable,
    }
    out_json = out_prefix.with_suffix(".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fmt(x):
        if x is None:
            return ""
        try:
            v = float(x)
            if abs(v) < 1e-4 and v != 0.0:
                return f"{v:.3e}"
            return f"{v:.6f}"
        except Exception:
            return str(x)

    md = [
        f"# Stability Sensitivity ({ms_path.stem})",
        "",
        f"Source: `{ms_path}`",
        "",
        f"Unstable runs: **{len(unstable_seeds)}**; seeds: `{unstable_seeds}`",
        "",
        "| Aggregation | n_runs | ΔR² mean | ΔR² min | ΔR² max | wins | losses | fallback_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| all runs | {agg_all.get('n_runs','')} | {_fmt(agg_all.get('delta_r2_mean'))} | {_fmt(agg_all.get('delta_r2_min'))} | {_fmt(agg_all.get('delta_r2_max'))} | {agg_all.get('wins','')} | {agg_all.get('losses','')} | {_fmt(agg_all.get('fallback_rate'))} |",
        f"| stable only | {agg_stable.get('n_runs','')} | {_fmt(agg_stable.get('delta_r2_mean'))} | {_fmt(agg_stable.get('delta_r2_min'))} | {_fmt(agg_stable.get('delta_r2_max'))} | {agg_stable.get('wins','')} | {agg_stable.get('losses','')} | {_fmt(agg_stable.get('fallback_rate'))} |",
        "",
    ]
    out_md = out_prefix.with_suffix(".md")
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"saved: {out_json}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()

