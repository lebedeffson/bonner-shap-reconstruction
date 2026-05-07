#!/usr/bin/env python3
"""
Compute SHAP scaling/cost diagnostics from one or more training summaries.
"""

import argparse
import glob
import json
import os
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="SHAP scaling report from summaries")
    p.add_argument("--summaries-glob", default="results/training_summary_*.json")
    p.add_argument("--require-shap-compute", action="store_true")
    p.add_argument("--output-json", default="")
    p.add_argument("--output-md", default="")
    return p.parse_args()


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.summaries_glob))
    rows = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            continue
        diag = s.get("diagnostics", {})
        sc = diag.get("shap_compute", {}) or {}
        if args.require_shap_compute and not sc:
            continue
        rows.append(
            {
                "summary": os.path.abspath(p),
                "tag": s.get("tag"),
                "train_size": s.get("shap_train_size"),
                "test_size": s.get("test_size"),
                "training_time_total": s.get("training_time_total"),
                "training_time_shap": s.get("training_time_shap"),
                "exact_calls": sc.get("exact_calls"),
                "exact_total_coalitions": sc.get("exact_total_coalitions"),
                "exact_total_utility_evals": sc.get("exact_total_utility_evals"),
                "permutation_calls": sc.get("permutation_calls"),
                "permutation_total_utility_evals": sc.get("permutation_total_utility_evals"),
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = args.output_json or f"results/shap_scaling_report_{ts}.json"
    out_md = args.output_md or f"results/shap_scaling_report_{ts}.md"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, ensure_ascii=False, indent=2)

    lines = [
        "# SHAP Scaling Report",
        "",
        f"- Source glob: `{args.summaries_glob}`",
        f"- Rows: `{len(rows)}`",
        "",
        "| Tag | Train | Test | Time total (s) | Time SHAP (s) | Exact calls | Exact utility evals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('tag','')} | "
            f"{r.get('train_size','')} | {r.get('test_size','')} | "
            f"{r.get('training_time_total','')} | {r.get('training_time_shap','')} | "
            f"{r.get('exact_calls','')} | {r.get('exact_total_utility_evals','')} |"
        )
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()

