#!/usr/bin/env python3
"""Practical readiness gate for SHAP-regularized runs."""

import argparse
import json
from datetime import datetime
from pathlib import Path


def _load_json(path: str):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f), p


def _pick_r2(summary: dict):
    md = summary.get("metrics_denorm", {}) or {}
    m = summary.get("metrics", {}) or {}
    return float(
        md.get("r2_weighted", md.get("r2", m.get("r2_weighted", m.get("r2", 0.0))))
    )


def _float(d: dict, *path, default=0.0):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return float(default)
        cur = cur[key]
    try:
        return float(cur)
    except Exception:
        return float(default)


def _check(name: str, value: float, thr: float, mode: str):
    if mode == "ge":
        ok = value >= thr
        rule = f">= {thr}"
    else:
        ok = value <= thr
        rule = f"<= {thr}"
    return {"name": name, "value": float(value), "rule": rule, "pass": bool(ok)}


def main():
    ap = argparse.ArgumentParser(description="Practical readiness gate")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--faithfulness", default="")
    ap.add_argument("--alignment", default="")
    ap.add_argument("--r2-min", type=float, default=0.80)
    ap.add_argument("--reg-share-min", type=float, default=0.05)
    ap.add_argument("--shap-contrib-min", type=float, default=1e-4)
    ap.add_argument("--auc-gap-min", type=float, default=0.0)
    ap.add_argument("--top-random-min", type=float, default=1.0)
    ap.add_argument("--align-cos-min", type=float, default=0.65)
    ap.add_argument("--out-md", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    summary, summary_path = _load_json(args.summary)
    reg = ((summary.get("diagnostics") or {}).get("regularization") or {})

    checks = []
    checks.append(_check("r2_weighted", _pick_r2(summary), args.r2_min, "ge"))
    checks.append(
        _check(
            "regularization_share_mean",
            _float(reg, "regularization_share", "mean"),
            args.reg_share_min,
            "ge",
        )
    )
    checks.append(
        _check(
            "shap_contribution_mean",
            _float(reg, "shap_contribution", "mean"),
            args.shap_contrib_min,
            "ge",
        )
    )

    faithfulness_used = ""
    if args.faithfulness:
        faith, fp = _load_json(args.faithfulness)
        faithfulness_used = str(fp)
        checks.append(
            _check(
                "faithfulness_auc_gap",
                float(faith.get("auc_gap_top_minus_bottom", 0.0)),
                args.auc_gap_min,
                "ge",
            )
        )
        checks.append(
            _check(
                "faithfulness_top_random_ratio",
                float(faith.get("top_random_ratio", 0.0)),
                args.top_random_min,
                "ge",
            )
        )

    alignment_used = ""
    if args.alignment:
        ali, apath = _load_json(args.alignment)
        alignment_used = str(apath)
        checks.append(
            _check(
                "alignment_cosine",
                float(ali.get("cosine_similarity", 0.0)),
                args.align_cos_min,
                "ge",
            )
        )

    overall = all(c["pass"] for c in checks)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.out_json) if args.out_json else Path("results") / f"practical_gate_{ts}.json"
    out_md = Path(args.out_md) if args.out_md else Path("results") / f"practical_gate_{ts}.md"

    report = {
        "summary": str(summary_path),
        "faithfulness": faithfulness_used,
        "alignment": alignment_used,
        "overall_pass": bool(overall),
        "checks": checks,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Practical Readiness Gate",
        "",
        f"- summary: `{summary_path}`",
        f"- faithfulness: `{faithfulness_used or '-'} `",
        f"- alignment: `{alignment_used or '-'} `",
        "",
        f"- overall: `{'PASS' if overall else 'FAIL'}`",
        "",
        "| check | value | rule | status |",
        "|---|---:|---:|---|",
    ]
    for c in checks:
        lines.append(
            f"| {c['name']} | {c['value']:.6f} | {c['rule']} | {'PASS' if c['pass'] else 'FAIL'} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {out_md}")
    print(f"Saved: {out_json}")
    print(f"Overall: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()

