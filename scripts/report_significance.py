#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon, ttest_rel
except Exception:  # pragma: no cover
    wilcoxon = None
    ttest_rel = None


def parse_args():
    p = argparse.ArgumentParser(description="Paired significance for two multiseed explainability JSON files")
    p.add_argument("--a", required=True, help="JSON A")
    p.add_argument("--b", required=True, help="JSON B")
    p.add_argument("--metric", default="auc_deletion_gap_top_bottom", help="Run-level metric key")
    p.add_argument("--out", default="")
    return p.parse_args()


def _get_metric(run, key):
    if key in run:
        return run[key]
    if key == "auc_deletion_gap_top_bottom":
        if "auc_deletion_top" in run and "auc_deletion_bottom" in run:
            return float(run["auc_deletion_top"]) - float(run["auc_deletion_bottom"])
    return None


def cliffs_delta(x, y):
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    n = len(x) * len(y)
    return float((gt - lt) / n) if n else float("nan")


def main():
    args = parse_args()
    A = json.load(open(args.a, "r", encoding="utf-8"))
    B = json.load(open(args.b, "r", encoding="utf-8"))

    runs_a = {int(r["seed"]): r for r in A.get("runs", []) if "seed" in r}
    runs_b = {int(r["seed"]): r for r in B.get("runs", []) if "seed" in r}
    seeds = sorted(set(runs_a) & set(runs_b))
    xa, xb = [], []
    for s in seeds:
        va = _get_metric(runs_a[s], args.metric)
        vb = _get_metric(runs_b[s], args.metric)
        if va is None or vb is None:
            continue
        xa.append(float(va))
        xb.append(float(vb))

    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    d = xa - xb
    mean = float(np.mean(d)) if d.size else float("nan")
    std = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    median = float(np.median(d)) if d.size else float("nan")
    wins = int(np.sum(d > 0))
    losses = int(np.sum(d < 0))
    ci95 = (
        float(mean - 1.96 * std / np.sqrt(max(1, d.size))),
        float(mean + 1.96 * std / np.sqrt(max(1, d.size))),
    ) if d.size else (float("nan"), float("nan"))
    p_wil = None
    p_t = None
    if d.size >= 3 and wilcoxon is not None:
        p_wil = float(wilcoxon(xa, xb, zero_method="wilcox").pvalue)
    if d.size >= 3 and ttest_rel is not None:
        p_t = float(ttest_rel(xa, xb).pvalue)
    denom = float(np.std(d, ddof=1) + 1e-12) if d.size > 1 else float("nan")
    cohen_d = float(mean / denom) if d.size > 1 else float("nan")
    cliff = cliffs_delta(xa, xb) if d.size else float("nan")

    out = {
        "file_a": str(Path(args.a).resolve()),
        "file_b": str(Path(args.b).resolve()),
        "metric": args.metric,
        "n_pairs": int(d.size),
        "seeds": seeds,
        "delta_mean": mean,
        "delta_std": std,
        "delta_median": median,
        "ci95": {"low": ci95[0], "high": ci95[1]},
        "wins": wins,
        "losses": losses,
        "wilcoxon_p": p_wil,
        "ttest_p": p_t,
        "cohen_d": cohen_d,
        "cliffs_delta": cliff,
    }

    out_path = args.out or f"results/significance_{Path(args.a).stem}_vs_{Path(args.b).stem}_{args.metric}.json"
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

