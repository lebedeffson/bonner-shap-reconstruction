#!/usr/bin/env python3
"""Report alignment metrics between two feature-importance vectors."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _load_importance(path: Path):
    df = pd.read_csv(path, index_col=0)
    if "importance" not in df.columns:
        raise ValueError(f"'importance' column not found in {path}")
    s = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
    s = s.clip(lower=0.0)
    return s


def _normalize(v: np.ndarray):
    v = np.asarray(v, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    s = float(v.sum())
    if s <= 1e-12:
        return np.full_like(v, 1.0 / max(len(v), 1), dtype=float)
    return v / s


def _rankdata(a: np.ndarray):
    # Simple average-rank for ties
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    if den <= 1e-12:
        return 0.0
    return float((x * y).sum() / den)


def _spearman(x: np.ndarray, y: np.ndarray):
    return _pearson(_rankdata(x), _rankdata(y))


def _cosine(x: np.ndarray, y: np.ndarray):
    den = np.linalg.norm(x) * np.linalg.norm(y)
    if den <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / den)


def _js_divergence(p: np.ndarray, q: np.ndarray):
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def _topk_overlap(a: pd.Series, b: pd.Series, k: int):
    ka = set(a.sort_values(ascending=False).head(k).index)
    kb = set(b.sort_values(ascending=False).head(k).index)
    inter = len(ka & kb)
    union = len(ka | kb)
    jacc = inter / union if union else 0.0
    return inter, jacc


def main():
    parser = argparse.ArgumentParser(description="Importance alignment report")
    parser.add_argument("--ref", required=True, help="reference importance csv")
    parser.add_argument("--cand", required=True, help="candidate importance csv")
    parser.add_argument("--label-ref", default="ref")
    parser.add_argument("--label-cand", default="cand")
    parser.add_argument("--k", default="3,5")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    ref_path = Path(args.ref)
    cand_path = Path(args.cand)
    ref = _load_importance(ref_path)
    cand = _load_importance(cand_path)

    common = ref.index.intersection(cand.index)
    if len(common) == 0:
        raise ValueError("No common feature names between files")

    ref = ref.loc[common]
    cand = cand.loc[common]

    ref_n = _normalize(ref.values)
    cand_n = _normalize(cand.values)

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    topk = {}
    for k in ks:
        k_eff = min(k, len(common))
        inter, jacc = _topk_overlap(ref, cand, k_eff)
        topk[str(k_eff)] = {"intersection": int(inter), "jaccard": float(jacc)}

    report = {
        "ref": str(ref_path),
        "cand": str(cand_path),
        "label_ref": args.label_ref,
        "label_cand": args.label_cand,
        "n_features_common": int(len(common)),
        "pearson": _pearson(ref_n, cand_n),
        "spearman": _spearman(ref_n, cand_n),
        "cosine_similarity": _cosine(ref_n, cand_n),
        "js_divergence": _js_divergence(ref_n, cand_n),
        "topk": topk,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = Path(args.out_md) if args.out_md else Path("results") / f"importance_alignment_{ts}.md"
    out_json = Path(args.out_json) if args.out_json else Path("results") / f"importance_alignment_{ts}.json"

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Importance Alignment Report",
        "",
        f"- ref: `{args.ref}` ({args.label_ref})",
        f"- cand: `{args.cand}` ({args.label_cand})",
        f"- common features: `{report['n_features_common']}`",
        "",
        f"- pearson: `{report['pearson']:.6f}`",
        f"- spearman: `{report['spearman']:.6f}`",
        f"- cosine similarity: `{report['cosine_similarity']:.6f}`",
        f"- JS divergence: `{report['js_divergence']:.6f}`",
        "",
        "| k | top-k intersection | top-k jaccard |",
        "|---:|---:|---:|",
    ]
    for k, v in report["topk"].items():
        lines.append(f"| {k} | {v['intersection']} | {v['jaccard']:.6f} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved: {out_md}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
