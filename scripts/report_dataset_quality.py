#!/usr/bin/env python3
"""Быстрый отчёт о качестве признаков по CSV датасетам."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.data_quality import summarize_feature_quality


def parse_args():
    parser = argparse.ArgumentParser(description="Report feature quality")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Формат: path/to.csv:target1,target2",
    )
    parser.add_argument("--out", default="results/dataset_quality_reports", help="Output dir")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for dataset_spec in args.dataset:
        csv_path, targets_raw = dataset_spec.split(":", 1)
        targets = [t.strip() for t in targets_raw.split(",") if t.strip()]
        path = Path(csv_path)
        df = pd.read_csv(path)
        features = [c for c in df.columns if c not in set(targets)]
        quality = summarize_feature_quality(df[features], corr_threshold=0.9999, max_pairs=50)
        payload = {
            "dataset": path.name,
            "rows": int(df.shape[0]),
            "features": int(len(features)),
            "targets": targets,
            "quality": quality,
        }
        out_path = out_dir / f"{path.stem}_quality.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
