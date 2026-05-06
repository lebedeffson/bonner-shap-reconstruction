#!/usr/bin/env python3
"""Remove heavy local artifacts from results/ (safe by default)."""

import argparse
from pathlib import Path


HEAVY_SUFFIXES = {".pt", ".npy", ".png", ".pdf"}


def main():
    ap = argparse.ArgumentParser(description="Cleanup heavy results artifacts")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--apply", action="store_true", help="Actually delete files")
    args = ap.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f"Not found: {root}")
        return

    targets = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in HEAVY_SUFFIXES]
    total_bytes = sum(p.stat().st_size for p in targets)
    print(f"Found files: {len(targets)}")
    print(f"Total size: {total_bytes / (1024 * 1024):.2f} MB")

    preview = targets[:40]
    for p in preview:
        print(f"- {p}")
    if len(targets) > len(preview):
        print(f"... and {len(targets) - len(preview)} more")

    if not args.apply:
        print("Dry run only. Use --apply to delete.")
        return

    for p in targets:
        p.unlink(missing_ok=True)
    print("Done.")


if __name__ == "__main__":
    main()

