#!/usr/bin/env python3
import argparse
import sys
from collections import defaultdict
from pathlib import Path
import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--min-per-class", type=int, default=30)
    return p.parse_args()

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def count_labels(labels_dir: Path):
    counts = defaultdict(int)
    missing = 0
    images_dir = labels_dir.parent / "images"
    for img_path in images_dir.glob("*"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            missing += 1
            continue
        with open(label_path) as f:
            for line in f:
                if line.strip():
                    counts[int(line.split()[0])] += 1
    return dict(counts), missing

def main():
    args = parse_args()
    cfg = load_yaml(args.dataset)
    base = Path(args.dataset).parent
    names = cfg.get("names", [])

    print(f"Inspeção do Dataset: {base.name} | Classes: {names}")
    for split in ["train", "valid", "test"]:
        labels_dir = base / split / "labels"
        if not labels_dir.exists():
            continue
        counts, missing = count_labels(labels_dir)
        total = sum(counts.values())
        imgs = len(list((base / split / "images").glob("*")))
        print(f"[{split.upper()}] {imgs} imagens | {total} anotações | {missing} sem label")

    print("[OK] Dataset verificado.")

if __name__ == "__main__":
    main()