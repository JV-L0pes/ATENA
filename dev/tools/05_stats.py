#!/usr/bin/env python3
"""
05_stats: estatísticas por classe e origem + verificação de vazamento pHash.

Saídas:
- data/stats/imbalance.json (contagens por classe)
- data/stats/leakage.json (percentual de pares val/test com vizinho no train a Hamming<=5)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import imagehash


STATS_DIR = Path("data/stats")


def load_yolo_names(dataset_yaml: Path) -> Dict[int, str]:
    import yaml  # type: ignore

    with open(dataset_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def count_classes(labels_dir: Path, names: Dict[int, str]) -> Dict[str, int]:
    counts = {n: 0 for n in names.values()}
    for p in labels_dir.glob("*.txt"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    cname = names.get(cid)
                    if cname is not None:
                        counts[cname] = counts.get(cname, 0) + 1
        except Exception:
            continue
    return counts


def phash_dir(images_dir: Path) -> Dict[str, imagehash.ImageHash]:
    hashes: Dict[str, imagehash.ImageHash] = {}
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        for p in images_dir.glob(ext):
            try:
                with Image.open(p) as im:
                    hashes[p.stem] = imagehash.phash(im, hash_size=64)
            except Exception:
                continue
    return hashes


def leakage_rate(train_hashes: Dict[str, imagehash.ImageHash], other_hashes: Dict[str, imagehash.ImageHash]) -> float:
    if not other_hashes or not train_hashes:
        return 0.0
    leaked = 0
    total = 0
    train_vals = list(train_hashes.values())
    for _, hv in other_hashes.items():
        total += 1
        if any((hv - tv) <= 5 for tv in train_vals):
            leaked += 1
    return leaked / total if total > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="05_stats")
    parser.add_argument("--dataset-yaml", type=str, default="data/merged_yolo/dataset.yaml")
    parser.add_argument("--root", type=str, default="data/merged_yolo")
    args = parser.parse_args()

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    names = load_yolo_names(Path(args.dataset_yaml))

    # Contagens por classe (train/val/test)
    out_counts = {}
    for split in ("train", "val", "test"):
        labels_dir = root / "labels" / split
        if labels_dir.exists():
            out_counts[split] = count_classes(labels_dir, names)
    with open(STATS_DIR / "imbalance.json", "w", encoding="utf-8") as f:
        json.dump(out_counts, f, ensure_ascii=False, indent=2)

    # Vazamento pHash: vizinho no train com Hamming<=5
    train_images = root / "images" / "train"
    val_images = root / "images" / "val"
    test_images = root / "images" / "test"
    leak = {"val_vs_train": 0.0, "test_vs_train": 0.0}
    if train_images.exists():
        tr_h = phash_dir(train_images)
        if val_images.exists():
            leak["val_vs_train"] = leakage_rate(tr_h, phash_dir(val_images))
        if test_images.exists():
            leak["test_vs_train"] = leakage_rate(tr_h, phash_dir(test_images))
    with open(STATS_DIR / "leakage.json", "w", encoding="utf-8") as f:
        json.dump(leak, f, ensure_ascii=False, indent=2)

    print("[OK] 05_stats concluído")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


