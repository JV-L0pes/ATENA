#!/usr/bin/env python3
"""
Aggregate múltiplos datasets YOLO recursivamente em um pool único data/clean_yolo/.
Procura pastas train/val/valid/test sob a raiz e copia images/ e labels/.
Garante nomes únicos prefixando com slug derivado do path relativo.
"""

import argparse
import shutil
from pathlib import Path


def derive_slug(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = [p for p in rel.parts if p not in ("images", "labels", "train", "val", "valid", "test")]
    slug = "-".join(parts) or "root"
    return slug.replace(" ", "_")


def copy_split(root: Path, out_images: Path, out_labels: Path) -> int:
    copied = 0
    for split in ("train", "val", "valid", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        slug = derive_slug(root, root.parent)
        for p in img_dir.rglob("*.*"):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = p.stem
            lbl = lbl_dir / f"{stem}.txt"
            if not lbl.exists():
                continue
            # construir nomes únicos
            dst_img = out_images / f"{slug}__{stem}{p.suffix.lower()}"
            dst_lbl = out_labels / f"{slug}__{stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst_img)
            shutil.copy2(lbl, dst_lbl)
            copied += 1
    return copied


def main() -> int:
    ap = argparse.ArgumentParser(description="aggregate_yolo")
    ap.add_argument("--root", type=str, default="datasets/datasets")
    ap.add_argument("--dst", type=str, default="data/clean_yolo")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.dst)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    total = 0
    # Copiar níveis conhecidos
    total += copy_split(root, out / "images", out / "labels")
    # Recursivo: varrer subpastas que contenham train/val/test
    for sub in root.rglob("*"):
        if not sub.is_dir():
            continue
        if (sub / "train" / "images").exists() and (sub / "train" / "labels").exists():
            total += copy_split(sub, out / "images", out / "labels")

    print(f"[OK] Aggregate: {total} imagens rotuladas copiadas para {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


