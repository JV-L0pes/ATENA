#!/usr/bin/env python3
"""
Sincroniza um dataset YOLO já estruturado (train/val/test) para um pool único
em data/clean_yolo/{images,labels}, preservando nomes de arquivos.
"""

import argparse
import shutil
from pathlib import Path


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            outp = dst / rel.name
            outp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, outp)


def main() -> int:
    parser = argparse.ArgumentParser(description="sync_from_yolo")
    parser.add_argument("--src", type=str, required=True, help="Raiz com train/val/test")
    parser.add_argument("--dst", type=str, default="data/clean_yolo", help="Destino pool único")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "val", "test"):
        copy_tree(src / split / "images", dst / "images")
        copy_tree(src / split / "labels", dst / "labels")

    print(f"[OK] Sincronizado {src} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


