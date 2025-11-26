#!/usr/bin/env python3
"""
ingest_domain: mescla rótulos do cliente (YOLO) em data/merged_yolo antes do treino.
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
            outp = dst / rel
            outp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, outp)


def main() -> int:
    parser = argparse.ArgumentParser(description="ingest_domain")
    parser.add_argument("--images", type=str, default="data/domain_yolo/images")
    parser.add_argument("--labels", type=str, default="data/domain_yolo/labels")
    parser.add_argument("--merged", type=str, default="data/merged_yolo")
    args = parser.parse_args()

    merged = Path(args.merged)
    copy_tree(Path(args.images), merged / "images" / "train")
    copy_tree(Path(args.labels), merged / "labels" / "train")
    print("[OK] ingest_domain: rótulos do cliente mesclados em train/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


