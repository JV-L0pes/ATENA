#!/usr/bin/env python3
import sys
from pathlib import Path


def main() -> int:
    p = Path("data/merged_yolo/splits/split_pseudo_camera.txt")
    if not p.exists():
        print("[WARN] split_pseudo_camera.txt não encontrado; gere via 04_dedup_and_split.py")
        return 0
    seen = {}
    ok = True
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            stem, pcid, tag = parts
            prev = seen.get(pcid)
            if prev is None:
                seen[pcid] = tag
            elif prev != tag and prev != "ignore" and tag != "ignore":
                print(f"[FAIL] pseudo_camera_id {pcid} em múltiplos conjuntos: {prev} vs {tag}")
                ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


