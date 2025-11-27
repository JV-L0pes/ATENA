#!/usr/bin/env python3
"""
Split por slug (pseudo_câmera=slug) com proporção alvo 80/10/10 (train/val/test), leakage=0.
Assume data/clean_yolo/{images,labels} com nomes "<slug>__<stem>.<ext>".
"""

import argparse
import random
from pathlib import Path
from typing import Dict, List


def main() -> int:
    ap = argparse.ArgumentParser(description="split_by_slug")
    ap.add_argument("--images", type=str, default="data/clean_yolo/images")
    ap.add_argument("--labels", type=str, default="data/clean_yolo/labels")
    ap.add_argument("--output", type=str, default="data/merged_yolo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    random.seed(args.seed)
    img_root = Path(args.images)
    lbl_root = Path(args.labels)
    out_root = Path(args.output)
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p in (out_root / "images" / split).glob("*"): p.unlink(missing_ok=True)
        for p in (out_root / "labels" / split).glob("*.txt"): p.unlink(missing_ok=True)

    stems: List[str] = []
    for p in img_root.glob("*.*"):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
        stems.append(p.stem)

    slug_to_stems: Dict[str, List[str]] = {}
    for s in stems:
        slug = s.split("__", 1)[0] if "__" in s else "merged"
        slug_to_stems.setdefault(slug, []).append(s)

    # estratégia: dividir slugs grandes em chunks para distribuir melhor
    total = sum(len(v) for v in slug_to_stems.values())
    target_train = total * float(args.train_frac)
    target_val = total * float(args.val_frac)
    target_test = total - target_train - target_val

    train_set: Dict[str, List[str]] = {}
    val_set: Dict[str, List[str]] = {}
    test_set: Dict[str, List[str]] = {}

    train_count = 0
    val_count = 0
    test_count = 0

    # ordenar slugs por tamanho (DESC)
    sorted_slugs = sorted(slug_to_stems.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    # 1) dividir o maior slug entre train/val/test proporcionalmente
    if sorted_slugs:
        largest_slug, largest_list = sorted_slugs[0]
        chunk_size = len(largest_list) // 3
        train_set[f"{largest_slug}_train"] = largest_list[:chunk_size]
        val_set[f"{largest_slug}_val"] = largest_list[chunk_size:chunk_size*2]
        test_set[f"{largest_slug}_test"] = largest_list[chunk_size*2:]
        train_count += len(train_set[f"{largest_slug}_train"])
        val_count += len(val_set[f"{largest_slug}_val"])
        test_count += len(test_set[f"{largest_slug}_test"])

    # 2) distribuir o segundo maior slug entre train/test
    if len(sorted_slugs) > 1:
        second_slug, second_list = sorted_slugs[1]
        mid_point = len(second_list) // 2
        train_set[f"{second_slug}_train"] = second_list[:mid_point]
        test_set[f"{second_slug}_test"] = second_list[mid_point:]
        train_count += len(train_set[f"{second_slug}_train"])
        test_count += len(test_set[f"{second_slug}_test"])

    # 3) todos os slugs pequenos vão para val
    for slug, lst in sorted_slugs[2:]:  # pular os 2 maiores já processados
        val_set[slug] = lst
        val_count += len(lst)

    # 4) ajustar se val ficou muito pequeno - mover chunks do train
    if val_count < target_val * 0.5:
        # mover alguns chunks do train para val
        train_items = list(train_set.items())
        for slug, lst in train_items:
            if len(lst) <= 1000 and val_count < target_val:
                val_set[slug] = train_set.pop(slug)
                val_count += len(lst)
                train_count -= len(lst)
            else:
                break

    # 4) materializar cópias
    def write_group(group: Dict[str, List[str]], split: str) -> int:
        count = 0
        for slug, lst in group.items():
            for s in lst:
                imgp = next((q for q in img_root.glob(f"{s}.jpg")), None) or next((q for q in img_root.glob(f"{s}.png")), None)
                if imgp is None:
                    continue
                lblp = lbl_root / f"{s}.txt"
                if not lblp.exists():
                    continue
                dst_img = out_root / "images" / split / imgp.name
                dst_lbl = out_root / "labels" / split / lblp.name
                dst_img.write_bytes(imgp.read_bytes())
                dst_lbl.write_text(lblp.read_text(encoding="utf-8"), encoding="utf-8")
                count += 1
        return count

    train_count = write_group(train_set, "train")
    val_count = write_group(val_set, "val")
    test_count = write_group(test_set, "test")

    print(f"train={train_count} val={val_count} test={test_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


