#!/usr/bin/env python3
"""
Validação aprofundada de labels YOLO em data/clean_yolo/:
- IDs válidos [0..10] (11 classes)
- Coords 0<=cx,cy<=1 e 0<w,h<=1; x1<y2 etc.
- Área em pixels > 16 px² (usa dimensões reais da imagem)
- Duplicatas por IoU>0.9 no mesmo arquivo
- Arquivos órfãos (imagem sem label e label sem imagem)
Saídas:
- data/stats/label_validation.csv (por erro)
- sumário no stdout
"""

import csv
import math
from pathlib import Path
from typing import List, Tuple

from PIL import Image


ROOT = Path("data/clean_yolo")
LBL = ROOT / "labels"
IMG = ROOT / "images"
OUT_CSV = Path("data/stats/label_validation.csv")


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return x1, y1, x2, y2


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_image_for(stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = IMG / f"{stem}{ext}"
        if p.exists():
            return p
    # fallback: busca recursiva (nome único)
    for ext in (".jpg", ".jpeg", ".png"):
        candidates = list(IMG.rglob(f"**/{stem}{ext}"))
        if candidates:
            return candidates[0]
    return None


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    invalid = 0
    dupes = 0
    orphan_imgs = 0
    orphan_lbls = 0
    total = 0

    img_stems = set(p.stem for p in IMG.rglob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    lbl_stems = set(p.stem for p in LBL.glob("*.txt"))

    # órfãos
    for s in sorted(img_stems - lbl_stems):
        orphan_imgs += 1
    for s in sorted(lbl_stems - img_stems):
        orphan_lbls += 1

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["file", "issue", "detail"])

        for lbl in sorted(LBL.glob("*.txt")):
            stem = lbl.stem
            imgp = find_image_for(stem)
            if imgp is None:
                wr.writerow([lbl.name, "orphan_label", "no_image_found"])
                invalid += 1
                continue
            try:
                with Image.open(imgp) as im:
                    W, H = im.size
            except Exception:
                wr.writerow([lbl.name, "image_open_fail", str(imgp)])
                invalid += 1
                continue

            # ler boxes e validar
            boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
            try:
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    parts = line.split()
                    if len(parts) < 5:
                        wr.writerow([lbl.name, "parse_error", line])
                        invalid += 1
                        continue
                    try:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                    except Exception:
                        wr.writerow([lbl.name, "parse_error", line])
                        invalid += 1
                        continue
                    if not (0 <= cid <= 10):
                        wr.writerow([lbl.name, "class_id_out_of_range", cid])
                        invalid += 1
                        continue
                    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                        wr.writerow([lbl.name, "coords_out_of_bounds", f"{cx},{cy},{w},{h}"])
                        invalid += 1
                        continue
                    x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
                    if not (x2 > x1 and y2 > y1):
                        wr.writerow([lbl.name, "invalid_box_order", f"{x1},{y1},{x2},{y2}"])
                        invalid += 1
                        continue
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    if area <= 16.0:
                        wr.writerow([lbl.name, "area_too_small", f"{area:.2f}"])
                        invalid += 1
                        continue
                    boxes.append((cid, (x1, y1, x2, y2)))
            except Exception as e:
                wr.writerow([lbl.name, "read_fail", str(e)])
                invalid += 1
                continue

            # duplicatas (mesma classe IoU>0.9)
            for i in range(len(boxes)):
                ci, bi = boxes[i]
                for j in range(i + 1, len(boxes)):
                    cj, bj = boxes[j]
                    if ci != cj:
                        continue
                    if iou(bi, bj) > 0.9:
                        wr.writerow([lbl.name, "duplicate_box", f"cls={ci}"])
                        dupes += 1
                        break

    print(f"invalid={invalid} dupes={dupes} orphan_imgs={orphan_imgs} orphan_lbls={orphan_lbls} total_anns={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


