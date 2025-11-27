#!/usr/bin/env python3
"""
Corrige labels YOLO em data/clean_yolo/labels:
- Remove linhas com class_id fora [0..10]
- Remove boxes com coords fora de [0,1] ou área <= 16 px² (em pixels reais)
- Deduplica boxes da mesma classe com IoU > 0.9 (mantém maior área)
- Não remove imagens; só regrava labels
Logs: logs/changes.log e data/stats/fix_summary.json
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image


ROOT = Path("data/clean_yolo")
LBL = ROOT / "labels"
IMG = ROOT / "images"
LOGS = Path("logs/changes.log")
SUMMARY = Path("data/stats/fix_summary.json")


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
    # busca recursiva
    for ext in (".jpg", ".jpeg", ".png"):
        cands = list(IMG.rglob(f"**/{stem}{ext}"))
        if cands:
            return cands[0]
    return None


def main() -> int:
    LOGS.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    removed_invalid = 0
    removed_dupes = 0
    files_changed = 0

    for lbl in sorted(LBL.glob("*.txt")):
        stem = lbl.stem
        imgp = find_image_for(stem)
        if imgp is None:
            # Sem imagem correspondente; não apaga label, só registra
            LOGS.write_text(LOGS.read_text(encoding="utf-8") + f"orphan_label {lbl}\n" if LOGS.exists() else f"orphan_label {lbl}\n", encoding="utf-8")
            continue
        try:
            with Image.open(imgp) as im:
                W, H = im.size
        except Exception:
            continue

        lines = [ln.strip() for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            continue

        # filtrar inválidos
        parsed: List[Tuple[int, float, float, float, float]] = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                removed_invalid += 1
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
            except Exception:
                removed_invalid += 1
                continue
            if not (0 <= cid <= 10):
                removed_invalid += 1
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                removed_invalid += 1
                continue
            x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, W, H)
            if not (x2 > x1 and y2 > y1):
                removed_invalid += 1
                continue
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area <= 16.0:
                removed_invalid += 1
                continue
            parsed.append((cid, cx, cy, w, h))

        # deduplicar por classe (IoU>0.9) mantendo maior área
        kept: List[Tuple[int, float, float, float, float]] = []
        for i in range(len(parsed)):
            ci, cxi, cyi, wi, hi = parsed[i]
            ai = wi * hi
            bi = yolo_to_xyxy(cxi, cyi, wi, hi, W, H)
            duplicate = False
            for j in range(len(kept)):
                cj, cxj, cyj, wj, hj = kept[j]
                if ci != cj:
                    continue
                bj = yolo_to_xyxy(cxj, cyj, wj, hj, W, H)
                if iou(bi, bj) > 0.9:
                    aj = wj * hj
                    if ai > aj:
                        kept[j] = parsed[i]
                    removed_dupes += 1
                    duplicate = True
                    break
            if not duplicate:
                kept.append(parsed[i])

        # salvar se mudou
        new_lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h in kept]
        if new_lines != lines:
            files_changed += 1
            LOGS.write_text(
                (LOGS.read_text(encoding="utf-8") if LOGS.exists() else "") + f"fixed {lbl.name} removed_invalid={removed_invalid} removed_dupes={removed_dupes}\n",
                encoding="utf-8",
            )
            lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    SUMMARY.write_text(
        json.dumps(
            {
                "removed_invalid": removed_invalid,
                "removed_dupes": removed_dupes,
                "files_changed": files_changed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"removed_invalid={removed_invalid} removed_dupes={removed_dupes} files_changed={files_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


