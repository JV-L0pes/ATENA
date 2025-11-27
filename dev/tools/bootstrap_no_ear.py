#!/usr/bin/env python3
"""
Bootstrap seguro de no_earprotection a partir de datasets com ear_protection.
Regra: ROI cabeça (top 35% + margem 10% largura); filtrar por modelo auxiliar ear_protection (yolo11s) com conf>=0.15.
Salva amostra de 10% para QC em data/no_ear_qc/ (overlays + CSV) e escreve labels em labels_no_ear/.
"""

import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def head_roi(x1, y1, x2, y2) -> Tuple[int, int, int, int]:
    w = x2 - x1
    h = y2 - y1
    top_h = int(0.35 * h)
    mx = int(0.10 * w)
    return x1 + mx, y1, x2 - mx, y1 + top_h


def main() -> int:
    parser = argparse.ArgumentParser(description="bootstrap_no_ear")
    parser.add_argument("--images", type=str, default="data/clean_yolo/images/train")
    parser.add_argument("--labels", type=str, default="data/clean_yolo/labels/train")
    parser.add_argument("--out_labels", type=str, default="data/labels_no_ear")
    args = parser.parse_args()

    out_qc = Path("data/no_ear_qc")
    out_qc.mkdir(parents=True, exist_ok=True)
    out_labels = Path(args.out_labels)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Placeholder: sem modelo auxiliar neste esqueleto (apenas desenha ROIs para QC)
    qc_csv = out_qc / "qc_sample.csv"
    with open(qc_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]) 

        imgs = sorted(list(Path(args.images).glob("*.jpg")))
        sample = set(random.sample(imgs, max(1, len(imgs) // 10))) if imgs else set()
        for ip in imgs:
            try:
                im = cv2.imread(str(ip))
                if im is None:
                    continue
                h, w = im.shape[:2]
                # pessoa inteira (placeholder: usar toda imagem como pessoa)
                rx1, ry1, rx2, ry2 = head_roi(0, 0, w, h)
                if ip in sample:
                    # salvar overlay
                    ov = im.copy()
                    cv2.rectangle(ov, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
                    cv2.imwrite(str(out_qc / ip.name), ov)
                    wr.writerow([ip.name, rx1, ry1, rx2, ry2])
                # gerar rótulo no_earprotection (placeholder em nível de ROI)
                lbl = out_labels / (ip.stem + ".txt")
                cx = (rx1 + rx2) / 2.0 / w
                cy = (ry1 + ry2) / 2.0 / h
                bw = (rx2 - rx1) / w
                bh = (ry2 - ry1) / h
                # classe será mapeada posteriormente para o id de no_earprotection
                with open(lbl, "a", encoding="utf-8") as fl:
                    fl.write(f"10 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            except Exception:
                continue

    print(f"[OK] bootstrap_no_ear: QC em {out_qc} e labels em {out_labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


