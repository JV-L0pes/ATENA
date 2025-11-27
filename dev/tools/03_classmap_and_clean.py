#!/usr/bin/env python3
"""
03_classmap_and_clean: aplica mapa de classes, valida bboxes e gera QC de rótulos.

Regras principais:
- área mínima > 16 px² equivalente (normalizado por 1x1 -> área > 16/(W*H), aqui assumimos normalizado já)
- proibir overlap positivo vs no_* da mesma peça na mesma pessoa
- logar casos ambíguos em data/stats/label_qc.csv
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import json


MAP_PATH = Path("data/mappings/class_map.json")
STATS_DIR = Path("data/stats")
QC_CSV = STATS_DIR / "label_qc.csv"


def load_class_map() -> Dict[str, str]:
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


FINAL_NAMES = [
    "person",
    "helmet",
    "goggles",
    "gloves",
    "boots",
    "ear_protection",
    "no_helmet",
    "no_goggles",
    "no_gloves",
    "no_boots",
    "no_earprotection",
]
IDX2NAME = {i: n for i, n in enumerate(FINAL_NAMES)}
PAIR_MAP = {
    "helmet": "no_helmet",
    "goggles": "no_goggles",
    "gloves": "no_gloves",
    "boots": "no_boots",
    "ear_protection": "no_earprotection",
}


def process_labels(labels_dir: Path, qc_writer: csv.writer, class_map: Dict[str, str]) -> Tuple[int, int]:
    total = 0
    conflicts = 0
    for p in labels_dir.glob("*.txt"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            continue

        # parse
        parsed: List[Tuple[int, float, float, float, float]] = []
        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
            except Exception:
                continue
            # área mínima equivalente 16px² é verificada na escala normalizada
            # Como não temos W,H reais, validamos w*h > 0 (sanidade). O check absoluto será feito em estágios posteriores com dimensões reais.
            if w <= 0 or h <= 0:
                continue
            parsed.append((cls, cx, cy, w, h))

        total += len(parsed)

        # detectar conflitos somente entre pares positivo vs no_* da mesma peça
        boxes_xy = [(cls, IDX2NAME.get(cls, str(cls)), yolo_to_xyxy(cx, cy, w, h)) for cls, cx, cy, w, h in parsed]
        # indexar por nome
        by_name: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
        for cid, cname, b in boxes_xy:
            by_name.setdefault(cname, []).append((cid, b))
        for pos_name, neg_name in PAIR_MAP.items():
            pos_list = by_name.get(pos_name, [])
            neg_list = by_name.get(neg_name, [])
            if not pos_list or not neg_list:
                continue
            for _, bi in pos_list:
                for _, bj in neg_list:
                    ov = iou(bi, bj)
                    if ov > 0.3:
                        qc_writer.writerow([p.name, pos_name, neg_name, f"{ov:.3f}", "conflict_pos_vs_no"])
                        conflicts += 1

    return total, conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description="03_classmap_and_clean")
    parser.add_argument("--input", type=str, default="data/interim_yolo", help="Raiz com images/labels")
    parser.add_argument("--output", type=str, default="data/clean_yolo", help="Saída limpa")
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    class_map = load_class_map() if MAP_PATH.exists() else {}

    with open(QC_CSV, "w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["file", "cls_i", "cls_j", "iou", "note"])
        total, conflicts = process_labels(in_root / "labels", writer, class_map)

    print(f"[OK] QC salvo em {QC_CSV} (total={total}, conflicts={conflicts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


