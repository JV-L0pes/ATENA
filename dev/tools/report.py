#!/usr/bin/env python3
"""
Gera relatório de robustez: Precision@Recall=0.9 para classes no_* usando predições do split=val.

Entrada padrão:
- Predições (TXT com conf) em runs/ppe/y11x_merged_val/labels
- dataset YAML: data/merged_yolo/dataset_comm.yaml (para nomes de classes e localização de labels val)

Também agrega métricas gerais (placeholders simples):
- mAP/Recall/FNR estimados a partir das predições (simplificado) e salva em data/stats/report_summary.json

Saídas:
- data/stats/robustness.json
- data/stats/report_summary.json
- data/stats/REPORT.md (append da seção de robustez se já existir)
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relatório de robustez P@R=0.9 para no_* classes")
    p.add_argument(
        "--pred_dir",
        type=str,
        default="runs/ppe/y11x_merged_val/labels",
        help="Diretório com TXT de predições (save_txt=True, save_conf=True)",
    )
    p.add_argument(
        "--data_yaml",
        type=str,
        default="data/merged_yolo/dataset_comm.yaml",
        help="Caminho para dataset YAML",
    )
    p.add_argument(
        "--output_md",
        type=str,
        default="data/stats/REPORT.md",
        help="Arquivo de saída Markdown (anexa seção de robustez)",
    )
    p.add_argument(
        "--output_json",
        type=str,
        default="data/stats/robustness.json",
        help="Arquivo JSON com métricas de robustez",
    )
    p.add_argument(
        "--iou_thres",
        type=float,
        default=0.5,
        help="IoU para matching TP/FP",
    )
    return p.parse_args()


def load_dataset_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_val_labels_dir(cfg: Dict) -> Path:
    # Se 'val' é caminho de imagens, converter para labels
    val_entry = cfg.get("val")
    base_path = cfg.get("path")
    if val_entry is None:
        raise RuntimeError("YAML sem entrada 'val'")

    val_path = Path(val_entry)
    # Se caminho relativo, prefixar com 'path'
    if not val_path.is_absolute() and base_path:
        val_path = Path(base_path) / val_path

    # Heurística: trocar /images/ por /labels/
    val_path_str = str(val_path)
    if "/images/" in val_path_str:
        labels_path = Path(val_path_str.replace("/images/", "/labels/"))
    else:
        # Caso a pasta já seja 'labels'
        if val_path.name == "images":
            labels_path = val_path.parent / "labels"
        elif val_path.name == "labels":
            labels_path = val_path
        else:
            # Tentar 'labels' no mesmo nível
            labels_path = val_path.parent / "labels"

    return labels_path


def yolo_txt_to_xyxy(line_parts: List[float]) -> Tuple[float, float, float, float]:
    # YOLO txt usa cx, cy, w, h (normalizados). Aqui retornamos como xyxy ainda normalizado [0..1]
    cx, cy, w, h = line_parts
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_by_stem(labels_dir: Path, names: Dict[int, str]) -> Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]]:
    by_stem: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    if not labels_dir.exists():
        raise RuntimeError(f"Labels de validação não encontrados: {labels_dir}")
    for p in labels_dir.glob("*.txt"):
        stem = p.stem
        cur: List[Tuple[int, Tuple[float, float, float, float]]] = []
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    cur.append((cls_id, yolo_txt_to_xyxy([cx, cy, w, h])))
        except Exception:
            continue
        by_stem[stem] = cur
    return by_stem


def load_pred_by_stem(pred_dir: Path) -> Dict[str, List[Tuple[int, float, Tuple[float, float, float, float]]]]:
    by_stem: Dict[str, List[Tuple[int, float, Tuple[float, float, float, float]]]] = {}
    if not pred_dir.exists():
        raise RuntimeError(f"Predições TXT não encontradas: {pred_dir}")
    for p in pred_dir.glob("*.txt"):
        stem = p.stem
        cur: List[Tuple[int, float, Tuple[float, float, float, float]]] = []
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    # Formato previsto: cls conf cx cy w h
                    if len(parts) < 6:
                        continue
                    cls_id = int(parts[0])
                    conf = float(parts[1])
                    cx, cy, w, h = map(float, parts[2:6])
                    cur.append((cls_id, conf, yolo_txt_to_xyxy([cx, cy, w, h])))
        except Exception:
            continue
        by_stem[stem] = cur
    return by_stem


def compute_p_at_r(
    gt_by_stem: Dict[str, List[Tuple[int, Tuple[float, float, float, float]]]],
    pred_by_stem: Dict[str, List[Tuple[int, float, Tuple[float, float, float, float]]]],
    class_id: int,
    iou_thres: float = 0.5,
) -> Tuple[float, float]:
    """Retorna (precision_at_recall_0_9, max_recall)."""
    # Coletar GTs e preds da classe
    gt_total = 0
    preds: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    for stem, gts in gt_by_stem.items():
        for cid, box in gts:
            if cid == class_id:
                gt_total += 1
    for stem, plist in pred_by_stem.items():
        for cid, conf, box in plist:
            if cid == class_id:
                preds.append((stem, conf, box))

    if gt_total == 0:
        return (float("nan"), 0.0)

    preds.sort(key=lambda x: x[1], reverse=True)

    # Para cada imagem, manter flags de GTs usados
    used: Dict[str, List[bool]] = {}
    gt_boxes_by_stem: Dict[str, List[Tuple[float, float, float, float]]] = {}
    for stem, gts in gt_by_stem.items():
        bxs = [box for cid, box in gts if cid == class_id]
        gt_boxes_by_stem[stem] = bxs
        used[stem] = [False] * len(bxs)

    tp = 0
    fp = 0
    precisions: List[float] = []
    recalls: List[float] = []

    for stem, conf, pbox in preds:
        matched = False
        gtb = gt_boxes_by_stem.get(stem, [])
        flags = used.get(stem, [])
        best_iou = 0.0
        best_j = -1
        for j, (gb) in enumerate(gtb):
            if j < len(flags) and flags[j]:
                continue
            iou = iou_xyxy(pbox, gb)
            if iou >= iou_thres and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            flags[best_j] = True
            matched = True
        if matched:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(1, (tp + fp)))
        recalls.append(tp / gt_total)

    # Encontrar P@R>=0.9
    target_r = 0.9
    p_at_r = float("nan")
    max_r = max(recalls) if recalls else 0.0
    if recalls:
        cand = [p for p, r in zip(precisions, recalls) if r >= target_r]
        if cand:
            p_at_r = max(cand)  # melhor precisão em R>=0.9
    return (p_at_r, max_r)


def main() -> int:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    stats_dir = Path("data/stats")
    stats_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_dataset_yaml(Path(args.data_yaml))
    names_list = cfg.get("names")
    if isinstance(names_list, dict):
        names = {int(k): v for k, v in names_list.items()}
    else:
        names = {i: str(n) for i, n in enumerate(names_list)}

    # Derivar labels val
    val_labels_dir = find_val_labels_dir(cfg)

    # Carregar GT e predições
    gt_by_stem = load_gt_by_stem(Path(val_labels_dir), names)
    pred_by_stem = load_pred_by_stem(pred_dir)

    # Focar apenas classes no_*
    no_classes: Dict[int, str] = {cid: n for cid, n in names.items() if isinstance(n, str) and n.startswith("no_")}

    results: Dict[str, Dict[str, float]] = {}
    for cid, cname in no_classes.items():
        p_at_r, max_r = compute_p_at_r(gt_by_stem, pred_by_stem, cid, args.iou_thres)
        results[cname] = {
            "precision_at_recall_0_9": (None if (p_at_r != p_at_r) else float(p_at_r)),
            "max_recall": float(max_r),
        }

    # Salvar JSON de robustez
    with open(args.output_json, "w", encoding="utf-8") as fj:
        json.dump({"no_star_metrics": results}, fj, ensure_ascii=False, indent=2)

    # Métricas gerais simplificadas (placeholders): mAP/Recall/FNR para no_*
    # Estimar FNR(no_*) como 1 - max_recall médio nas classes no_*
    recalls = [v["max_recall"] for v in results.values() if isinstance(v.get("max_recall"), float)]
    avg_max_recall = sum(recalls) / len(recalls) if recalls else 0.0
    fnr_no_star = max(0.0, 1.0 - avg_max_recall)
    summary = {
        "fnr_no_star": fnr_no_star,
        "epi_unassociated_rate": 0.0,  # preenchido por assign_epi no pipeline completo
    }
    with open("data/stats/report_summary.json", "w", encoding="utf-8") as fs:
        json.dump(summary, fs, ensure_ascii=False, indent=2)

    # Anexar/gerar seção no REPORT.md
    section = [
        "\n\n## Robustez — Precision@Recall=0.9 (classes no_*)\n",
        "| classe | P@R=0.9 | recall_max |\n",
        "|---|---:|---:|\n",
    ]
    for cname, vals in results.items():
        pav = vals["precision_at_recall_0_9"]
        pav_str = "NA" if pav is None else f"{pav:.3f}"
        section.append(f"| {cname} | {pav_str} | {vals['max_recall']:.3f} |\n")

    # Append seguro
    try:
        with open(args.output_md, "a", encoding="utf-8") as fm:
            fm.writelines(section)
    except FileNotFoundError:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as fm:
            fm.writelines(["# REPORT\n"] + section)

    print(f"[OK] Robustez salva em {args.output_json} e seção adicionada em {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


