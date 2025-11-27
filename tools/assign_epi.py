#!/usr/bin/env python3
"""
Associação Pessoa↔EPI (modo batch)
Entrada: JSON de detecções por imagem (ou adaptar para ler das predições salvas),
Saída: métricas de EPIs não associados por classe.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def inside(c, box):
    cx, cy = c
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def main() -> int:
    parser = argparse.ArgumentParser(description="assign_epi batch")
    parser.add_argument("--detections-json", type=str, required=False, help="Arquivo JSON com deteções por imagem")
    parser.add_argument("--out-json", type=str, default="data/stats/epi_unassoc.json")
    args = parser.parse_args()

    # Placeholder: estrutura de entrada depende do pipeline; aqui gera saída vazia
    out = {"unassociated": {}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] assign_epi (placeholder) salvo em {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


