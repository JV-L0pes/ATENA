#!/usr/bin/env python3
"""
Inferência com thresholds por classe/condição, TTA e opções de tiling/WBF.
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="infer")
    parser.add_argument("--model", type=str, default="runs/ppe/y11x_merged/weights/best.pt")
    parser.add_argument("--thresholds", type=str, default="inference_thresholds_day.json")
    parser.add_argument("--images", type=str, required=False)
    parser.add_argument("--tile", nargs=2, type=float, default=None, help="tile_size overlap")
    parser.add_argument("--tile-classes", type=str, default="ear_protection,no_earprotection")
    parser.add_argument("--wbf", type=float, default=None, help="WBF IoU threshold (opcional)")
    args = parser.parse_args()

    # Placeholder: apenas valida entradas e encerra
    if not Path(args.model).exists():
        print(f"[ERRO] Modelo não encontrado: {args.model}")
        return 1
    if Path(args.thresholds).exists():
        with open(args.thresholds, "r", encoding="utf-8") as f:
            _ = json.load(f)
    print("[OK] infer (placeholder) pronto para integrar com pipeline Ultralytics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


