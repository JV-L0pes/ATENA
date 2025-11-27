#!/usr/bin/env python3
"""
02_convert_to_yolo: converte COCO/VOC/Roboflow para YOLOv11; erros em data/failed/*.log
Implementação básica com placeholders; detecta caso já estiver em YOLO.
"""

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="02_convert_to_yolo")
    parser.add_argument("--input", type=str, default="data/raw", help="Raiz de entrada")
    parser.add_argument("--output", type=str, default="data/interim_yolo", help="Saída YOLO")
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    # Placeholder: copiar estrutura se já for YOLO (images/labels)
    for ds in sorted(in_root.glob("*")):
        if not ds.is_dir():
            continue
        images = list(ds.rglob("images"))
        labels = list(ds.rglob("labels"))
        if images and labels:
            # Apenas registrar que já está em YOLO
            print(f"[INFO] Dataset {ds.name} parece já estar em formato YOLO (images/labels)")
        else:
            print(f"[WARN] Conversão não implementada para {ds.name} (placeholder)")
    print("[OK] 02_convert_to_yolo concluído (placeholder)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


