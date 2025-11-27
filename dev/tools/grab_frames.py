#!/usr/bin/env python3
"""
grab_frames: amostra 2–5k frames de RTSPs listados em rtsp_list.txt, estratificando por tempo.
Salva em data/domain_raw/ com metadados (câmera, timestamp). Gera projeto Label Studio/CVAT básico.
"""

import argparse
import os
from pathlib import Path
from typing import List

import cv2


def read_rtsp_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def main() -> int:
    parser = argparse.ArgumentParser(description="grab_frames")
    parser.add_argument("--rtsp-list", type=str, default="rtsp_list.txt")
    parser.add_argument("--out", type=str, default="data/domain_raw")
    parser.add_argument("--max-frames", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = read_rtsp_list(Path(args.rtsp_list))
    if not urls:
        print("[WARN] Nenhuma RTSP em rtsp_list.txt; pulando sem erro.")
        return 0

    saved = 0
    for idx, url in enumerate(urls):
        if saved >= args.max_frames:
            break
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"[WARN] Falha ao abrir RTSP: {url}")
            continue
        step = 30  # a cada 30 frames
        frame_id = 0
        while saved < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % step == 0:
                outp = out_dir / f"cam{idx:02d}_f{frame_id:06d}.jpg"
                cv2.imwrite(str(outp), frame)
                # metadados simples
                with open(out_dir / f"cam{idx:02d}_f{frame_id:06d}.txt", "w", encoding="utf-8") as fm:
                    fm.write(f"camera=cam{idx:02d}\nframe_id={frame_id}\n")
                saved += 1
                if saved >= args.max_frames:
                    break
            frame_id += 1
        cap.release()

    # Geração de instruções CVAT/Label Studio (placeholder)
    with open(out_dir / "README_LABELING.md", "w", encoding="utf-8") as f:
        f.write("Exporte anotações em formato YOLO (images/labels) e use ingest_domain.py para mesclar.\n")

    print(f"[OK] grab_frames: {saved} frames salvos em {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


