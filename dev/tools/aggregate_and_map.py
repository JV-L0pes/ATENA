#!/usr/bin/env python3
"""
Aggregate + Map: agrega múltiplos datasets YOLO e aplica mapeamento de classes
para ontologia final (11 classes), ignorando rótulos fora da ontologia.

Regras:
- Detecta classes por data.yaml (names) ou labels/classes.txt
- Converte ids→nomes→alvos; ignora nomes não mapeados
- Copia imagens e escreve labels no destino unificado
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import yaml


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
FINAL_INDEX = {n: i for i, n in enumerate(FINAL_NAMES)}


def load_class_map(map_path: Path) -> Dict[str, str]:
    with open(map_path, "r", encoding="utf-8") as f:
        return json.load(f)


def derive_slug(ds_root: Path, global_root: Path) -> str:
    rel = ds_root.relative_to(global_root)
    return "-".join(rel.parts).replace(" ", "_")


def read_names_for_dataset(ds_root: Path) -> Optional[Dict[int, str]]:
    # Tenta data.yaml
    dy = ds_root / "data.yaml"
    if dy.exists():
        try:
            cfg = yaml.safe_load(dy.read_text(encoding="utf-8"))
            names = cfg.get("names")
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {i: str(n) for i, n in enumerate(names)}
        except Exception:
            pass
    # Tenta labels/classes.txt
    ct = ds_root / "labels" / "classes.txt"
    if ct.exists():
        try:
            lines = [ln.strip() for ln in ct.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return {i: lines[i] for i in range(len(lines))}
        except Exception:
            pass
    return None


def map_name_to_final(raw: str, class_map: Dict[str, str]) -> Optional[int]:
    key = raw.strip().lower().replace(" ", "_").replace("-", "_")
    target = class_map.get(key)
    if target is None:
        return None
    if target == "__ignore__":
        return None
    return FINAL_INDEX.get(target)


def process_dataset(ds_root: Path, global_root: Path, out_img: Path, out_lbl: Path, class_map: Dict[str, str]) -> int:
    names = read_names_for_dataset(ds_root)
    if not names:
        return 0
    slug = derive_slug(ds_root, global_root)
    copied = 0
    for split in ("train", "val", "valid", "test"):
        img_dir = ds_root / split / "images"
        lbl_dir = ds_root / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        for imgp in img_dir.rglob("*.*"):
            if imgp.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = imgp.stem
            lblp = lbl_dir / f"{stem}.txt"
            if not lblp.exists():
                continue
            # Ler e remapear labels
            new_lines: List[str] = []
            try:
                for ln in lblp.read_text(encoding="utf-8").splitlines():
                    ln = ln.strip()
                    if not ln:
                        continue
                    parts = ln.split()
                    if len(parts) < 5:
                        continue
                    cid = int(parts[0])
                    cx, cy, w, h = parts[1:5]
                    raw_name = names.get(cid, str(cid))
                    mapped_id = map_name_to_final(raw_name, class_map)
                    if mapped_id is None:
                        continue
                    new_lines.append(f"{mapped_id} {cx} {cy} {w} {h}")
            except Exception:
                continue
            if not new_lines:
                continue
            # copiar
            dst_img = out_img / f"{slug}__{stem}{imgp.suffix.lower()}"
            dst_lbl = out_lbl / f"{slug}__{stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(imgp, dst_img)
            dst_lbl.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            copied += 1
    return copied


def main() -> int:
    ap = argparse.ArgumentParser(description="aggregate_and_map")
    ap.add_argument("--root", type=str, default="datasets/datasets")
    ap.add_argument("--dst", type=str, default="data/clean_yolo")
    ap.add_argument("--class-map", type=str, default="data/mappings/class_map.json")
    args = ap.parse_args()

    global_root = Path(args.root)
    out = Path(args.dst)
    out_img = out / "images"
    out_lbl = out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    class_map = load_class_map(Path(args.class_map))

    total = 0
    # processar raiz e subpastas que tenham splits
    # primeiro, raiz
    total += process_dataset(global_root, global_root, out_img, out_lbl, class_map)
    for sub in global_root.rglob("*"):
        if not sub.is_dir():
            continue
        if (sub / "train").exists() or (sub / "val").exists() or (sub / "valid").exists() or (sub / "test").exists():
            total += process_dataset(sub, global_root, out_img, out_lbl, class_map)

    print(f"[OK] Aggregate+Map: {total} imagens rotuladas mapeadas para {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


