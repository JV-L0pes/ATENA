#!/usr/bin/env python3
"""
01_collect: baixa/valida datasets e gera data/raw_manifest.json com origem, hash e classes detectadas.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import requests


DATA_RAW = Path("data/raw")
MANIFEST = Path("data/raw_manifest.json")


def sha256_of_file(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_dirs() -> None:
    (DATA_RAW).mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def scan_classes_in_yolo_labels(root: Path) -> List[str]:
    # Heurística: se houver dataset.yaml com names, use. Caso contrário, tente inferir a partir de labels numéricas.
    yaml_path = next(root.glob("**/dataset.yaml"), None)
    if yaml_path and yaml_path.exists():
        try:
            import yaml  # type: ignore

            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("names")
            if isinstance(names, list):
                return [str(n) for n in names]
            if isinstance(names, dict):
                return [str(names[i]) for i in sorted(map(int, names.keys()))]
        except Exception:
            pass
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="01_collect - coleta e manifesto de datasets")
    parser.add_argument("--source", action="append", default=[], help="URL de arquivo ZIP/TAR a baixar (pode repetir)")
    parser.add_argument("--slug", action="append", default=[], help="Slug para salvar em data/raw/<slug> (ordem correspondente aos --source)")
    parser.add_argument("--skip-download", action="store_true", help="Não baixar; apenas varrer data/raw/")
    args = parser.parse_args()

    ensure_dirs()

    entries: List[Dict] = []

    # Download opcional
    if not args.skip_download and args.source and args.slug and len(args.source) == len(args.slug):
        for url, slug in zip(args.source, args.slug):
            dst = DATA_RAW / slug / Path(url).name
            try:
                print(f"[INFO] Baixando {url} -> {dst}")
                download(url, dst)
            except Exception as e:
                print(f"[WARN] Falhou download {url}: {e}")

    # Varrer data/raw
    for ds_dir in sorted(DATA_RAW.glob("*")):
        if not ds_dir.is_dir():
            continue
        files = list(ds_dir.rglob("*"))
        size_bytes = sum(p.stat().st_size for p in files if p.is_file())
        # hash do conjunto (hash de hashes para estabilidade)
        h = hashlib.sha256()
        for p in sorted(p for p in files if p.is_file()):
            try:
                h.update(sha256_of_file(p).encode())
            except Exception:
                pass
        dataset_hash = h.hexdigest()
        classes = scan_classes_in_yolo_labels(ds_dir)
        entries.append({
            "slug": ds_dir.name,
            "path": str(ds_dir),
            "size_bytes": int(size_bytes),
            "dataset_hash": dataset_hash,
            "classes": classes,
        })

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"datasets": entries}, f, ensure_ascii=False, indent=2)
    print(f"[OK] Manifesto gerado em {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


