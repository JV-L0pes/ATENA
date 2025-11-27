#!/usr/bin/env python3
"""
04_dedup_and_split: deduplicação por pHash e split por pseudo_câmera (clusters pHash).

Flags chave:
--split-by-pseudo-camera --pseudo-camera-min-size 50 --val-pseudo-cameras 2 --test-pseudo-cameras 1
"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys

from PIL import Image
import imagehash


def hamming(a, b) -> int:
    return a - b if isinstance(a, imagehash.ImageHash) else int(a) - int(b)


def compute_phash(p: Path) -> imagehash.ImageHash:
    with Image.open(p) as im:
        return imagehash.phash(im, hash_size=64)


def main() -> int:
    parser = argparse.ArgumentParser(description="04_dedup_and_split")
    parser.add_argument("--images", type=str, default="data/clean_yolo/images", help="Pasta de imagens")
    parser.add_argument("--labels", type=str, default="data/clean_yolo/labels", help="Pasta de labels")
    parser.add_argument("--output", type=str, default="data/merged_yolo", help="Saída YOLO final")
    parser.add_argument("--split-by-pseudo-camera", action="store_true", help="Ativar split por pseudo_câmera")
    parser.add_argument("--pseudo-camera-min-size", type=int, default=50)
    parser.add_argument("--val-pseudo-cameras", type=int, default=2)
    parser.add_argument("--test-pseudo-cameras", type=int, default=1)
    parser.add_argument("--hamming-radius", type=int, default=10, help="Raio de Hamming para clusterização (default 10)")
    args = parser.parse_args()

    img_root = Path(args.images)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    stems: List[str] = []
    for p in img_root.rglob("*.jpg"):
        stems.append(p.stem)
    for p in img_root.rglob("*.png"):
        stems.append(p.stem)
    stems = sorted(set(stems))
    # Agrupar por slug (prefixo do nome até "__")
    slug_to_stems: Dict[str, List[str]] = {}
    for s in stems:
        slug = s.split("__", 1)[0] if "__" in s else "merged"
        slug_to_stems.setdefault(slug, []).append(s)

    # clusterização simples por pHash (raio <= args.hamming_radius) com cache
    stats_dir = Path("data/stats")
    stats_dir.mkdir(parents=True, exist_ok=True)
    cache_path = stats_dir / "phash_cache.json"
    cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    hashes: Dict[str, imagehash.ImageHash] = {}
    total = len(stems)
    for idx, s in enumerate(stems, 1):
        ip = next((q for q in (img_root.rglob(f"**/{s}.jpg")) ), None)
        if ip is None:
            ip = next((q for q in (img_root.rglob(f"**/{s}.png")) ), None)
        if ip is None:
            continue
        try:
            cached = cache.get(s)
            if cached is not None:
                hashes[s] = imagehash.hex_to_hash(cached)
            else:
                hv = compute_phash(ip)
                hashes[s] = hv
                cache[s] = hv.__str__()
            if idx % 500 == 0:
                # progresso
                print(f"[pHash] {idx}/{total}")
                sys.stdout.flush()
                try:
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")
                except Exception:
                    pass
        except Exception:
            continue
    # flush final do cache
    try:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass

    # Clusterizar POR SLUG
    per_slug_clusters: Dict[str, List[List[str]]] = {}
    for slug, lst in slug_to_stems.items():
        cl: List[List[str]] = []
        for s in lst:
            hv = hashes.get(s)
            if hv is None:
                continue
            placed = False
            for c in cl:
                ref = c[0]
                if (hashes.get(ref) - hv) <= int(getattr(args, "hamming_radius", 10)):
                    c.append(s)
                    placed = True
                    break
            if not placed:
                cl.append([s])
        per_slug_clusters[slug] = cl

    # montar pseudo_camera_id e split básico
    dataset_slug = "merged"
    pseudo_map: Dict[str, str] = {}
    test: set = set()
    val: set = set()
    train: set = set()
    # Seleção por slug: 1 cluster por slug -> test; 1–2 clusters por slug -> val; restantes -> train
    for slug, clusters in per_slug_clusters.items():
        # ordenar clusters por tamanho decrescente
        clusters = sorted(clusters, key=lambda c: len(c), reverse=True)
        # filtrar por tamanho mínimo
        clusters = [c for c in clusters if len(c) >= args.__dict__["pseudo_camera_min_size"]]
        if not clusters:
            continue
        # test
        pcid_test = f"{dataset_slug}__{slug}__cluster0"
        for s in clusters[0]:
            pseudo_map[s] = pcid_test
        test.add(pcid_test)
        # val (próximos até N)
        for i in range(1, 1 + args.val_pseudo_cameras):
            if i >= len(clusters):
                break
            pcid_val = f"{dataset_slug}__{slug}__cluster{i}"
            for s in clusters[i]:
                pseudo_map[s] = pcid_val
            val.add(pcid_val)
        # train (restante)
        for i in range(1 + args.val_pseudo_cameras, len(clusters)):
            pcid_tr = f"{dataset_slug}__{slug}__cluster{i}"
            for s in clusters[i]:
                pseudo_map[s] = pcid_tr
            train.add(pcid_tr)

    # Se ainda vazio (caso extremo), jogar tudo em train
    if not pseudo_map:
        for s in stems:
            pseudo_map[s] = f"{dataset_slug}__train_all"
        train.add(f"{dataset_slug}__train_all")
    
    # salvar split index
    (out_root / "splits").mkdir(parents=True, exist_ok=True)
    with open(out_root / "splits" / "split_pseudo_camera.txt", "w", encoding="utf-8") as f:
        for s, pc in sorted(pseudo_map.items()):
            tag = "train" if pc in train else ("val" if pc in val else ("test" if pc in test else "ignore"))
            f.write(f"{s},{pc},{tag}\n")

    # materializar estrutura YOLO: copiar arquivos para merged_yolo/{images,labels}/{train,val,test}
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        # limpar conteúdo anterior para evitar resíduos de execuções prévias
        for p in (out_root / "images" / split).glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        for p in (out_root / "labels" / split).glob("*.txt"):
            try:
                p.unlink()
            except Exception:
                pass

    copied = {"train": 0, "val": 0, "test": 0}
    for stem, pc in pseudo_map.items():
        split = "train" if pc in train else ("val" if pc in val else ("test" if pc in test else None))
        if split is None:
            continue
        # localizar imagem e label no conjunto limpo
        imgp = next((q for q in img_root.rglob(f"**/{stem}.jpg")), None)
        if imgp is None:
            imgp = next((q for q in img_root.rglob(f"**/{stem}.png")), None)
        if imgp is None:
            continue
        lblp = Path(str(imgp).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lblp.exists():
            # tentar labels raiz
            lblp = (Path(args.labels) / f"{stem}.txt")
            if not lblp.exists():
                continue
        dst_img = out_root / "images" / split / imgp.name
        dst_lbl = out_root / "labels" / split / f"{stem}.txt"
        try:
            dst_img.write_bytes(imgp.read_bytes())
            dst_lbl.write_text(lblp.read_text(encoding="utf-8"), encoding="utf-8")
            copied[split] += 1
        except Exception:
            continue

    print(f"[OK] 04_dedup_and_split: train={copied['train']} val={copied['val']} test={copied['test']}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


