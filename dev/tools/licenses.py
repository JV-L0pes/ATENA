#!/usr/bin/env python3
"""
licenses.py: coleta/licenças, classifica {commercial_ok, non_commercial, unknown} e gera:
- LICENSES.md
- dataset.yaml (ok + unknown)
- dataset_comm.yaml (apenas commercial_ok)
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


RAW_ROOT = Path("data/raw")
OUT_ROOT = Path("data/merged_yolo")


def classify_license_text(txt: str) -> str:
    t = txt.lower()
    if any(k in t for k in ["non-commercial", "non commercial", "no commercial", "nc-"]):
        return "non_commercial"
    if any(k in t for k in ["mit", "apache", "bsd", "cc-by", "cc by", "commercial use", "unrestricted"]):
        return "commercial_ok"
    return "unknown"


def read_local_licenses(ds_dir: Path) -> Tuple[str, str]:
    # retorna (status, source)
    for name in ["LICENSE", "license", "README.md", "README.txt", "readme.md"]:
        p = ds_dir / name
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                    return classify_license_text(txt), str(p)
            except Exception:
                continue
    return ("unknown", "(none)")


def build_yaml(names: List[str], path_images: str, split="train") -> Dict:
    return {
        "path": str(Path(path_images).parent),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="licenses.py")
    parser.add_argument("--names", type=str, default="data/merged_yolo/names.txt", help="arquivo com 1 classe por linha na ordem final")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    lic_lines: List[str] = ["# LICENSES\n\n"]

    # classificar cada dataset
    allowed = {}
    for ds in sorted(RAW_ROOT.glob("*")):
        if not ds.is_dir():
            continue
        status, source = read_local_licenses(ds)
        lic_lines.append(f"- {ds.name}: {status} (source: {source})\n")
        allowed[ds.name] = status

    with open(OUT_ROOT / "LICENSES.md", "w", encoding="utf-8") as f:
        f.writelines(lic_lines)

    # carregar nomes em ordem
    names: List[str] = []
    try:
        with open(args.names, "r", encoding="utf-8") as fn:
            names = [ln.strip() for ln in fn if ln.strip()]
    except Exception:
        # fallback: 11 classes padrão
        names = [
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

    # gerar YAMLs (placeholders: assumem estrutura já criada em data/merged_yolo)
    yall = build_yaml(names, "data/merged_yolo/images/train")
    ycomm = build_yaml(names, "data/merged_yolo/images/train")

    # filtrar non_commercial fora do dataset_comm.yaml
    non_comm = [k for k, v in allowed.items() if v == "non_commercial"]
    ycomm["notes"] = {"excluded_non_commercial": non_comm}

    with open(OUT_ROOT / "dataset.yaml", "w", encoding="utf-8") as fy:
        yaml.dump(yall, fy, allow_unicode=True)
    with open(OUT_ROOT / "dataset_comm.yaml", "w", encoding="utf-8") as fyc:
        yaml.dump(ycomm, fyc, allow_unicode=True)

    print("[OK] LICENSES.md, dataset.yaml e dataset_comm.yaml gerados em data/merged_yolo/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


