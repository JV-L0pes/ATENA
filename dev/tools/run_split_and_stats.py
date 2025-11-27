#!/usr/bin/env python3
"""
Orquestra o split por pseudo_câmera e as estatísticas, evitando múltiplas chamadas de terminal.
Executa os scripts 04_dedup_and_split.py e 05_stats.py no mesmo processo e imprime um resumo.
"""

import json
import sys
from pathlib import Path
from importlib.machinery import SourceFileLoader


ROOT = Path(__file__).resolve().parents[1]


def run_split(images: Path, labels: Path, output: Path) -> None:
    mod_path = ROOT / "tools" / "04_dedup_and_split.py"
    mod = SourceFileLoader("dedup_split", str(mod_path)).load_module()
    sys.argv = [str(mod_path),
                "--images", str(images),
                "--labels", str(labels),
                "--output", str(output),
                "--split-by-pseudo-camera",
                "--pseudo-camera-min-size", "50",
                "--val-pseudo-cameras", "2",
                "--test-pseudo-cameras", "1"]
    mod.main()


def run_stats(dataset_yaml: Path, root: Path) -> None:
    mod_path = ROOT / "tools" / "05_stats.py"
    mod = SourceFileLoader("stats_mod", str(mod_path)).load_module()
    sys.argv = [str(mod_path),
                "--dataset-yaml", str(dataset_yaml),
                "--root", str(root)]
    mod.main()


def count_files(base: Path) -> dict:
    out = {}
    for split in ("train", "val", "test"):
        imgs = list((base / "images" / split).glob("*.*"))
        lbls = list((base / "labels" / split).glob("*.txt"))
        out[split] = {"images": len(imgs), "labels": len(lbls)}
    return out


def main() -> int:
    images = ROOT / "data" / "clean_yolo" / "images"
    labels = ROOT / "data" / "clean_yolo" / "labels"
    merged = ROOT / "data" / "merged_yolo"

    run_split(images, labels, merged)
    counts = count_files(merged)
    print("COUNTS:", json.dumps(counts))

    # dataset.yaml pode ter sido gerado por licenses.py; se não existir, usar o merged root mesmo
    dataset_yaml = merged / "dataset.yaml"
    if not dataset_yaml.exists():
        # criar mínimo para o stats rodar (names fictícios)
        names = ["person","helmet","goggles","gloves","boots","ear_protection","no_helmet","no_goggles","no_gloves","no_boots","no_earprotection"]
        dataset_yaml.write_text(json.dumps({"path": str(merged), "train": "images/train", "val": "images/val", "test": "images/test", "nc": 11, "names": names}), encoding="utf-8")

    run_stats(dataset_yaml, merged)
    leak_path = ROOT / "data" / "stats" / "leakage.json"
    if leak_path.exists():
        print("LEAKAGE:", leak_path.read_text(encoding="utf-8"))
    else:
        print("LEAKAGE: n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


