#!/usr/bin/env python3
import sys
import yaml


EXPECTED = [
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


def main() -> int:
    paths = [
        "data/merged_yolo/dataset.yaml",
        "data/merged_yolo/dataset_comm.yaml",
    ]
    ok = True
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                y = yaml.safe_load(f)
            names = y.get("names")
            if isinstance(names, dict):
                names = [names[i] for i in range(len(names))]
            if names != EXPECTED:
                print(f"[FAIL] {p}: names != EXPECTED")
                ok = False
        except Exception as e:
            print(f"[FAIL] {p}: {e}")
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


