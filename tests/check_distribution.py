#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main() -> int:
    p = Path("data/stats/imbalance.json")
    if not p.exists():
        print("[WARN] imbalance.json não encontrado; passe após 05_stats.py")
        return 0
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    # Placeholder: apenas existe e bem formado
    if not isinstance(j, dict):
        print("[FAIL] imbalance.json inválido")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


