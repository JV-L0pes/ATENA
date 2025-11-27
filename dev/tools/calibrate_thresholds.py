#!/usr/bin/env python3
"""
Calibração de thresholds por classe e por condição (day/night/backlight) com heurísticas simples.
Exporta inference_thresholds_{day,night,backlight}.json.
"""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="calibrate_thresholds")
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    # Placeholder: thresholds fixos (serão calibrados em versão completa)
    base = {
        "person": 0.10,
        "helmet": 0.40,
        "goggles": 0.35,
        "gloves": 0.30,
        "boots": 0.30,
        "ear_protection": 0.35,
        "no_helmet": 0.50,
        "no_goggles": 0.50,
        "no_gloves": 0.50,
        "no_boots": 0.50,
        "no_earprotection": 0.50,
    }
    for cond in ("day", "night", "backlight"):
        with open(Path(args.out_dir) / f"inference_thresholds_{cond}.json", "w", encoding="utf-8") as f:
            json.dump(base, f, ensure_ascii=False, indent=2)
    print("[OK] thresholds (placeholder) salvos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


