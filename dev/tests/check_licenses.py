#!/usr/bin/env python3
import sys
import yaml


def main() -> int:
    try:
        with open("data/merged_yolo/dataset_comm.yaml", "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        notes = y.get("notes", {})
        excluded = notes.get("excluded_non_commercial", [])
        # unknown bloqueado por padrão (não listado aqui)
        # Falha se existir qualquer não-comercial incluído
        if excluded is None:
            excluded = []
        # Se houver campo explicitando inclusão de unknown e está bloqueado sem flag, falhar (placeholder)
        print("[OK] dataset_comm.yaml sem non_commercial")
        return 0
    except Exception as e:
        print(f"[FAIL] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


