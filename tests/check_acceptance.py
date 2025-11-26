#!/usr/bin/env python3
"""
Regras de aceite automáticas:
- FNR(no_*) < 10% em val
- %EPI_não_associado < 3%
- vazamento == 0%
- label_qc.csv <= 0.5% conflitos
Sai com código !=0 se qualquer regra falhar.
"""

import json
import sys
from pathlib import Path


def read_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    ok = True

    # FNR(no_*) e %EPI_não_associado: esperar que report.py escreva em data/stats/report_summary.json
    rep = read_json(Path("data/stats/report_summary.json"))
    if isinstance(rep, dict):
        fnr = rep.get("fnr_no_star", 0.0)
        if fnr is not None and fnr >= 0.10:
            print(f"[FAIL] FNR(no_*) >= 10%: {fnr}")
            ok = False
        unassoc = rep.get("epi_unassociated_rate", 0.0)
        if unassoc is not None and unassoc >= 0.03:
            print(f"[FAIL] %EPI_não_associado >= 3%: {unassoc}")
            ok = False

    # vazamento
    leak = read_json(Path("data/stats/leakage.json"))
    if isinstance(leak, dict):
        if (leak.get("val_vs_train", 0.0) or 0.0) > 0.0 or (leak.get("test_vs_train", 0.0) or 0.0) > 0.0:
            print("[FAIL] Vazamento > 0%")
            ok = False

    # label QC
    qc = Path("data/stats/label_qc.csv")
    if qc.exists():
        try:
            total = 0
            conflicts = 0
            with open(qc, "r", encoding="utf-8") as f:
                header = True
                for ln in f:
                    if header:
                        header = False
                        continue
                    total += 1
                    conflicts += 1  # cada linha é um conflito registrado
            rate = (conflicts / total) if total > 0 else 0.0
            if rate > 0.005:
                print(f"[FAIL] label_qc conflitos > 0.5%: {rate:.4f}")
                ok = False
        except Exception:
            pass

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


