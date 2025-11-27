#!/usr/bin/env python3
"""
Treinador YOLOv11x com diretórios fixos, early-stop guiado por FNR e pós-validação
para hard negatives. Inclui FT condicional e extensão opcional de treino.

Regras principais:
- Modelo fixo: YOLOv11x
- Diretórios fixos: project=runs/ppe, name=y11x_merged
- Early-stop (janela 20): parar se ΔFNR(no_*) < 0,3 pp e ΔmAP@50–95 < 0,2
- Extensão +25 épocas se ΔFNR(no_*) > 0,3 pp no fim do Estágio 1
- FT condicional 50 épocas se FNR(no_*) ≥ 10% ou aceite falhar
- Após treino: gerar predições do split=val com save_txt/save_conf para hard-negative mining
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def set_reproducibility_seeds(seed: int = 42) -> None:
    """Configura seeds e flags determinísticas."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        import torch  # type: ignore

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        import torch.backends.cudnn as cudnn  # type: ignore

        cudnn.deterministic = True
        cudnn.benchmark = False
    except Exception:
        pass


def ensure_env() -> None:
    """Configura variáveis de ambiente úteis para evitar fragmentação de VRAM."""
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:256",
    )


def _read_fnr_summary(summary_path: Path) -> Optional[float]:
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        v = js.get("fnr_no_star")
        return float(v) if v is not None else None
    except Exception:
        return None


def _compute_window_deltas(prev_fnr: Optional[float], curr_fnr: Optional[float], prev_map: Optional[float], curr_map: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    d_fnr = None
    d_map = None
    if prev_fnr is not None and curr_fnr is not None:
        d_fnr = prev_fnr - curr_fnr  # FNR menor é melhor (delta positivo = melhoria)
    if prev_map is not None and curr_map is not None:
        d_map = curr_map - prev_map
    return d_fnr, d_map


def main(
    data_yaml: str = "data/merged_yolo/dataset_comm.yaml",
    model_name: str = "yolov11x.pt",
    device: str = "0",
    total_epochs: int = 200,
    window_epochs: int = 20,
) -> int:
    set_reproducibility_seeds(42)
    ensure_env()

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        print(f"[ERRO] Ultralytics não disponível: {e}")
        return 1

    project = Path("runs/ppe")
    name = "y11x_merged"
    project.mkdir(parents=True, exist_ok=True)

    # Parâmetros de treino conforme plano aprovado
    train_params = dict(
        data=str(data_yaml),
        imgsz=1280,
        epochs=window_epochs,
        batch="auto",
        device=device,
        workers=16,
        seed=42,
        patience=50,
        cache="ram",
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.2,
        close_mosaic=20,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.08,
        scale=0.8,
        shear=2,
        fliplr=0.5,
        optimizer="adamw",
        cos_lr=True,
        pretrained=True,
        multi_scale=True,
        fl_gamma=1.5,
        ema=True,
        save_period=25,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.0005,
        cls=0.7,
        save_json=True,
        csv=True,
        plots=True,
        project=str(project),
        name=name,
        exist_ok=True,
    )

    print("[INFO] Iniciando treino YOLOv11x...")
    model = YOLO(model_name)

    # Treino em janelas com possível early-stop
    epochs_done = 0
    prev_fnr = None
    prev_map = None
    last_results = None
    while epochs_done < total_epochs:
        # Ajustar épocas da janela (último bloco pode ser menor)
        remaining = total_epochs - epochs_done
        train_params["epochs"] = min(window_epochs, remaining)
        # Resume se já treinou algo
        train_params["resume"] = True if epochs_done > 0 else False
        try:
            last_results = model.train(**train_params)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("[WARN] OOM detectado: fallback imgsz=1024...")
                train_params["imgsz"] = 1024
                last_results = model.train(**train_params)
            else:
                raise

        epochs_done += train_params["epochs"]

        # Validar e coletar métricas (mAP) e FNR (via report)
        print("[INFO] Validação pós-janela para cálculo de FNR e mAP...")
        vres = model.val(data=str(data_yaml))
        curr_map = None
        try:
            curr_map = float(getattr(vres.box, "map", None))
        except Exception:
            curr_map = None

        # gerar predições do val + relatório para FNR
        try:
            _ = model.val(
                data=str(data_yaml),
                split="val",
                save_txt=True,
                save_conf=True,
                project=str(project),
                name=f"{name}_val",
                exist_ok=True,
            )
            # rodar report para obter fnr
            os.system(
                f"python tools/report.py --pred_dir runs/ppe/{name}_val/labels --data_yaml {data_yaml} >/dev/null 2>&1"
            )
            curr_fnr = _read_fnr_summary(Path("data/stats/report_summary.json"))
        except Exception:
            curr_fnr = None

        # Early-stop: ΔFNR < 0.003 e ΔmAP < 0.002 na janela
        d_fnr, d_map = _compute_window_deltas(prev_fnr, curr_fnr, prev_map, curr_map)
        print(f"[INFO] Epochs={epochs_done}/{total_epochs} FNR={curr_fnr} ΔFNR={d_fnr} mAP={curr_map} ΔmAP={d_map}")
        if prev_fnr is not None and prev_map is not None and d_fnr is not None and d_map is not None:
            if d_fnr < 0.003 and d_map < 0.002:
                print("[INFO] Early-stop acionado por baixa melhoria em FNR e mAP.")
                break

        prev_fnr = curr_fnr
        prev_map = curr_map

    # Caminhos resultantes
    save_dir: Optional[Path] = None
    try:
        # results.save_dir é geralmente um pathlib.Path
        save_dir = Path(getattr(results, "save_dir", project / name))
    except Exception:
        save_dir = project / name

    # Pós-treino: gerar predições do split=val com save_txt/save_conf (hard negatives)
    try:
        print("[INFO] Gerando predições do split=val (save_txt/save_conf)...")
        model = YOLO(str((save_dir / "weights" / "best.pt")))
        _ = model.val(
            data=str(data_yaml),
            split="val",
            save_txt=True,
            save_conf=True,
            project=str(project),
            name=f"{name}_val",
            exist_ok=True,
        )
        print("[OK] Predições de validação salvas em runs/ppe/y11x_merged_val")
    except Exception as e:
        print(f"[WARN] Falhou gerar predições do val: {e}")

    # Snapshot de hparams/env (melhor esforço)
    try:
        (project / name).mkdir(parents=True, exist_ok=True)
        os.system(
            f"yolo settings show > {project / name / 'hparams.txt'} 2>/dev/null || true"
        )
        os.system(f"pip freeze > {project / name / 'env.txt'} 2>/dev/null || true")
        # git commit opcional
        os.system(
            f"git rev-parse HEAD > {project / name / 'git_commit.txt'} 2>/dev/null || true"
        )
    except Exception:
        pass

    print("[OK] Treino finalizado.")

    # Extensão opcional: se última janela teve ΔFNR > 0.003, estender +25 epochs
    # (se não atingiu total_epochs)
    if epochs_done < total_epochs and d_fnr is not None and d_fnr > 0.003:
        print("[INFO] Melhoria relevante em FNR detectada. Estendendo +25 épocas...")
        try:
            train_params["epochs"] = 25
            train_params["resume"] = True
            _ = model.train(**train_params)
        except Exception as e:
            print(f"[WARN] Falha ao estender treino: {e}")

    # FT condicional: se FNR(no_*) >= 10% ou aceite falhar, rodar 50 épocas com hyper de FT
    fnr_val = _read_fnr_summary(Path("data/stats/report_summary.json"))
    need_ft = (fnr_val is not None and fnr_val >= 0.10)
    if need_ft:
        print(f"[INFO] FNR(no_*)={fnr_val:.3f} >= 0.10 → iniciando Fine-Tune 50 épocas...")
        try:
            ft_model = YOLO(str((save_dir / "weights" / "best.pt")))
            _ = ft_model.train(
                data=str(data_yaml),
                epochs=50,
                imgsz=train_params.get("imgsz", 1280),
                batch=train_params.get("batch", "auto"),
                device=train_params.get("device", device),
                workers=train_params.get("workers", 16),
                seed=42,
                patience=30,
                project=str(project),
                name=name,
                exist_ok=True,
                lr0=0.002,
                lrf=0.02,
                copy_paste=0.0,
                pretrained=True,
                cos_lr=True,
                optimizer="adamw",
            )
        except Exception as e:
            print(f"[WARN] Fine-Tune falhou: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


