#!/usr/bin/env python3
"""
ATHENA - Orquestrador de Treino e Deploy
=======================================
1) Gera YAML SH17 verdadeiro
2) Audita dataset
3) Treina Fase 1 (YOLO11m, épocas configuradas no trainer - 500)
4) (Opcional) Treina Fase 2
5) Valida e gera relatórios
6) Copia modelo final para athena_model_latest.pt para o backend carregar automaticamente
"""

import logging
import shutil
import json
from datetime import datetime
from pathlib import Path
import argparse

# Importar treinador existente
from scripts.athena_train_2phase_optimized import AthenaEPITrainer2PhaseOptimized


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("athena_train_and_deploy")


def copy_for_deploy(model_path: Path, label: str) -> Path:
    """Copia o modelo para um caminho canônico usado pelo backend."""
    if not model_path or not model_path.exists():
        raise FileNotFoundError(f"Modelo {label} não encontrado: {model_path}")

    deploy_path = Path("athena_model_latest.pt")
    shutil.copy2(model_path, deploy_path)
    logger.info(f"✅ Modelo {label} copiado para: {deploy_path}")

    # Salvar metadados de deploy
    metadata = {
        "label": label,
        "source_path": str(model_path),
        "deploy_path": str(deploy_path),
        "timestamp": datetime.now().isoformat()
    }
    with open("athena_model_deploy.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("📄 Metadados de deploy salvos em athena_model_deploy.json")

    return deploy_path


def main():
    parser = argparse.ArgumentParser(description="ATHENA - Orquestrador de Treino e Deploy")
    parser.add_argument("--phase", type=int, choices=[1, 2, 12], default=12,
                        help="1=apenas Fase1, 2=apenas Fase2, 12=ambas (padrão)")
    parser.add_argument("--skip_audit", action="store_true", help="Pula auditoria do dataset")
    args = parser.parse_args()

    trainer = AthenaEPITrainer2PhaseOptimized()

    # Fase 1
    if args.phase in (1, 12):
        logger.info("\n🚀 FASE 1: Preparando dataset e auditoria...")
        dataset_yaml_phase1 = trainer.create_dataset_config_phase1()
        if not dataset_yaml_phase1:
            logger.error("Falha ao criar YAML Fase 1")
            return 1

        if not args.skip_audit:
            ok_soft = trainer.audit_dataset(dataset_yaml_phase1)
            ok_strict = trainer.audit_dataset_strict(dataset_yaml_phase1, sample_n=200, fail_fast=True)
            if not ok_soft or not ok_strict:
                logger.error("❌ Auditoria falhou. Abortando treino.")
                return 1

        logger.info("\n🎯 FASE 1: Treinando modelo base...")
        phase1_model = trainer.train_phase1_complete(dataset_yaml_phase1)
        if not phase1_model:
            logger.error("Falha no treinamento da Fase 1")
            return 1

        logger.info("\n📊 FASE 1: Validação...")
        trainer.validate_model(phase1_model, "FASE 1")

        # Copiar para deploy (modelo base)
        try:
            copy_for_deploy(phase1_model, label="phase1_base")
        except Exception as e:
            logger.warning(f"Não foi possível preparar deploy da Fase 1: {e}")

    # Fase 2
    if args.phase in (2, 12):
        logger.info("\n🚀 FASE 2: Preparando dataset (mesmas classes SH17)...")
        dataset_yaml_phase2 = trainer.create_dataset_config_phase2()
        if not dataset_yaml_phase2:
            logger.error("Falha ao criar YAML Fase 2")
            return 1

        base_model_path = Path("athena_training_2phase_optimized") / "models" / "phase1_complete" / "athena_phase1_tesla_t4" / "weights" / "best.pt"
        if not base_model_path.exists():
            logger.error(f"Modelo base da Fase 1 não encontrado: {base_model_path}")
            return 1

        logger.info("\n🎯 FASE 2: Fine-tuning...")
        phase2_model = trainer.train_phase2_finetuning(str(base_model_path), dataset_yaml_phase2)
        if not phase2_model:
            logger.error("Falha no treinamento da Fase 2")
            return 1

        logger.info("\n📊 FASE 2: Validação...")
        trainer.validate_model(phase2_model, "FASE 2")

        # Copiar para deploy (modelo final)
        copy_for_deploy(phase2_model, label="phase2_final")

    logger.info("\n🎉 Treino e deploy concluídos. Reinicie o backend para carregar o novo modelo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


