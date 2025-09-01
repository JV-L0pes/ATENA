"""
Módulo de Snapshot
Sistema de captura e gerenciamento de snapshots
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from .config import CONFIG
from .utils import save_frame_as_image, load_image_as_frame

logger = logging.getLogger(__name__)

class SnapshotManager:
    """Gerenciador de snapshots"""
    
    def __init__(self):
        self.snapshot_dir = CONFIG.SNAPSHOT_DIR
        self.snapshot_format = CONFIG.SNAPSHOT_FORMAT
        self.snapshot_quality = CONFIG.SNAPSHOT_QUALITY
        
        # Criar diretório se não existir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache de snapshots
        self.snapshots = []
        self.max_snapshots = 100
        
        # Carregar snapshots existentes
        self._load_existing_snapshots()
        
        logger.info(f"Snapshot Manager inicializado em {self.snapshot_dir}")
    
    def _load_existing_snapshots(self):
        """Carrega snapshots existentes do diretório"""
        try:
            pattern = f"*.{self.snapshot_format}"
            snapshot_files = list(self.snapshot_dir.glob(pattern))
            
            for file_path in snapshot_files:
                # Extrair informações do nome do arquivo
                info = self._parse_filename(file_path.name)
                if info:
                    self.snapshots.append(info)
            
            # Ordenar por timestamp
            self.snapshots.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Limitar número de snapshots em memória
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[:self.max_snapshots]
            
            logger.info(f"Carregados {len(self.snapshots)} snapshots existentes")
            
        except Exception as e:
            logger.error(f"Erro ao carregar snapshots existentes: {e}")
    
    def _parse_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Extrai informações do nome do arquivo"""
        try:
            # Formato esperado: YYYY-MM-DD_HH-MM-SS_frameID_epi_summary.ext
            parts = filename.replace(f".{self.snapshot_format}", "").split("_")
            
            if len(parts) >= 3:
                date_str = parts[0]
                time_str = parts[1]
                frame_id = int(parts[2])
                
                # Combinar data e hora
                datetime_str = f"{date_str} {time_str.replace('-', ':')}"
                timestamp = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
                
                # Extrair resumo de EPI se disponível
                epi_summary = {}
                if len(parts) > 3:
                    try:
                        epi_summary = json.loads(parts[3])
                    except:
                        pass
                
                return {
                    "filename": filename,
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "epi_summary": epi_summary,
                    "filepath": self.snapshot_dir / filename
                }
            
        except Exception as e:
            logger.debug(f"Erro ao parsear nome do arquivo {filename}: {e}")
        
        return None
    
    def _generate_filename(self, frame_id: int, epi_summary: Dict[str, int] = None) -> str:
        """Gera nome único para o snapshot"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if epi_summary:
            epi_str = json.dumps(epi_summary).replace('"', '').replace(':', '-')
            return f"{timestamp}_{frame_id}_{epi_str}.{self.snapshot_format}"
        else:
            return f"{timestamp}_{frame_id}.{self.snapshot_format}"
    
    def capture_snapshot(self, frame: np.ndarray, frame_id: int, epi_summary: Dict[str, int] = None) -> Optional[Path]:
        """Captura snapshot do frame atual"""
        try:
            if frame is None:
                logger.warning("Frame é None, não é possível capturar snapshot")
                return None
            
            # Gerar nome do arquivo
            filename = self._generate_filename(frame_id, epi_summary)
            filepath = self.snapshot_dir / filename
            
            # Salvar frame como imagem
            success = save_frame_as_image(frame, filepath, self.snapshot_quality)
            
            if success:
                # Adicionar à lista de snapshots
                snapshot_info = {
                    "filename": filename,
                    "timestamp": datetime.now(),
                    "frame_id": frame_id,
                    "epi_summary": epi_summary or {},
                    "filepath": filepath
                }
                
                self.snapshots.insert(0, snapshot_info)
                
                # Limitar número de snapshots em memória
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots = self.snapshots[:self.max_snapshots]
                
                logger.info(f"Snapshot capturado: {filename}")
                return filepath
            else:
                logger.error("Falha ao salvar snapshot")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao capturar snapshot: {e}")
            return None
    
    def get_snapshot_list(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retorna lista de snapshots"""
        try:
            end_index = offset + limit
            return self.snapshots[offset:end_index]
        except Exception as e:
            logger.error(f"Erro ao obter lista de snapshots: {e}")
            return []
    
    def get_snapshot_by_id(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """Retorna snapshot por frame ID"""
        try:
            for snapshot in self.snapshots:
                if snapshot["frame_id"] == frame_id:
                    return snapshot
            return None
        except Exception as e:
            logger.error(f"Erro ao buscar snapshot por ID: {e}")
            return None
    
    def delete_snapshot(self, filename: str) -> bool:
        """Remove snapshot específico"""
        try:
            filepath = self.snapshot_dir / filename
            
            if filepath.exists():
                # Remover arquivo
                filepath.unlink()
                
                # Remover da lista
                self.snapshots = [s for s in self.snapshots if s["filename"] != filename]
                
                logger.info(f"Snapshot removido: {filename}")
                return True
            else:
                logger.warning(f"Snapshot não encontrado: {filename}")
                return False
                
        except Exception as e:
            logger.error(f"Erro ao remover snapshot: {e}")
            return False
    
    def cleanup_old_snapshots(self, max_age_hours: int = 24) -> int:
        """Remove snapshots antigos"""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            removed_count = 0
            
            snapshots_to_remove = []
            for snapshot in self.snapshots:
                if snapshot["timestamp"].timestamp() < cutoff_time:
                    snapshots_to_remove.append(snapshot)
            
            for snapshot in snapshots_to_remove:
                if self.delete_snapshot(snapshot["filename"]):
                    removed_count += 1
            
            logger.info(f"Removidos {removed_count} snapshots antigos")
            return removed_count
            
        except Exception as e:
            logger.error(f"Erro ao limpar snapshots antigos: {e}")
            return 0
    
    def get_snapshot_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos snapshots"""
        try:
            total_snapshots = len(self.snapshots)
            
            if total_snapshots == 0:
                return {
                    "total": 0,
                    "oldest": None,
                    "newest": None,
                    "total_size_mb": 0
                }
            
            # Calcular tamanho total
            total_size = sum(
                snapshot["filepath"].stat().st_size 
                for snapshot in self.snapshots 
                if snapshot["filepath"].exists()
            )
            total_size_mb = total_size / (1024 * 1024)
            
            return {
                "total": total_snapshots,
                "oldest": self.snapshots[-1]["timestamp"].isoformat(),
                "newest": self.snapshots[0]["timestamp"].isoformat(),
                "total_size_mb": round(total_size_mb, 2)
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas: {e}")
            return {"error": str(e)}

class EPISnapshotSystem:
    """Sistema de snapshot para EPIs"""
    
    def __init__(self):
        self.snapshot_manager = SnapshotManager()
        self.last_frame = None
        self.last_frame_id = 0
        
        logger.info("Sistema de snapshot EPI inicializado")
    
    def capture_snapshot(self, frame: np.ndarray = None, frame_id: int = None, epi_summary: Dict[str, int] = None) -> Optional[Path]:
        """Captura snapshot"""
        try:
            # Usar frame atual se não especificado
            if frame is None:
                frame = self.last_frame
            
            if frame is None:
                logger.warning("Nenhum frame disponível para snapshot")
                return None
            
            # Usar frame ID atual se não especificado
            if frame_id is None:
                frame_id = self.last_frame_id
            
            # Capturar snapshot
            snapshot_path = self.snapshot_manager.capture_snapshot(frame, frame_id, epi_summary)
            
            if snapshot_path:
                logger.info(f"Snapshot capturado com sucesso: {snapshot_path.name}")
            
            return snapshot_path
            
        except Exception as e:
            logger.error(f"Erro ao capturar snapshot: {e}")
            return None
    
    def update_frame(self, frame: np.ndarray, frame_id: int):
        """Atualiza frame atual"""
        self.last_frame = frame.copy() if frame is not None else None
        self.last_frame_id = frame_id
    
    def get_snapshot_list(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """Retorna lista de snapshots"""
        return self.snapshot_manager.get_snapshot_list(limit, offset)
    
    def get_snapshot_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas dos snapshots"""
        return self.snapshot_manager.get_snapshot_stats()
    
    def cleanup(self):
        """Cleanup do sistema"""
        try:
            # Limpar snapshots antigos (mais de 24 horas)
            self.snapshot_manager.cleanup_old_snapshots(24)
            logger.info("Sistema de snapshot finalizado")
        except Exception as e:
            logger.error(f"Erro no cleanup do snapshot: {e}")
