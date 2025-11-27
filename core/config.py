"""
Configurações do Core - Sistema de Detecção Athena
"""
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """Configurações centralizadas do sistema"""
    
    # Modelo
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")
    MODEL_CONF_THRESH = float(os.getenv("MODEL_CONF_THRESH", "0.25"))
    MODEL_IOU_THRESH = float(os.getenv("MODEL_IOU_THRESH", "0.45"))
    MODEL_MAX_DETECTIONS = int(os.getenv("MODEL_MAX_DETECTIONS", "300"))
    
    # Dispositivo
    FORCE_CPU_ONLY = os.getenv("FORCE_CPU_ONLY", "false").lower() == "true"
    DEVICE_PREFERENCE = os.getenv("DEVICE_PREFERENCE", "auto")
    
    # Vídeo
    VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")
    VIDEO_TYPE = os.getenv("VIDEO_TYPE", "rtsp")
    RTSP_URL = os.getenv("RTSP_URL", "")
    VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
    VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "640"))
    VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "480"))
    
    # Detecção
    DETECTION_FRAME_SKIP = int(os.getenv("DETECTION_FRAME_SKIP", "2"))
    DETECTION_RESIZE_FACTOR = float(os.getenv("DETECTION_RESIZE_FACTOR", "1.0"))
    
    # EPIs Requeridos
    REQUIRED_EPIS = set(os.getenv("REQUIRED_EPIS", "helmet,safety-vest,gloves,glasses").split(','))
    
    # Storage
    STORAGE_VIDEOS = Path("storage/videos")
    STORAGE_UPLOADS = Path("storage/uploads")
    STORAGE_REPORTS = Path("storage/reports")
    STORAGE_SNAPSHOTS = Path("storage/snapshots")
    STORAGE_LOGS = Path("storage/logs")
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Retorna configurações do modelo"""
        return {
            "model_path": cls.MODEL_PATH,
            "conf_thresh": cls.MODEL_CONF_THRESH,
            "iou_thresh": cls.MODEL_IOU_THRESH,
            "max_detections": cls.MODEL_MAX_DETECTIONS,
        }
    
    @classmethod
    def validate(cls) -> bool:
        """Valida configurações"""
        model_path = Path(cls.MODEL_PATH)
        if not model_path.exists():
            # Tentar caminho alternativo
            alt_path = Path("athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt")
            if alt_path.exists():
                cls.MODEL_PATH = str(alt_path)
            else:
                return False
        
        # Criar diretórios de storage
        cls.STORAGE_VIDEOS.mkdir(parents=True, exist_ok=True)
        cls.STORAGE_UPLOADS.mkdir(parents=True, exist_ok=True)
        cls.STORAGE_REPORTS.mkdir(parents=True, exist_ok=True)
        cls.STORAGE_SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        cls.STORAGE_LOGS.mkdir(parents=True, exist_ok=True)
        
        return True

