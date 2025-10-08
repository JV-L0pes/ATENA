"""
Configurações do Backend Athena
Centraliza todas as constantes e configurações
"""

import os
from pathlib import Path
from typing import Dict, Any

class BackendConfig:
    """Configurações do backend"""
    
    # Configurações da API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "3000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # Configurações do modelo - Usando o melhor modelo treinado da Fase 1
    MODEL_PATH = os.getenv("MODEL_PATH", "athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt")
    MODEL_CONF_THRESH = float(os.getenv("MODEL_CONF_THRESH", "0.25"))
    MODEL_IOU_THRESH = float(os.getenv("MODEL_IOU_THRESH", "0.45"))
    MODEL_MAX_DETECTIONS = int(os.getenv("MODEL_MAX_DETECTIONS", "50"))
    
    # Configurações de dispositivo (CPU/GPU)
    FORCE_CPU_ONLY = os.getenv("FORCE_CPU_ONLY", "false").lower() == "true"  # Padrão GPU com fallback
    DEVICE_PREFERENCE = os.getenv("DEVICE_PREFERENCE", "auto")  # Padrão auto (GPU se disponível, senão CPU)
    
    # Configurações de vídeo - Suporte para HTTP via Tailscale
    VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # Padrão: webcam local
    VIDEO_TYPE = os.getenv("VIDEO_TYPE", "usb")  # usb, ip, rtsp, udp, http
    RTSP_URL = os.getenv("RTSP_URL", "http://100.67.254.55:8554/webcam")  # HTTP stream do PC local
    VIDEO_FPS = int(os.getenv("VIDEO_FPS", "30"))
    VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "640"))
    VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "480"))
    
    # Configurações de detecção
    DETECTION_ENABLE_TRACKING = os.getenv("DETECTION_ENABLE_TRACKING", "true").lower() == "true"
    DETECTION_FRAME_QUEUE_SIZE = int(os.getenv("DETECTION_FRAME_QUEUE_SIZE", "10"))
    DETECTION_RESULT_QUEUE_SIZE = int(os.getenv("DETECTION_RESULT_QUEUE_SIZE", "10"))
    
    # Configurações de performance
    PERFORMANCE_MAX_DRAW_FPS = int(os.getenv("PERFORMANCE_MAX_DRAW_FPS", "30"))
    PERFORMANCE_MAX_CACHE_SIZE = int(os.getenv("PERFORMANCE_MAX_CACHE_SIZE", "100"))
    PERFORMANCE_DEBOUNCE_DELAY = int(os.getenv("PERFORMANCE_DEBOUNCE_DELAY", "100"))
    
    # Configurações de logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurações de CORS
    CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    
    # Configurações de snapshot
    SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", "snapshots"))
    SNAPSHOT_FORMAT = os.getenv("SNAPSHOT_FORMAT", "jpg")
    SNAPSHOT_QUALITY = int(os.getenv("SNAPSHOT_QUALITY", "95"))
    
    # Configurações de histórico
    HISTORY_MAX_ENTRIES = int(os.getenv("HISTORY_MAX_ENTRIES", "1000"))
    HISTORY_CLEANUP_INTERVAL = int(os.getenv("HISTORY_CLEANUP_INTERVAL", "3600"))  # 1 hora
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Retorna configurações do modelo"""
        return {
            "model_path": cls.MODEL_PATH,
            "conf_thresh": cls.MODEL_CONF_THRESH,
            "iou_thresh": cls.MODEL_IOU_THRESH,
            "max_detections": cls.MODEL_MAX_DETECTIONS,
            "force_cpu_only": cls.FORCE_CPU_ONLY,
            "device_preference": cls.DEVICE_PREFERENCE
        }
    
    @classmethod
    def get_video_config(cls) -> Dict[str, Any]:
        """Retorna configurações de vídeo"""
        return {
            "source": cls.VIDEO_SOURCE,
            "fps": cls.VIDEO_FPS,
            "width": cls.VIDEO_WIDTH,
            "height": cls.VIDEO_HEIGHT
        }
    
    @classmethod
    def get_detection_config(cls) -> Dict[str, Any]:
        """Retorna configurações de detecção"""
        return {
            "enable_tracking": cls.DETECTION_ENABLE_TRACKING,
            "frame_queue_size": cls.DETECTION_FRAME_QUEUE_SIZE,
            "result_queue_size": cls.DETECTION_RESULT_QUEUE_SIZE
        }
    
    @classmethod
    def get_performance_config(cls) -> Dict[str, Any]:
        """Retorna configurações de performance"""
        return {
            "max_draw_fps": cls.PERFORMANCE_MAX_DRAW_FPS,
            "max_cache_size": cls.PERFORMANCE_MAX_CACHE_SIZE,
            "debounce_delay": cls.PERFORMANCE_DEBOUNCE_DELAY
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Valida configurações"""
        try:
            # Verificar se modelo existe
            model_path = Path(cls.MODEL_PATH)
            if not model_path.exists():
                print(f"❌ Modelo não encontrado: {model_path}")
                print(f"📁 Procurando por modelos treinados...")
                
                # Procurar por modelos treinados
                train_dir = Path("yolov5/runs/train")
                if train_dir.exists():
                    for train_folder in train_dir.iterdir():
                        if train_folder.is_dir():
                            weights_dir = train_folder / "weights"
                            if weights_dir.exists():
                                best_model = weights_dir / "best.pt"
                                if best_model.exists():
                                    print(f"✅ Modelo encontrado: {best_model}")
                                    # Atualizar caminho do modelo
                                    cls.MODEL_PATH = str(best_model)
                                    break
                
                if not Path(cls.MODEL_PATH).exists():
                    print(f"❌ Nenhum modelo treinado encontrado")
                    return False
            
            # Verificar diretório de snapshots
            cls.SNAPSHOT_DIR.mkdir(exist_ok=True)
            
            # Validar valores numéricos
            if cls.API_PORT < 1 or cls.API_PORT > 65535:
                print(f"❌ Porta inválida: {cls.API_PORT}")
                return False
            
            if cls.MODEL_CONF_THRESH < 0 or cls.MODEL_CONF_THRESH > 1:
                print(f"❌ Threshold de confiança inválido: {cls.MODEL_CONF_THRESH}")
                return False
            
            if cls.MODEL_IOU_THRESH < 0 or cls.MODEL_IOU_THRESH > 1:
                print(f"❌ Threshold IoU inválido: {cls.MODEL_IOU_THRESH}")
                return False
            
            print("✅ Configurações validadas com sucesso")
            print(f"🎯 Modelo: {cls.MODEL_PATH}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao validar configurações: {e}")
            return False

# Instância global
CONFIG = BackendConfig()
