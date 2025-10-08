"""
API Backend Integrada com Sistema Otimizado da Fase 1 - Athena Dashboard
Sistema de detecção de EPIs com modelo otimizado da Fase 1 (best.pt)
"""

import asyncio
import json
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Importar módulos locais
from .config import CONFIG
from .utils import setup_logging, encode_frame_jpeg, get_system_info

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar sistema otimizado da Fase 1
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    from athena_realtime_optimized import AthenaDetectionSystemOptimized, AthenaPhase1Detector
    from webcam_recovery import StableWebcamCapture
    logger.info("✅ Usando sistema de tempo real otimizado")
except ImportError:
    from athena_detection_optimized import AthenaDetectionSystemOptimized, AthenaPhase1Detector
    StableWebcamCapture = None
    logger.info("⚠️ Usando sistema padrão (tempo real não disponível)")

# ===== DETECTOR OTIMIZADO INTEGRADO =====

class EPIDetectorOptimizedAPI:
    """Detector de EPIs com modelo otimizado da Fase 1 - Performance Superior"""
    
    def __init__(self, model_path: str = None, video_source: str = None):
        # Usar modelo mais recente se existir
        if model_path is None:
            latest = Path("athena_model_latest.pt")
            if latest.exists():
                model_path = str(latest)
            else:
                model_path = "athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt"
        
        self.model_path = model_path
        self.video_source = video_source or os.getenv("RTSP_URL", "0")
        self.detector = None
        self.is_initialized = False
        
        # Configuração de câmera
        self.camera_type = "usb"
        self.camera_source = 0
        
        # Sistema de recuperação de webcam
        self.webcam_capture = None
        if StableWebcamCapture:
            self.webcam_capture = StableWebcamCapture(self.camera_source)
        
        # Estado atual
        self.current_detections = []
        self.current_frame = None
        self.processed_frame = None
        self.frame_count = 0
        
        # Estatísticas
        self.stats = {
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0,
            "total_pessoas": 0,
            "compliance_score": 0.0,
            "detection_rate": 0.0,
            "avg_confidence": 0.0
        }
        
        logging.info("EPI Detector Otimizado API inicializado")
    
    def initialize_model(self):
        """Inicializa o modelo otimizado da Fase 1"""
        try:
            logging.info(f"🚀 Carregando modelo otimizado da Fase 1: {self.model_path}")
            
            # Verificar se o modelo existe
            if not Path(self.model_path).exists():
                logging.error(f"❌ Modelo não encontrado: {self.model_path}")
                return False
            
            # Inicializar detector otimizado com fonte de vídeo configurada
            self.detector = AthenaPhase1Detector(model_path=self.model_path, video_source=self.video_source)
            
            # Configurar detector SEM iniciar captura de vídeo
            self.detector.setup_detector()
            
            # IMPORTANTE: NÃO iniciar o sistema de captura do detector
            # para evitar conflitos com o sistema de recuperação
            
            self.is_initialized = True
            logging.info("✅ Modelo otimizado da Fase 1 carregado com sucesso!")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro ao carregar modelo otimizado: {e}")
            return False
    
    def start_system(self):
        """Inicia sistema completo"""
        if not self.is_initialized:
            if not self.initialize_model():
                return False
        
        # Configurar câmera
        if not self._setup_camera():
            logging.error("❌ Falha ao configurar câmera")
            return False
        
        # Iniciar sistema de recuperação apenas para USB
        if self.webcam_capture and self.camera_type == "usb":
            if self.webcam_capture.initialize():
                logging.info("✅ Sistema de recuperação de webcam iniciado")
            else:
                logging.error("❌ Falha ao iniciar sistema de recuperação")
                return False
        elif self.camera_type == "rtsp":
            logging.info("📹 Sistema RTSP configurado - aguardando stream do PC local")
        
        logging.info("🎯 Sistema otimizado iniciado")
        return True
    
    def _setup_camera(self):
        """Configura a câmera - Suporte para RTSP via Tailscale"""
        try:
            # Detectar tipo de câmera baseado na fonte
            if self.video_source.startswith("rtsp://"):
                self.camera_type = "rtsp"
                self.camera_source = self.video_source
                logging.info(f"📹 Configurando RTSP: {self.camera_source}")
                
                # Testar conexão RTSP
                cap = cv2.VideoCapture(self.camera_source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para baixa latência
                
                if not cap.isOpened():
                    logging.warning(f"⚠️ RTSP não disponível ainda: {self.camera_source}")
                    logging.info("💡 Configure o ffmpeg no PC local para iniciar o stream")
                    # Não falhar aqui - o RTSP pode não estar ativo ainda
                    cap.release()
                    return True  # Permitir que o sistema inicie
                
                cap.release()
                logging.info(f"✅ RTSP configurado: {self.camera_source}")
                return True
                
            elif self.video_source.startswith("udp://"):
                self.camera_type = "udp"
                self.camera_source = self.video_source
                logging.info(f"📹 Configurando UDP: {self.camera_source}")
                
                # Testar conexão UDP
                cap = cv2.VideoCapture(self.camera_source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para baixa latência
                
                if not cap.isOpened():
                    logging.warning(f"⚠️ UDP não disponível ainda: {self.camera_source}")
                    logging.info("💡 Configure o ffmpeg no PC local para iniciar o stream UDP")
                    # Não falhar aqui - o UDP pode não estar ativo ainda
                    cap.release()
                    return True  # Permitir que o sistema inicie
                
                cap.release()
                logging.info(f"✅ UDP configurado: {self.camera_source}")
                return True
                
            elif self.video_source.startswith("http://"):
                self.camera_type = "http"
                self.camera_source = self.video_source
                logging.info(f"📹 Configurando HTTP: {self.camera_source}")
                
                # Testar conexão HTTP
                cap = cv2.VideoCapture(self.camera_source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para baixa latência
                
                if not cap.isOpened():
                    logging.warning(f"⚠️ HTTP não disponível ainda: {self.camera_source}")
                    logging.info("💡 Configure o ffmpeg no PC local para iniciar o stream HTTP")
                    # Não falhar aqui - o HTTP pode não estar ativo ainda
                    cap.release()
                    return True  # Permitir que o sistema inicie
                
                cap.release()
                logging.info(f"✅ HTTP configurado: {self.camera_source}")
                return True
                
            elif self.camera_type == "ip":
                # Para câmeras IP
                if not self.camera_source or not isinstance(self.camera_source, str):
                    logging.error(f"❌ URL da câmera IP inválida: {self.camera_source}")
                    return False
                
                # Testar conexão IP
                cap = cv2.VideoCapture(self.camera_source)
                if not cap.isOpened():
                    logging.error(f"❌ Falha ao conectar câmera IP: {self.camera_source}")
                    return False
                cap.release()
                
            else:
                # Para câmeras USB
                try:
                    camera_index = int(self.camera_source) if isinstance(self.camera_source, str) else self.camera_source
                except (ValueError, TypeError):
                    logging.error(f"❌ Índice da câmera USB inválido: {self.camera_source}")
                    return False
                
                # Usar sistema de recuperação se disponível
                if self.webcam_capture:
                    logging.info("🔄 Usando sistema de recuperação de webcam")
                    if self.webcam_capture.initialize():
                        logging.info("✅ Sistema de recuperação iniciado")
                        return True
                    else:
                        logging.error("❌ Falha ao inicializar sistema de recuperação")
                        return False
                else:
                    logging.warning("⚠️ Sistema de recuperação não disponível")
                    return True  # Permitir continuar sem câmera local
            
            logging.info(f"✅ Câmera {self.camera_type.upper()} configurada")
            return True
            
        except Exception as e:
            logging.error(f"❌ Erro ao configurar câmera: {e}")
            return False
    
    def get_current_frame_from_webcam(self) -> Optional[np.ndarray]:
        """Captura frame atual usando sistema de recuperação ou HTTP/RTSP"""
        # Se for HTTP/RTSP/UDP, usar VideoCapture diretamente
        if self.camera_type in ["http", "rtsp", "udp"]:
            cap = cv2.VideoCapture(self.camera_source)
            if cap.isOpened():
                # Configurações para RTSP
                if self.camera_type == "rtsp":
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return frame
        
        # Fallback para sistema de recuperação
        if self.webcam_capture and self.webcam_capture.is_initialized:
            ret, frame = self.webcam_capture.read()
            if ret and frame is not None:
                return frame
        return None
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Processa frame com detector otimizado"""
        if not self.is_initialized or not self.detector:
            logging.warning("⚠️ Detector não inicializado, inicializando agora...")
            if not self.initialize_model():
                return {"detections": [], "stats": self.stats}
        
        try:
            # Processar frame com detector otimizado
            results = self.detector.process_frame(frame)
            
            # Atualizar estado
            self.current_frame = frame.copy()
            self.processed_frame = results.get('processed_frame', frame)
            self.current_detections = results.get('detections', [])
            self.frame_count += 1
            
            # Atualizar estatísticas
            self._update_stats(results)
            
            # Log para debug
            if self.frame_count % 30 == 0:  # Log a cada 30 frames
                logging.info(f"🔍 Frame {self.frame_count}: {len(self.current_detections)} detecções")
            
            return {
                "detections": self.current_detections,
                "stats": self.stats,
                "processed_frame": self.processed_frame
            }
            
        except Exception as e:
            logging.error(f"❌ Erro ao processar frame: {e}")
            return {"detections": [], "stats": self.stats}
    
    def _update_stats(self, results: Dict[str, Any]):
        """Atualiza estatísticas baseadas nos resultados"""
        detections = results.get('detections', [])
        summary = results.get('summary', {})
        
        # Mapear classes do modelo da Fase 1 para classes do sistema
        class_mapping = {
            'hardhat': 'helmet',
            'safety_vest': 'vest',
            'person': 'person',
            'no-helmet': 'no-helmet',
            'no-vest': 'no-vest'
        }
        
        # Reset contadores
        self.stats.update({
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0,
            "total_pessoas": 0
        })
        
        # Contar detecções por classe
        for detection in detections:
            class_name = detection.get('class_name', '')
            mapped_class = class_mapping.get(class_name, class_name)
            
            if mapped_class == 'helmet':
                self.stats["com_capacete"] += 1
            elif mapped_class == 'no-helmet':
                self.stats["sem_capacete"] += 1
            elif mapped_class == 'vest':
                self.stats["com_colete"] += 1
            elif mapped_class == 'no-vest':
                self.stats["sem_colete"] += 1
            elif mapped_class == 'person':
                self.stats["total_pessoas"] += 1
        
        # Calcular métricas avançadas
        total_people = self.stats["total_pessoas"]
        if total_people > 0:
            compliant_helmets = self.stats["com_capacete"]
            compliant_vests = self.stats["com_colete"]
            total_epis = total_people * 2  # capacete + colete
            
            self.stats["compliance_score"] = (compliant_helmets + compliant_vests) / total_epis
            self.stats["detection_rate"] = len(detections) / max(1, total_people)
            
            # Confiança média
            confidences = [d.get('confidence', 0.0) for d in detections]
            self.stats["avg_confidence"] = np.mean(confidences) if confidences else 0.0
    
    def get_current_detections(self) -> List[Dict]:
        """Retorna detecções atuais"""
        return self.current_detections
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Retorna frame processado"""
        return self.processed_frame if self.processed_frame is not None else self.current_frame
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas"""
        return self.stats.copy()
    
    def switch_camera(self, camera_type: str, camera_source):
        """Troca a fonte da câmera"""
        try:
            logging.info(f"🔄 Trocando câmera para {camera_type}: {camera_source}")
            
            # Atualizar configuração
            self.camera_type = camera_type
            self.camera_source = camera_source
            
            # Reconfigurar câmera
            if self._setup_camera():
                logging.info(f"✅ Câmera {camera_type.upper()} ativada")
                return True
            else:
                logging.error(f"❌ Falha ao trocar câmera")
                return False
                
        except Exception as e:
            logging.error(f"❌ Erro ao trocar câmera: {e}")
            return False
    
    def restart_system(self):
        """Reinicia o sistema"""
        try:
            logging.info("🔄 Reiniciando sistema otimizado")
            
            # Reinicializar detector
            if self.detector:
                self.detector.cleanup()
            
            # Reinicializar sistema
            return self.start_system()
            
        except Exception as e:
            logging.error(f"❌ Erro ao reiniciar sistema: {e}")
            return False
    
    def cleanup(self):
        """Para sistema"""
        if self.detector:
            self.detector.cleanup()

# ===== CONTINUAÇÃO DA API =====

# Configurar logging
logger = setup_logging(CONFIG.LOG_LEVEL, CONFIG.LOG_FORMAT)

# Inicializar FastAPI
app = FastAPI(
    title="Athena EPI Detection API - Otimizada Fase 1",
    description="API para detecção de EPIs usando modelo otimizado da Fase 1",
    version="3.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.CORS_ALLOW_ORIGINS,
    allow_credentials=CONFIG.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class DetectionBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str
    conf: float
    track_id: Optional[int] = None
    quality_score: Optional[str] = None
    compliance_score: Optional[float] = None

class DetectionData(BaseModel):
    frame_id: int
    boxes: List[DetectionBox]
    epi_summary: Dict[str, Any]
    compliance_score: float
    detection_rate: float
    avg_confidence: float

class ConfigData(BaseModel):
    conf_thresh: float = 0.5
    iou: float = 0.45
    max_detections: int = 100
    batch_size: int = 16
    enable_tracking: bool = True

class SnapshotResponse(BaseModel):
    saved: bool
    url: Optional[str] = None
    message: str

# Estado da aplicação
class AppState:
    def __init__(self):
        self.detection_system = None  # Será EPIDetectorOptimizedAPI
        self.snapshot_system = None
        self.history_system = None
        self.connection_status = "disconnected"
        self.stats = {
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0,
            "total_pessoas": 0,
            "compliance_score": 0.0,
            "detection_rate": 0.0,
            "avg_confidence": 0.0
        }
        self.config = ConfigData()
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.version = "v3.0.0"
        self.api_version = "v2.0.0"
        self.model_loaded = False
        
        # Configuração de câmera
        self.camera_config = {
            "type": "usb",
            "usb": {"index": 0},
            "ip": {"url": "", "username": "", "password": "", "timeout": 10}
        }
        self.current_camera_source = 0
        
    def get_status(self) -> Dict[str, Any]:
        """Status do sistema"""
        uptime = time.time() - self.start_time
        system_info = get_system_info()
        
        return {
            "status": "online" if self.detection_system and self.model_loaded else "offline",
            "fps": self.fps,
            "uptime_s": int(uptime),
            "frame_count": self.frame_count,
            "connection_status": self.connection_status,
            "last_update": datetime.now().isoformat(),
            "version": self.version,
            "api_version": self.api_version,
            "model_loaded": self.model_loaded,
            "model_type": "Fase 1 Otimizado",
            "system_info": system_info
        }

# Instância global
app_state = AppState()

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        app_state.connection_status = "connected"
        logger.info(f"WebSocket conectado. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if len(self.active_connections) == 0:
            app_state.connection_status = "disconnected"
        logger.info(f"WebSocket desconectado. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        if self.active_connections:
            await asyncio.gather(
                *[connection.send_text(message) for connection in self.active_connections],
                return_exceptions=True
            )

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Inicializa sistema na startup"""
    try:
        logger.info("🚀 Inicializando sistema otimizado da Fase 1...")
        
        # Validar configurações
        if not CONFIG.validate_config():
            raise Exception("Configurações inválidas")
        
        # ===== INICIALIZAÇÃO SISTEMA OTIMIZADO =====
        app_state.detection_system = EPIDetectorOptimizedAPI(video_source=os.getenv("RTSP_URL", "0"))
        
        # Configurar câmera baseado nas variáveis de ambiente RTSP
        video_type = os.getenv("VIDEO_TYPE", "usb")
        rtsp_url = os.getenv("RTSP_URL", "0")
        
        if video_type == "rtsp" and rtsp_url.startswith("rtsp://"):
            app_state.detection_system.camera_type = "rtsp"
            app_state.detection_system.camera_source = rtsp_url
            logging.info(f"📹 Configurando RTSP via env: {rtsp_url}")
        elif video_type == "http" and rtsp_url.startswith("http://"):
            app_state.detection_system.camera_type = "http"
            app_state.detection_system.camera_source = rtsp_url
            logging.info(f"📹 Configurando HTTP via env: {rtsp_url}")
        elif video_type == "udp" and rtsp_url.startswith("udp://"):
            app_state.detection_system.camera_type = "udp"
            app_state.detection_system.camera_source = rtsp_url
            logging.info(f"📹 Configurando UDP via env: {rtsp_url}")
        else:
            # Fallback para configuração padrão
            camera_config = app_state.camera_config
            app_state.detection_system.camera_type = camera_config.get("type", "usb")
            
            if camera_config.get("type") == "ip":
                app_state.detection_system.camera_source = camera_config.get("ip", {}).get("url", "")
            else:
                app_state.detection_system.camera_source = camera_config.get("usb", {}).get("index", 0)
        
        # Inicializar sistema
        if app_state.detection_system.start_system():
            app_state.model_loaded = True
            logger.info("✅ Sistema otimizado da Fase 1 inicializado com sucesso!")
        else:
            logger.error("❌ Falha ao inicializar sistema otimizado")
            raise Exception("Falha na inicialização do sistema")
        
        # Inicializar outros sistemas
        from .snapshot import EPISnapshotSystem
        from .history import EPIHistorySystem
        
        app_state.snapshot_system = EPISnapshotSystem()
        app_state.history_system = EPIHistorySystem()
        
        logger.info("✅ Sistema completo inicializado com sucesso!")
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar sistema: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup na finalização"""
    logger.info("🧹 Finalizando aplicação...")
    if app_state.detection_system:
        app_state.detection_system.cleanup()

# ===== ENDPOINTS DA API =====

@app.get("/api")
async def api_info():
    """Informações da API"""
    return {
        "message": "Athena EPI Detection API - Otimizada Fase 1",
        "version": "3.0.0",
        "status": "running",
        "detector": "Modelo atual",
        "model_path": str(Path(app_state.detection_system.model_path)) if app_state.detection_system else "",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - app_state.start_time,
        "systems": {
            "detection": app_state.detection_system is not None,
            "model_loaded": app_state.model_loaded,
            "snapshot": app_state.snapshot_system is not None,
            "history": app_state.history_system is not None
        },
        "model_info": {
            "type": "Modelo atual",
            "path": str(Path(app_state.detection_system.model_path)) if app_state.detection_system else "",
            "classes": 17,
            "optimized": True
        }
    }

@app.get("/classes")
async def get_model_classes():
    """Retorna as classes do modelo carregado"""
    if not app_state.detection_system or not app_state.detection_system.detector:
        raise HTTPException(status_code=503, detail="Sistema de detecção não inicializado")
    
    # Obter classes do detector
    detector = app_state.detection_system.detector
    if hasattr(detector, 'class_names'):
        return {
            "classes": detector.class_names,
            "total_classes": len(detector.class_names),
            "model_path": str(detector.model_path) if hasattr(detector, 'model_path') else "unknown"
        }
    else:
        # Fallback para classes padrão do modelo best.pt
        return {
            "classes": [
                'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask-medical', 
                'foot', 'tools', 'glasses', 'gloves', 'helmet', 'hands', 'head', 
                'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
            ],
            "total_classes": 17,
            "model_path": "athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt"
        }

@app.get("/stream.mjpg")
async def video_stream():
    """Stream MJPEG com detecções desenhadas"""
    if not app_state.detection_system:
        raise HTTPException(status_code=503, detail="Sistema de detecção não inicializado")
    
    async def generate_frames():
        cap = None
        try:
            # Usar sistema de recuperação se disponível
            logging.info(f"🔍 Debug: webcam_capture = {app_state.detection_system.webcam_capture}")
            logging.info(f"🔍 Debug: camera_type = {app_state.detection_system.camera_type}")
            
            if (app_state.detection_system.webcam_capture and 
                app_state.detection_system.camera_type == "usb"):
                
                logging.info("🔄 Usando sistema de recuperação para stream")
                
                while True:
                    # Capturar frame usando sistema de recuperação
                    frame = app_state.detection_system.get_current_frame_from_webcam()
                    if frame is not None:
                        # Processar frame com sistema otimizado
                        results = app_state.detection_system.process_frame(frame)
                        processed_frame = results.get('processed_frame', frame)
                        
                        # Codificar frame
                        _, buffer = cv2.imencode('.jpg', processed_frame, 
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        await asyncio.sleep(0.033)  # ~30 FPS
                        
            else:
                # Fallback para método tradicional
                if app_state.detection_system.camera_type == "ip":
                    cap = cv2.VideoCapture(app_state.detection_system.camera_source)
                elif app_state.detection_system.camera_type in ["rtsp", "http", "udp"]:
                    cap = cv2.VideoCapture(app_state.detection_system.camera_source)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para baixa latência
                    
                    # Configurações específicas para RTSP/HTTP/UDP
                    if app_state.detection_system.camera_type == "rtsp":
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        # Configurações para lidar com erros H.264
                        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                else:
                    cap = cv2.VideoCapture(app_state.detection_system.camera_source)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not cap.isOpened():
                    raise HTTPException(status_code=503, detail="Câmera não disponível")
                
                while True:
                    ret, frame = cap.read()
                    if ret:
                        # Processar frame com sistema otimizado
                        results = app_state.detection_system.process_frame(frame)
                        processed_frame = results.get('processed_frame', frame)
                        
                        # Converter para JPEG
                        frame_bytes = encode_frame_jpeg(processed_frame, CONFIG.SNAPSHOT_QUALITY)
                        
                        # Enviar frame MJPEG
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                        )
                    
                    # Controlar FPS
                    await asyncio.sleep(1/CONFIG.VIDEO_FPS)
                
        except Exception as e:
            logger.error(f"❌ Erro no stream: {e}")
            await asyncio.sleep(1)
        finally:
            if cap:
                cap.release()
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/events/detections")
async def sse_detections():
    """SSE para detecções - Sistema Otimizado"""
    async def event_generator():
        cap = None
        try:
            # Configurar captura de vídeo baseado no tipo
            if app_state.detection_system.camera_type in ["rtsp", "http", "udp"]:
                cap = cv2.VideoCapture(app_state.detection_system.camera_source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para baixa latência
                
                # Configurações específicas para RTSP/HTTP/UDP
                if app_state.detection_system.camera_type == "rtsp":
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    # Configurações para lidar com erros H.264
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                
                logging.info(f"📹 SSE usando {app_state.detection_system.camera_type}: {app_state.detection_system.camera_source}")
            elif app_state.detection_system.camera_type == "ip":
                cap = cv2.VideoCapture(app_state.detection_system.camera_source)
            else:
                cap = cv2.VideoCapture(app_state.detection_system.camera_source)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not cap.isOpened():
                raise HTTPException(status_code=503, detail="Câmera não disponível")
            
            while True:
                ret, frame = cap.read()
                if ret:
                    # Processar frame com sistema otimizado
                    results = app_state.detection_system.process_frame(frame)
                    
                    detections = results.get('detections', [])
                    stats = results.get('stats', {})
                    
                    # Atualizar stats globais
                    app_state.stats = stats
                    app_state.frame_count = app_state.detection_system.frame_count
                    
                    # Formatar dados para SSE
                    event_data = {
                        "frame_id": app_state.frame_count,
                        "boxes": detections,
                        "epi_summary": stats,
                        "total_people": stats.get("total_pessoas", 0),
                        "compliance_rate": stats.get("compliance_score", 0.0),
                        "detection_rate": stats.get("detection_rate", 0.0),
                        "avg_confidence": stats.get("avg_confidence", 0.0),
                        "violations": []
                    }
                    
                    # Log para debug
                    logging.info(f"🔍 SSE Frame {app_state.frame_count}: {len(detections)} detecções")
                    
                    # Enviar dados
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                    # Adicionar ao histórico
                    if detections and app_state.history_system:
                        app_state.history_system.add_detection(
                            app_state.frame_count, detections, stats
                        )
                else:
                    logging.warning("⚠️ SSE: Frame não capturado")
                
                # Aguardar próximo evento (10 FPS)
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"❌ Erro no SSE: {e}")
            await asyncio.sleep(1)
        finally:
            if cap:
                cap.release()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.get("/stats")
async def get_stats():
    """Retorna estatísticas atuais"""
    if app_state.detection_system:
        base_stats = app_state.detection_system.get_stats()
    else:
        base_stats = app_state.stats
    
    # Adicionar métricas avançadas
    advanced_stats = {
        **base_stats,
        "last_update": datetime.now().isoformat(),
        "system_status": "online" if app_state.detection_system and app_state.model_loaded else "offline",
        "model_type": "Modelo atual",
        "model_path": str(Path(app_state.detection_system.model_path)) if app_state.detection_system else ""
    }
    
    return advanced_stats

@app.get("/status")
async def get_status():
    """Status do sistema"""
    return app_state.get_status()

@app.get("/config")
async def get_config():
    """Configuração atual"""
    return app_state.config

@app.put("/config")
async def update_config(config: ConfigData):
    """Atualiza configuração"""
    try:
        app_state.config = config
        
        # Aplicar ao detector se disponível
        if app_state.detection_system and hasattr(app_state.detection_system, 'detector'):
            if app_state.detection_system.detector:
                app_state.detection_system.detector.update_config({
                    'conf_thresh': config.conf_thresh,
                    'iou_thresh': config.iou,
                    'max_detections': config.max_detections
                })
        
        logger.info("⚙️ Configuração atualizada")
        return {"message": "Configuração atualizada", "config": config}
        
    except Exception as e:
        logger.error(f"❌ Erro ao atualizar configuração: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de histórico
@app.get("/history")
async def get_history(offset: int = 0, limit: int = 50):
    """Histórico de detecções"""
    if not app_state.history_system:
        raise HTTPException(status_code=503, detail="Sistema de histórico não disponível")
    
    history_data = app_state.history_system.get_history(limit, offset)
    history_stats = app_state.history_system.get_history_stats()
    
    return {
        "data": history_data,
        "stats": history_stats,
        "offset": offset,
        "limit": limit
    }

# Endpoints de câmera
@app.get("/camera/config")
async def get_camera_config():
    """Configuração atual da câmera"""
    return app_state.camera_config

@app.put("/camera/config")
async def update_camera_config(config: dict):
    """Atualiza configuração da câmera"""
    try:
        app_state.camera_config = config
        
        if app_state.detection_system:
            camera_type = config.get("type", "usb")
            if camera_type == "ip":
                camera_source = config.get("ip", {}).get("url", "")
            else:
                camera_source = config.get("usb", {}).get("index", 0)
            
            success = app_state.detection_system.switch_camera(camera_type, camera_source)
            if not success:
                raise HTTPException(status_code=500, detail="Falha ao trocar câmera")
        
        return {"message": "Configuração da câmera atualizada", "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar configuração: {str(e)}")

@app.post("/camera/restart")
async def restart_detection_system():
    """Reinicia o sistema de detecção"""
    try:
        if not app_state.detection_system:
            raise HTTPException(status_code=503, detail="Sistema de detecção não inicializado")
        
        success = app_state.detection_system.restart_system()
        if not success:
            raise HTTPException(status_code=500, detail="Falha ao reiniciar sistema")
        
        return {
            "message": "Sistema de detecção reiniciado com sucesso",
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao reiniciar sistema: {str(e)}")

# Rota raiz
@app.get("/")
async def root():
    """Redireciona para o frontend"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/athena")

@app.get("/athena")
async def serve_frontend():
    """Serve o frontend"""
    from fastapi.responses import FileResponse
    import os
    
    frontend_path = os.path.join("frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        raise HTTPException(status_code=404, detail="Frontend não encontrado")

# Servir arquivos estáticos
app.mount("/snapshots", StaticFiles(directory=str(CONFIG.SNAPSHOT_DIR)), name="snapshots")
app.mount("/assets", StaticFiles(directory="frontend/assets"), name="frontend_assets")
app.mount("/js", StaticFiles(directory="frontend/js"), name="frontend_js")
app.mount("/styles", StaticFiles(directory="frontend/styles"), name="frontend_styles")

# Função para iniciar servidor
def start_server(host: str = None, port: int = None):
    """Inicia servidor da API"""
    host = host or CONFIG.API_HOST
    port = port or CONFIG.API_PORT
    
    logger.info(f"🚀 Iniciando servidor OTIMIZADO em {host}:{port}")
    
    uvicorn.run(
        "backend.api_optimized:app",
        host=host,
        port=port,
        reload=CONFIG.API_RELOAD,
        log_level=CONFIG.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    start_server()
