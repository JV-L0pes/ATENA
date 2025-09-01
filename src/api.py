"""
API Backend para o Dashboard Athena
Integração com modelo YOLOv5 para detecção de EPIs
"""

import asyncio
import json
import logging
import time
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
from epi_detector import EPIDetectionSystem
from epi_snapshot_system import EPISnapshotSystem

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Athena EPI Detection API",
    description="API para detecção de EPIs usando YOLOv5",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic para validação
class DetectionBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str
    conf: float
    track_id: Optional[int] = None

class DetectionData(BaseModel):
    frame_id: int
    boxes: List[DetectionBox]
    epi_summary: Dict[str, int]

class ConfigData(BaseModel):
    conf_thresh: float = 0.35
    iou: float = 0.45
    max_detections: int = 50
    batch_size: int = 1
    enable_tracking: bool = True

class SnapshotResponse(BaseModel):
    saved: bool
    url: Optional[str] = None
    message: str

# Estado global da aplicação
class AppState:
    def __init__(self):
        self.detection_system = None
        self.snapshot_system = None
        self.connection_status = "disconnected"
        self.stats = {
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0
        }
        self.history = []
        self.config = ConfigData()
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do sistema"""
        uptime = time.time() - self.start_time
        return {
            "fps": self.fps,
            "uptime_s": int(uptime),
            "frame_count": self.frame_count,
            "connection_status": self.connection_status,
            "last_update": datetime.now().isoformat()
        }

# Instância global do estado
app_state = AppState()

# WebSocket connections
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
        """Envia mensagem para todas as conexões ativas"""
        if self.active_connections:
            await asyncio.gather(
                *[connection.send_text(message) for connection in self.active_connections],
                return_exceptions=True
            )

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Inicializa sistemas na inicialização da aplicação"""
    try:
        logger.info("Inicializando sistemas...")
        
        # Inicializar sistema de detecção
        app_state.detection_system = EPIDetectionSystem()
        app_state.detection_system.initialize_system()
        
        # Inicializar sistema de snapshot
        app_state.snapshot_system = EPISnapshotSystem()
        
        logger.info("Sistemas inicializados com sucesso")
        
    except Exception as e:
        logger.error(f"Erro ao inicializar sistemas: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup na finalização da aplicação"""
    logger.info("Finalizando aplicação...")
    if app_state.detection_system:
        app_state.detection_system.cleanup()

# Endpoints da API

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "Athena EPI Detection API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde da API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - app_state.start_time
    }

@app.get("/stream.mjpg")
async def video_stream():
    """Stream de vídeo MJPEG"""
    if not app_state.detection_system:
        raise HTTPException(status_code=503, detail="Sistema de detecção não inicializado")
    
    async def generate_frames():
        while True:
            try:
                # Obter frame do sistema de detecção
                frame = app_state.detection_system.get_current_frame()
                if frame is not None:
                    # Converter para JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Enviar frame MJPEG
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                
                # Aguardar próximo frame
                await asyncio.sleep(1/30)  # 30 FPS
                
            except Exception as e:
                logger.error(f"Erro no stream: {e}")
                await asyncio.sleep(1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws/detections")
async def websocket_detections(websocket: WebSocket):
    """WebSocket para detecções em tempo real"""
    await manager.connect(websocket)
    try:
        while True:
            # Aguardar mensagens do cliente
            data = await websocket.receive_text()
            
            # Processar mensagem se necessário
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Erro no WebSocket: {e}")
        manager.disconnect(websocket)

@app.get("/events/detections")
async def sse_detections():
    """Server-Sent Events para detecções"""
    async def event_generator():
        while True:
            try:
                # Obter detecções atuais
                if app_state.detection_system:
                    detections = app_state.detection_system.get_current_detections()
                    if detections:
                        # Formatar dados para SSE
                        event_data = {
                            "frame_id": app_state.frame_count,
                            "boxes": detections,
                            "epi_summary": app_state.stats
                        }
                        
                        yield f"data: {json.dumps(event_data)}\n\n"
                
                # Aguardar próximo evento
                await asyncio.sleep(0.1)  # 10 FPS para eventos
                
            except Exception as e:
                logger.error(f"Erro no SSE: {e}")
                await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.post("/snapshot", response_model=SnapshotResponse)
async def take_snapshot():
    """Captura snapshot do frame atual"""
    try:
        if not app_state.snapshot_system:
            raise HTTPException(status_code=503, detail="Sistema de snapshot não disponível")
        
        # Capturar snapshot
        snapshot_path = app_state.snapshot_system.capture_snapshot()
        
        if snapshot_path and snapshot_path.exists():
            # URL relativa para o snapshot
            snapshot_url = f"/snapshots/{snapshot_path.name}"
            
            return SnapshotResponse(
                saved=True,
                url=snapshot_url,
                message="Snapshot capturado com sucesso"
            )
        else:
            return SnapshotResponse(
                saved=False,
                message="Erro ao capturar snapshot"
            )
            
    except Exception as e:
        logger.error(f"Erro ao capturar snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Retorna estatísticas atuais"""
    return app_state.stats

@app.get("/history")
async def get_history(offset: int = 0, limit: int = 50):
    """Retorna histórico de detecções"""
    # Implementar lógica de histórico
    history_data = app_state.history[offset:offset + limit]
    return {
        "data": history_data,
        "total": len(app_state.history),
        "offset": offset,
        "limit": limit
    }

@app.get("/status")
async def get_status():
    """Retorna status do sistema"""
    return app_state.get_status()

@app.get("/config")
async def get_config():
    """Retorna configuração atual"""
    return app_state.config

@app.put("/config")
async def update_config(config: ConfigData):
    """Atualiza configuração do sistema"""
    try:
        # Atualizar configuração
        app_state.config = config
        
        # Aplicar configurações ao sistema de detecção
        if app_state.detection_system:
            app_state.detection_system.update_config(config.dict())
        
        logger.info("Configuração atualizada com sucesso")
        return {"message": "Configuração atualizada", "config": config}
        
    except Exception as e:
        logger.error(f"Erro ao atualizar configuração: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Servir arquivos estáticos
app.mount("/snapshots", StaticFiles(directory="snapshots"), name="snapshots")

# Middleware para atualizar estatísticas
@app.middleware("http")
async def update_stats_middleware(request, call_next):
    """Middleware para atualizar estatísticas"""
    response = await call_next(request)
    
    # Atualizar contadores se necessário
    if request.url.path == "/stats":
        # Atualizar FPS
        current_time = time.time()
        if hasattr(app_state, 'last_frame_time'):
            time_diff = current_time - app_state.last_frame_time
            if time_diff > 0:
                app_state.fps = 1.0 / time_diff
        app_state.last_frame_time = current_time
        
        # Incrementar contador de frames
        app_state.frame_count += 1
    
    return response

# Função para iniciar servidor
def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Inicia o servidor da API"""
    logger.info(f"Iniciando servidor em {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
