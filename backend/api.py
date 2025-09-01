"""
API Principal do Backend Athena
Integração com todos os módulos do sistema
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
from .config import CONFIG
from .utils import setup_logging, encode_frame_jpeg, get_system_info
from src.epi_detector import EPIDetectionSystem
from .snapshot import EPISnapshotSystem
from .history import EPIHistorySystem

# Configurar logging
logger = setup_logging(CONFIG.LOG_LEVEL, CONFIG.LOG_FORMAT)

# Inicializar FastAPI
app = FastAPI(
    title="Athena EPI Detection API",
    description="API para detecção de EPIs usando YOLOv5",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.CORS_ALLOW_ORIGINS,
    allow_credentials=CONFIG.CORS_ALLOW_CREDENTIALS,
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
    img_size: int = 640
    enable_tracking: bool = True
    device_preference: str = "auto"  # auto, cpu, cuda
    force_cpu_only: bool = False

    performance: Dict[str, Any] = {
        "max_fps": 30,
        "quality": "auto",
        "gpu_acceleration": True
    }

class SnapshotResponse(BaseModel):
    saved: bool
    url: Optional[str] = None
    message: str

class HistoryItem(BaseModel):
    id: int
    timestamp: str
    total_pessoas: int
    sem_capacete: int
    sem_colete: int
    violations: List[str] = []
    frame_id: Optional[int] = None

# Estado global da aplicação
class AppState:
    def __init__(self):
        self.detection_system = None
        self.snapshot_system = None
        self.history_system = None
        self.connection_status = "disconnected"
        self.stats = {
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0,
            "total_pessoas": 0,
            "compliance_rate": 0.0
        }
        self.config = ConfigData()
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0
        self.version = "v2.1.0"
        self.api_version = "v1.0.0"
        self.model_loaded = False
        self.system_logs = []
        
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do sistema"""
        uptime = time.time() - self.start_time
        
        # Obter informações do sistema
        system_info = get_system_info()
        
        return {
            "status": "online" if self.detection_system else "offline",
            "fps": self.fps,
            "uptime_s": int(uptime),
            "frame_count": self.frame_count,
            "connection_status": self.connection_status,
            "last_update": datetime.now().isoformat(),
            "version": self.version,
            "api_version": self.api_version,
            "model_loaded": self.model_loaded,
            "system_info": system_info
        }
        
    def add_log(self, level: str, message: str):
        """Adiciona log ao sistema"""
        log_entry = {
            "id": len(self.system_logs) + 1,
            "level": level.upper(),
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.system_logs.append(log_entry)
        
        # Manter apenas os últimos 100 logs
        if len(self.system_logs) > 100:
            self.system_logs = self.system_logs[-100:]

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
        
        # Validar configurações
        if not CONFIG.validate_config():
            raise Exception("Configurações inválidas")
        
        # Inicializar sistema de detecção
        app_state.detection_system = EPIDetectionSystem()
        
        # Obter configurações do modelo
        model_config = CONFIG.get_model_config()
        
        # Inicializar com configurações de dispositivo
        app_state.detection_system.initialize_system(
            model_path=model_config["model_path"],
            force_cpu_only=model_config["force_cpu_only"],
            device_preference=model_config["device_preference"]
        )
        app_state.add_log("INFO", f"Sistema de detecção inicializado - Dispositivo: {model_config['device_preference']}")
        
        # Inicializar sistema de snapshot
        app_state.snapshot_system = EPISnapshotSystem()
        app_state.add_log("INFO", "Sistema de snapshot inicializado")
        
        # Inicializar sistema de histórico
        app_state.history_system = EPIHistorySystem()
        app_state.add_log("INFO", "Sistema de histórico inicializado")
        
        # Marcar modelo como carregado
        app_state.model_loaded = True
        app_state.add_log("INFO", "Modelo YOLOv5 carregado com sucesso")
        
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
    
    if app_state.snapshot_system:
        app_state.snapshot_system.cleanup()
    
    if app_state.history_system:
        app_state.history_system.cleanup()

# Endpoints da API

@app.get("/api")
async def api_info():
    """Informações da API"""
    return {
        "message": "Athena EPI Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde da API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - app_state.start_time,
        "systems": {
            "detection": app_state.detection_system is not None,
            "snapshot": app_state.snapshot_system is not None,
            "history": app_state.history_system is not None
        }
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
                    frame_bytes = encode_frame_jpeg(frame, CONFIG.SNAPSHOT_QUALITY)
                    
                    # Enviar frame MJPEG
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                    )
                else:
                    # Se não há frame, enviar frame vazio ou aguardar
                    await asyncio.sleep(0.1)
                    continue
                
                # Aguardar próximo frame
                await asyncio.sleep(1/CONFIG.VIDEO_FPS)
                
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
                    stats = app_state.detection_system.get_stats()
                    
                    # Formatar dados para SSE com validação avançada
                    event_data = {
                        "frame_id": app_state.frame_count,
                        "boxes": detections,
                        "epi_summary": stats
                    }
                    
                    # Adicionar detalhes de validação se disponível
                    if hasattr(app_state.detection_system.detector, 'get_detection_summary'):
                        try:
                            summary = app_state.detection_system.detector.get_detection_summary()
                            event_data.update({
                                "validation_details": summary.get("validation_details", {}),
                                "compliance_rate": summary.get("epi_summary", {}).get("compliance_rate", 0.0),
                                "total_people": summary.get("epi_summary", {}).get("total_pessoas", 0),
                                "violations": summary.get("epi_summary", {}).get("violations", [])
                            })
                        except Exception as e:
                            logger.warning(f"Erro ao obter detalhes de validação: {e}")
                    
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                    # Adicionar ao histórico se houver detecções
                    if detections and app_state.history_system:
                        app_state.history_system.add_detection(
                            app_state.frame_count, detections, stats
                        )
                
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
        
        # Obter frame atual
        frame = None
        frame_id = 0
        epi_summary = {}
        
        if app_state.detection_system:
            frame = app_state.detection_system.get_current_frame()
            frame_id = app_state.frame_count
            epi_summary = app_state.detection_system.get_stats()
        
        # Capturar snapshot
        snapshot_path = app_state.snapshot_system.capture_snapshot(frame, frame_id, epi_summary)
        
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
    """Retorna estatísticas atuais expandidas"""
    if app_state.detection_system:
        base_stats = app_state.detection_system.get_stats()
    else:
        base_stats = app_state.stats
    
    # Adicionar métricas avançadas
    advanced_stats = {
        **base_stats,
        "total_pessoas": base_stats.get("com_capacete", 0) + base_stats.get("sem_capacete", 0),
        "compliance_rate": 0.0,
        "last_update": datetime.now().isoformat(),
        "system_status": "online" if app_state.detection_system else "offline"
    }
    
    # Calcular taxa de compliance
    total_people = advanced_stats["total_pessoas"]
    if total_people > 0:
        compliant_people = base_stats.get("com_capacete", 0) + base_stats.get("com_colete", 0)
        advanced_stats["compliance_rate"] = round(compliant_people / (total_people * 2), 2)
    
    return advanced_stats

@app.get("/history")
async def get_history(
    offset: int = 0, 
    limit: int = 50, 
    compliance_filter: str = None,
    start_date: str = None,
    end_date: str = None
):
    """Retorna histórico de violações de EPI"""
    if not app_state.history_system:
        raise HTTPException(status_code=503, detail="Sistema de histórico não disponível")
    
    # Filtrar por data se especificado
    # Aplicar filtros (simplificado)
    compliance_filter_param = compliance_filter if compliance_filter else None
    
    history_data = app_state.history_system.get_history(limit, offset, compliance_filter_param)
    history_stats = app_state.history_system.get_history_stats()
    
    return {
        "data": history_data,
        "stats": history_stats,
        "offset": offset,
        "limit": limit,
        "filters": {
            "compliance_filter": compliance_filter,
            "start_date": start_date,
            "end_date": end_date
        }
    }

@app.get("/history/stats")
async def get_history_stats():
    """Retorna estatísticas do histórico"""
    if not app_state.history_system:
        raise HTTPException(status_code=503, detail="Sistema de histórico não disponível")
    
    return app_state.history_system.get_history_stats()

@app.get("/history/trend")
async def get_compliance_trend(hours: int = 24):
    """Retorna tendência de compliance"""
    if not app_state.history_system:
        raise HTTPException(status_code=503, detail="Sistema de histórico não disponível")
    
    return app_state.history_system.get_compliance_trend(hours)

@app.get("/history/search")
async def search_history(query: str, limit: int = 50):
    """Busca no histórico"""
    if not app_state.history_system:
        raise HTTPException(status_code=503, detail="Sistema de histórico não disponível")
    
    return app_state.history_system.search_history(query, limit)

@app.get("/reports/violations")
async def get_violations_report(
    period: str = "24h",  # 24h, 7d, 30d, custom
    start_date: str = None,
    end_date: str = None
):
    """Retorna relatório de violações de EPI para gráficos"""
    if not app_state.history_system:
        raise HTTPException(status_code=503, detail="Sistema de histórico não disponível")
    
    # Obter dados para o período especificado
    if period == "custom" and start_date and end_date:
        data = app_state.history_system.get_violations_data(start_date, end_date)
    else:
        data = app_state.history_system.get_violations_data(period)
    
    return {
        "period": period,
        "data": data,
        "summary": {
            "total_violations": len(data),
            "total_people_without_epi": sum(item.get("sem_capacete", 0) + item.get("sem_colete", 0) for item in data),
            "most_common_violation": "Sem capacete" if sum(item.get("sem_capacete", 0) for item in data) > sum(item.get("sem_colete", 0) for item in data) else "Sem colete"
        }
    }

@app.get("/reports/trends")
async def get_trends_report(hours: int = 24):
    """Retorna dados de tendências de violações para gráficos"""
    if not app_state.history_system:
        raise HTTPException(status_code=503, detail="Sistema de histórico não disponível")
    
    trends_data = app_state.history_system.get_violations_trends(hours)
    
    return {
        "period_hours": hours,
        "data": trends_data,
        "intervals": list(trends_data.keys()) if trends_data else []
    }

@app.get("/status")
async def get_status():
    """Retorna status do sistema"""
    return app_state.get_status()

@app.get("/config")
async def get_config():
    """Retorna configuração atual"""
    return app_state.config

@app.get("/logs")
async def get_logs(level: str = None, limit: int = 50):
    """Retorna logs do sistema"""
    logs = app_state.system_logs
    
    # Filtrar por nível se especificado
    if level:
        logs = [log for log in logs if log["level"].lower() == level.lower()]
    
    # Limitar quantidade
    logs = logs[-limit:] if limit > 0 else logs
    
    return {
        "logs": logs,
        "total": len(logs),
        "levels": list(set(log["level"] for log in app_state.system_logs))
    }

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

# Rota para servir o frontend
@app.get("/athena")
async def serve_frontend():
    """Serve o arquivo index.html do frontend"""
    from fastapi.responses import FileResponse
    import os
    
    frontend_path = os.path.join("frontend", "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    else:
        raise HTTPException(status_code=404, detail="Frontend não encontrado")

# Rota raiz redireciona para o frontend
@app.get("/")
async def root():
    """Redireciona para o frontend"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/athena")

# Servir arquivos estáticos
app.mount("/snapshots", StaticFiles(directory=str(CONFIG.SNAPSHOT_DIR)), name="snapshots")
app.mount("/assets", StaticFiles(directory="frontend/assets"), name="frontend_assets")
app.mount("/js", StaticFiles(directory="frontend/js"), name="frontend_js")
app.mount("/styles", StaticFiles(directory="frontend/styles"), name="frontend_styles")

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
def start_server(host: str = None, port: int = None):
    """Inicia o servidor da API"""
    host = host or CONFIG.API_HOST
    port = port or CONFIG.API_PORT
    
    logger.info(f"Iniciando servidor em {host}:{port}")
    
    uvicorn.run(
        "backend.api:app",
        host=host,
        port=port,
        reload=CONFIG.API_RELOAD,
        log_level=CONFIG.LOG_LEVEL.lower()
    )



if __name__ == "__main__":
    start_server()
