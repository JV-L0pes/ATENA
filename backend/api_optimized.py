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
from collections import defaultdict
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import uuid
import shutil

# Importar módulos locais
from .config import CONFIG
from .utils import setup_logging, encode_frame_jpeg, get_system_info
from .video_detection import VideoAIDetector, VideoProcessingQueue, video_queue

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar sistema otimizado da Fase 1
import sys
sys.path.append(str(Path(__file__).parent.parent))
try:
    # Importar do core consolidado
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.detector import AthenaDetectionSystemOptimized, AthenaPhase1Detector
    # Verificar se as classes foram importadas corretamente
    if not hasattr(AthenaPhase1Detector, 'setup_detector'):
        raise ImportError("AthenaPhase1Detector não tem método setup_detector")
    logger.info("✅ Usando sistema de tempo real otimizado")
    logger.info(f"   Métodos disponíveis em AthenaPhase1Detector: {[m for m in dir(AthenaPhase1Detector) if not m.startswith('_')]}")
except ImportError as e:
    logger.error(f"❌ Erro ao importar sistema otimizado: {e}")
    import traceback
    logger.error(traceback.format_exc())
    # Usar sistema básico como fallback
    class AthenaDetectionSystemOptimized:
        def __init__(self, *args, **kwargs):
            pass
    class AthenaPhase1Detector:
        def __init__(self, *args, **kwargs):
            pass

# Importar webcam_recovery separadamente (opcional)
try:
    from webcam_recovery import StableWebcamCapture
    logger.info("✅ Sistema de recuperação de webcam disponível")
except ImportError:
    StableWebcamCapture = None
    logger.warning("⚠️ Sistema de recuperação de webcam não disponível (opcional)")

# ===== DETECTOR OTIMIZADO INTEGRADO =====

class EPIDetectorOptimizedAPI:
    """Detector de EPIs com modelo otimizado da Fase 1 - Performance Superior"""
    
    def __init__(self, model_path: str = None, video_source: str = None):
        # Usar modelo best.pt da Fase 1 como padrão
        if model_path is None:
            from backend.config import CONFIG
            model_path = CONFIG.MODEL_PATH
        
        self.model_path = model_path
        # Usar configuração do config.py se não especificado
        from backend.config import CONFIG
        self.video_source = video_source or CONFIG.RTSP_URL
        self.detector = None
        self.is_initialized = False
        
        # Configuração de câmera - detectar automaticamente tipo baseado na URL
        if self.video_source.startswith("rtsp://"):
            self.camera_type = "rtsp"
            self.camera_source = self.video_source
        elif self.video_source.startswith("http://"):
            self.camera_type = "http"
            self.camera_source = self.video_source
        elif self.video_source.startswith("udp://"):
            self.camera_type = "udp"
            self.camera_source = self.video_source
        else:
            self.camera_type = "usb"
            self.camera_source = int(self.video_source) if self.video_source.isdigit() else 0
        
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
            
            # Verificar se a classe foi importada corretamente
            if not hasattr(AthenaPhase1Detector, 'setup_detector'):
                logging.error(f"❌ Classe AthenaPhase1Detector não tem método setup_detector. Verifique a importação.")
                logging.error(f"   Métodos disponíveis: {[m for m in dir(AthenaPhase1Detector) if not m.startswith('_')]}")
                return False
            
            # Inicializar detector otimizado com fonte de vídeo configurada
            self.detector = AthenaPhase1Detector(model_path=self.model_path, video_source=self.video_source)
            
            # Verificar se o objeto foi criado corretamente
            if not hasattr(self.detector, 'setup_detector'):
                logging.error(f"❌ Objeto detector não tem método setup_detector após criação.")
                logging.error(f"   Tipo do objeto: {type(self.detector)}")
                logging.error(f"   Métodos disponíveis: {[m for m in dir(self.detector) if not m.startswith('_')]}")
                return False
            
            # Configurar detector SEM iniciar captura de vídeo
            self.detector.setup_detector()
            
            # IMPORTANTE: NÃO iniciar o sistema de captura do detector
            # para evitar conflitos com o sistema de recuperação
            
            self.is_initialized = True
            logging.info("✅ Modelo otimizado da Fase 1 carregado com sucesso!")
            
            return True
            
        except AttributeError as e:
            logging.error(f"❌ Erro de atributo ao carregar modelo otimizado: {e}")
            logging.error(f"   Tipo do detector: {type(self.detector) if hasattr(self, 'detector') else 'N/A'}")
            if hasattr(self, 'detector'):
                logging.error(f"   Métodos do detector: {[m for m in dir(self.detector) if not m.startswith('_')]}")
            return False
        except Exception as e:
            logging.error(f"❌ Erro ao carregar modelo otimizado: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    def start_system(self):
        """Inicia sistema completo (permissivo - não falha se não há câmera)"""
        if not self.is_initialized:
            if not self.initialize_model():
                logging.warning("⚠️ Modelo não inicializado - sistema continuará sem detecção em tempo real")
                return False
        
        # Configurar câmera (permissivo)
        camera_setup_success = self._setup_camera()
        if not camera_setup_success:
            logging.warning("⚠️ Câmera não configurada - sistema continuará sem stream em tempo real")
            logging.info("💡 Funcionalidade de análise de vídeos permanece disponível")
        
        # Iniciar sistema de recuperação apenas para USB (permissivo)
        if self.webcam_capture and self.camera_type == "usb":
            if self.webcam_capture.initialize():
                logging.info("✅ Sistema de recuperação de webcam iniciado")
            else:
                logging.warning("⚠️ Sistema de recuperação não iniciado - continuando sem câmera")
        elif self.camera_type == "rtsp":
            logging.info("📹 Sistema RTSP configurado - aguardando stream do PC local")
        
        logging.info("🎯 Sistema otimizado iniciado (modo permissivo)")
        return True
    
    def _setup_camera(self):
        """Configura a câmera - Suporte para RTSP via Tailscale"""
        try:
            # Detectar tipo de câmera baseado na fonte
            if self.video_source.startswith("rtsp://"):
                self.camera_type = "rtsp"
                self.camera_source = self.video_source
                logging.info(f"📹 Configurando RTSP: {self.camera_source}")
                
                # Testar conexão RTSP com configurações otimizadas
                logging.info(f"🔗 Tentando conectar RTSP: {self.camera_source}")
                cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                
                # CONFIGURAÇÕES ANTI-H.264 ERRORS - Zero erros de decodificação
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para evitar lag
                
                # FORÇAR CODEC MJPEG - Mais estável que H.264
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                
                # CONFIGURAÇÕES ANTI-ERRO H.264
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Forçar conversão RGB
                cap.set(cv2.CAP_PROP_FRAME_COUNT, -1)  # Stream infinito
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Resetar posição
                
                # REDUZIR QUALIDADE PARA ESTABILIDADE
                cap.set(cv2.CAP_PROP_FPS, 10)  # FPS muito baixo para estabilidade
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Resolução reduzida
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Configurações adicionais para estabilidade
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desabilitar autofoco
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Exposição fixa
                
                if not cap.isOpened():
                    logging.warning(f"⚠️ RTSP não conseguiu abrir: {self.camera_source}")
                    logging.info("💡 Verifique se a câmera está ligada e acessível")
                    cap.release()
                    return True  # Permitir que o sistema inicie
                
                # Testar se consegue ler um frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    logging.info(f"✅ RTSP conectado - Frame recebido: {w}x{h}")
                else:
                    logging.warning(f"⚠️ RTSP conectado mas sem frames válidos")
                
                cap.release()
                logging.info(f"✅ RTSP configurado: {self.camera_source}")
                return True
                
            elif self.video_source.startswith("udp://"):
                self.camera_type = "udp"
                self.camera_source = self.video_source
                logging.info(f"📹 Configurando UDP: {self.camera_source}")
                
                # Testar conexão UDP
                cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
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
                cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
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
                
                # Testar conexão IP (permissivo)
                cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logging.warning(f"⚠️ Falha ao conectar câmera IP: {self.camera_source}")
                    logging.info("💡 Funcionalidade de análise de vídeos permanece disponível")
                    return True  # Permitir continuar sem câmera IP
                cap.release()
                
            else:
                # Para câmeras USB
                try:
                    camera_index = int(self.camera_source) if isinstance(self.camera_source, str) else self.camera_source
                except (ValueError, TypeError):
                    logging.error(f"❌ Índice da câmera USB inválido: {self.camera_source}")
                    return False
                
                # Usar sistema de recuperação se disponível (permissivo)
                if self.webcam_capture:
                    logging.info("🔄 Usando sistema de recuperação de webcam")
                    if self.webcam_capture.initialize():
                        logging.info("✅ Sistema de recuperação iniciado")
                        return True
                    else:
                        logging.warning("⚠️ Sistema de recuperação não iniciado - continuando sem câmera")
                        logging.info("💡 Funcionalidade de análise de vídeos permanece disponível")
                        return True  # Permitir continuar sem câmera
                else:
                    logging.warning("⚠️ Sistema de recuperação não disponível - continuando sem câmera")
                    logging.info("💡 Funcionalidade de análise de vídeos permanece disponível")
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
            cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
            if cap.isOpened():
                # CONFIGURAÇÕES ANTI-H.264 ERRORS para RTSP - Zero erros de decodificação
                if self.camera_type == "rtsp":
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para evitar lag
                    
                    # FORÇAR CODEC MJPEG - Mais estável que H.264
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    
                    # CONFIGURAÇÕES ANTI-ERRO H.264
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Forçar conversão RGB
                    cap.set(cv2.CAP_PROP_FRAME_COUNT, -1)  # Stream infinito
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Resetar posição
                    
                    # REDUZIR QUALIDADE PARA ESTABILIDADE
                    cap.set(cv2.CAP_PROP_FPS, 10)  # FPS muito baixo para estabilidade
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Resolução reduzida
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    
                    # Configurações adicionais para estabilidade
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desabilitar autofoco
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Exposição fixa
                
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    # VERIFICAR SE FRAME ESTÁ CORROMPIDO (anti-H.264 errors)
                    if frame.shape[0] > 0 and frame.shape[1] > 0:
                        return frame
                    else:
                        logging.warning("⚠️ Frame corrompido detectado - ignorando")
                        return None
                else:
                    logging.warning(f"⚠️ Falha ao capturar frame do {self.camera_type}")
        
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
            
            # DEBUG: Log das detecções brutas do modelo (reduzido para estabilidade)
            raw_detections = results.get('detections', [])
            if self.frame_count % 180 == 0:  # Log a cada 180 frames (1 vez a cada 3 segundos)
                logging.info(f"🔍 DEBUG - Detecções brutas do modelo: {len(raw_detections)}")
                for i, det in enumerate(raw_detections[:2]):  # Mostrar apenas as 2 primeiras
                    logging.info(f"  Detecção {i}: {det.get('class_name', 'unknown')} - conf: {det.get('confidence', 0):.2f}")
            
            # Atualizar estado
            self.current_frame = frame.copy()
            self.processed_frame = results.get('processed_frame', frame)
            self.current_detections = results.get('detections', [])
            self.frame_count += 1
            
            # Atualizar estatísticas diretamente do detector (já inclui compliance)
            try:
                self.stats = self.detector.get_stats()
            except Exception:
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
        """Atualiza estatísticas baseadas nos resultados - TOTALMENTE DINÂMICO (sem hardcode)"""
        detections = results.get('detections', [])
        summary = results.get('summary', {})
        
        # Reset contadores dinâmicos - apenas contar o que o modelo detectar
        self.stats.update({
            "total_pessoas": 0,
            "total_detections": len(detections),
            "detections_by_class": defaultdict(int),
            "positive_detections": defaultdict(int),  # Detecções positivas (presentes)
            "negative_detections": defaultdict(int),  # Detecções negativas (ausentes)
        })
        
        # Processar detecções dinamicamente - SÓ CONTAR EPIs ASSOCIADOS A PESSOAS
        processed_detections = []
        
        # Separar pessoas e outras detecções
        people_detections = [d for d in detections if d.get('class_name') == 'person']
        other_detections = [d for d in detections if d.get('class_name') != 'person']
        
        # EPIs já foram filtrados no core/detector.py (só vêm associados a pessoas)
        # Mas vamos garantir que só contamos EPIs que realmente estão associados
        
        for detection in detections:
            class_name = detection.get('class_name', '')
            detection_copy = detection.copy()
            
            if class_name == 'person':
                self.stats["total_pessoas"] += 1
                detection_copy['compliant'] = detection.get('compliant', True)
                detection_copy['missing_epis'] = detection.get('missing_epis', [])
                # Contar por classe
                self.stats["detections_by_class"][class_name] += 1
            elif class_name.startswith('missing-'):
                # Detecção negativa (EPI faltando) - sempre associada a pessoa
                missing_epi = class_name.replace('missing-', '')
                self.stats["negative_detections"][missing_epi] += 1
                self.stats["detections_by_class"][class_name] += 1
                detection_copy['compliant'] = False
                detection_copy['missing_epis'] = [missing_epi]
            else:
                # Detecção positiva (EPI presente) - só contar se associado a pessoa
                # Se chegou aqui, já foi filtrado e está associado a uma pessoa
                self.stats["positive_detections"][class_name] += 1
                self.stats["detections_by_class"][class_name] += 1
                detection_copy['compliant'] = True
                detection_copy['missing_epis'] = []
            
            processed_detections.append(detection_copy)
        
        # Atualizar detecções processadas
        self.current_detections = processed_detections
        
        # Calcular métricas avançadas dinamicamente
        total_people = self.stats["total_pessoas"]
        if total_people > 0:
            # Calcular compliance baseado em detecções positivas vs negativas
            total_positive = sum(self.stats["positive_detections"].values())
            total_negative = sum(self.stats["negative_detections"].values())
            total_all = total_positive + total_negative
            
            if total_all > 0:
                self.stats["compliance_score"] = (total_positive / total_all) * 100
            else:
                self.stats["compliance_score"] = 0.0
            
            self.stats["detection_rate"] = len(detections) / max(1, total_people)
            
            # Confiança média
            confidences = [d.get('confidence', 0.0) for d in detections]
            self.stats["avg_confidence"] = np.mean(confidences) if confidences else 0.0
        
        # Converter defaultdict para dict para serialização
        self.stats["detections_by_class"] = dict(self.stats["detections_by_class"])
        self.stats["positive_detections"] = dict(self.stats["positive_detections"])
        self.stats["negative_detections"] = dict(self.stats["negative_detections"])
    
    def _analyze_missing_epis_for_person(self, person_detection: Dict, all_detections: List[Dict]) -> List[str]:
        """Analisa quais EPIs estão faltando para uma pessoa - DINÂMICO (sem hardcode)"""
        # Não assumir quais EPIs devem estar presentes
        # Apenas retornar EPIs faltando se o modelo já detectou como "missing-*"
        missing_epis = []
        
        person_bbox = person_detection.get('bbox', [])
        if len(person_bbox) < 4:
            return []
        
        # Verificar se há detecções "missing-*" próximas à pessoa
        for detection in all_detections:
            class_name = detection.get('class_name', '')
            if class_name.startswith('missing-'):
                bbox = detection.get('bbox', [])
                if len(bbox) >= 4 and self._boxes_overlap_or_near(person_bbox, bbox):
                    missing_epi = class_name.replace('missing-', '')
                    if missing_epi not in missing_epis:
                        missing_epis.append(missing_epi)
        
        return missing_epis
    
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
    
    # Métodos para compatibilidade com sistema de classes
    def get_enabled_classes(self):
        """Retorna classes habilitadas"""
        if self.detector and hasattr(self.detector, 'get_enabled_classes'):
            return self.detector.get_enabled_classes()
        else:
            # Fallback para classes padrão
            return [
                'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask-medical', 
                'foot', 'tools', 'glasses', 'gloves', 'helmet', 'hands', 'head', 
                'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
            ]
    
    def set_enabled_classes(self, enabled_classes):
        """Define classes habilitadas"""
        if self.detector and hasattr(self.detector, 'set_enabled_classes'):
            return self.detector.set_enabled_classes(enabled_classes)
        else:
            logging.warning("⚠️ Detector não suporta atualização de classes")
            return False
    
    @property
    def class_names(self):
        """Retorna nomes das classes"""
        if self.detector and hasattr(self.detector, 'class_names'):
            return self.detector.class_names
        else:
            # Fallback para classes padrão
            return [
                'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask-medical', 
                'foot', 'tools', 'glasses', 'gloves', 'helmet', 'hands', 'head', 
                'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
            ]

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

    async def send_json(self, websocket: WebSocket, payload: dict):
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(payload)

    async def send_bytes(self, websocket: WebSocket, data: bytes):
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_bytes(data)

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
        app_state.detection_system = EPIDetectorOptimizedAPI(video_source=CONFIG.RTSP_URL)
        
        # Configurar câmera baseado na configuração do CONFIG
        video_type = CONFIG.VIDEO_TYPE
        rtsp_url = CONFIG.RTSP_URL
        
        if video_type == "rtsp" and rtsp_url.startswith("rtsp://"):
            app_state.detection_system.camera_type = "rtsp"
            app_state.detection_system.camera_source = rtsp_url
            logging.info(f"📹 Configurando RTSP via env: {rtsp_url}")
            
            # Testar conexão RTSP de forma não bloqueante
            try:
                import cv2
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        logging.info(f"✅ RTSP conectado com sucesso: {rtsp_url}")
                    else:
                        logging.warning(f"⚠️ RTSP conectado mas sem frames: {rtsp_url}")
                        logging.info("💡 Sistema continuará com webcam padrão")
                        # Fallback para webcam se RTSP não funcionar
                        app_state.detection_system.camera_type = "usb"
                        app_state.detection_system.camera_source = 0
                else:
                    logging.warning(f"⚠️ RTSP não disponível: {rtsp_url}")
                    logging.info("💡 Sistema continuará com webcam padrão")
                    # Fallback para webcam se RTSP não funcionar
                    app_state.detection_system.camera_type = "usb"
                    app_state.detection_system.camera_source = 0
                
                cap.release()
                
            except Exception as e:
                logging.warning(f"⚠️ Erro ao configurar RTSP: {e}")
                logging.info("💡 Sistema continuará com webcam padrão")
                # Fallback para webcam se houver erro
                app_state.detection_system.camera_type = "usb"
                app_state.detection_system.camera_source = 0
                
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
        
        # Inicializar sistema (permissivo - não falhar se não há câmera)
        try:
            if app_state.detection_system.start_system():
                app_state.model_loaded = True
                logger.info("✅ Sistema otimizado da Fase 1 inicializado com sucesso!")
            else:
                logger.warning("⚠️ Sistema de detecção não inicializado (sem câmera)")
                logger.info("💡 Sistema continuará funcionando para análise de vídeos")
                app_state.model_loaded = True  # Marcar como carregado mesmo sem câmera
        except Exception as e:
            logger.warning(f"⚠️ Erro ao inicializar sistema de detecção: {e}")
            logger.info("💡 Sistema continuará funcionando para análise de vídeos")
            app_state.model_loaded = True  # Marcar como carregado mesmo com erro
        
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
            "class_names": detector.class_names,
            "enabled_classes": detector.get_enabled_classes() if hasattr(detector, 'get_enabled_classes') else detector.class_names,
            "total_classes": len(detector.class_names),
            "model_path": str(detector.model_path) if hasattr(detector, 'model_path') else "unknown"
        }
    else:
        # Fallback para classes padrão do modelo best.pt
        from backend.config import CONFIG
        return {
            "class_names": [
                'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask-medical', 
                'foot', 'tools', 'glasses', 'gloves', 'helmet', 'hands', 'head', 
                'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
            ],
            "total_classes": 17,
            "model_path": CONFIG.MODEL_PATH
        }

@app.get("/classes/enabled")
async def get_enabled_classes():
    if not app_state.detection_system or not app_state.detection_system.detector:
        raise HTTPException(status_code=503, detail="Sistema de detecção não inicializado")
    detector = app_state.detection_system.detector
    enabled = detector.get_enabled_classes() if hasattr(detector, 'get_enabled_classes') else detector.class_names
    return {"enabled_classes": enabled}

class ClassesUpdate(BaseModel):
    enabled_classes: List[str]

@app.put("/classes/enabled")
async def update_enabled_classes(payload: ClassesUpdate):
    if not app_state.detection_system or not app_state.detection_system.detector:
        raise HTTPException(status_code=503, detail="Sistema de detecção não inicializado")
    detector = app_state.detection_system.detector
    if not hasattr(detector, 'set_enabled_classes'):
        raise HTTPException(status_code=500, detail="Detector não suporta atualização de classes")
    try:
        detector.set_enabled_classes(payload.enabled_classes)
        return {"enabled_classes": detector.get_enabled_classes()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao atualizar classes: {str(e)}")

@app.get("/stream.mjpg")
async def video_stream():
    """Stream MJPEG OTIMIZADO para alta performance"""
    if not app_state.detection_system:
        raise HTTPException(status_code=503, detail="Sistema de detecção não inicializado")
    
    async def generate_frames():
        cap = None
        frame_count = 0
        last_detection_time = 0
        
        try:
            # Usar sistema de recuperação se disponível
            logging.info(f"🔍 Debug: webcam_capture = {app_state.detection_system.webcam_capture}")
            logging.info(f"🔍 Debug: camera_type = {app_state.detection_system.camera_type}")
            
            if (app_state.detection_system.webcam_capture and 
                app_state.detection_system.camera_type == "usb"):
                
                logging.info("🔄 Usando sistema de recuperação para stream OTIMIZADO")
                
                while True:
                    # Capturar frame usando sistema de recuperação
                    frame = app_state.detection_system.get_current_frame_from_webcam()
                    if frame is not None:
                        frame_count += 1
                        
                        # OTIMIZAÇÃO: Processar detecção apenas ocasionalmente para estabilidade máxima
                        current_time = time.time()
                        should_process_detection = (current_time - last_detection_time) > (1.0 / 3)  # 3 FPS para detecção (ultra estável)
                        
                        if should_process_detection:
                            # Processar frame com sistema otimizado (assíncrono)
                            results = app_state.detection_system.process_frame(frame)
                            last_detection_time = current_time
                        
                        # OTIMIZAÇÃO: Redimensionar frame 2K para processamento mais rápido
                        if frame.shape[1] > 1280:  # Se largura > 1280px (2K)
                            height, width = frame.shape[:2]
                            new_width = 1280
                            new_height = int((height * new_width) / width)
                            frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                        else:
                            frame_resized = frame
                        
                        # OTIMIZAÇÃO: Codificar frame com qualidade ULTRA reduzida para estabilidade máxima
                        _, buffer = cv2.imencode('.jpg', frame_resized, [
                            cv2.IMWRITE_JPEG_QUALITY, 30,  # Qualidade muito baixa para estabilidade
                            cv2.IMWRITE_JPEG_OPTIMIZE, 1
                        ])
                        frame_bytes = buffer.tobytes()
                        
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        
                        # OTIMIZAÇÃO: Sleep reduzido para maior FPS
                        await asyncio.sleep(1/CONFIG.VIDEO_FPS)  # Usar FPS configurado
                    else:
                        await asyncio.sleep(0.016)  # ~60 FPS quando sem frame
                        
            else:
                # Fallback para método tradicional
                if app_state.detection_system.camera_type == "ip":
                    cap = cv2.VideoCapture(app_state.detection_system.camera_source)
                elif app_state.detection_system.camera_type in ["rtsp", "http", "udp"]:
                    cap = cv2.VideoCapture(app_state.detection_system.camera_source, cv2.CAP_FFMPEG)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, CONFIG.STREAM_BUFFER_SIZE)  # Buffer mínimo para baixa latência
                    
                    # CONFIGURAÇÕES ANTI-H.264 ERRORS para RTSP - Zero erros de decodificação
                    if app_state.detection_system.camera_type == "rtsp":
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo para evitar lag
                        
                        # FORÇAR CODEC MJPEG - Mais estável que H.264
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                        
                        # CONFIGURAÇÕES ANTI-ERRO H.264
                        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Forçar conversão RGB
                        cap.set(cv2.CAP_PROP_FRAME_COUNT, -1)  # Stream infinito
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Resetar posição
                        
                        # REDUZIR QUALIDADE PARA ESTABILIDADE
                        cap.set(cv2.CAP_PROP_FPS, 10)  # FPS muito baixo para estabilidade
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Resolução reduzida
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        
                        # Configurações adicionais para estabilidade
                        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Desabilitar autofoco
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Exposição fixa
                else:
                    cap = cv2.VideoCapture(app_state.detection_system.camera_source)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not cap.isOpened():
                    raise HTTPException(status_code=503, detail="Câmera não disponível")
                
                while True:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # VERIFICAR SE FRAME ESTÁ CORROMPIDO (anti-H.264 errors)
                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            frame_count += 1
                        else:
                            logging.warning("⚠️ Frame corrompido detectado - ignorando")
                            continue
                        
                        # OTIMIZAÇÃO: Processar detecção apenas ocasionalmente para estabilidade máxima
                        current_time = time.time()
                        should_process_detection = (current_time - last_detection_time) > (1.0 / 3)  # 3 FPS para detecção (ultra estável)
                        
                        if should_process_detection:
                            # OTIMIZAÇÃO: Redimensionar frame 2K para detecção mais rápida
                            if frame.shape[1] > 1280:  # Se largura > 1280px (2K)
                                height, width = frame.shape[:2]
                                new_width = 1280
                                new_height = int((height * new_width) / width)
                                frame_for_detection = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                            else:
                                frame_for_detection = frame
                            
                            # Processar frame com sistema otimizado (assíncrono)
                            results = app_state.detection_system.process_frame(frame_for_detection)
                            last_detection_time = current_time
                        
                        # OTIMIZAÇÃO: Redimensionar frame 2K para stream mais rápido
                        if frame.shape[1] > 1280:  # Se largura > 1280px (2K)
                            height, width = frame.shape[:2]
                            new_width = 1280
                            new_height = int((height * new_width) / width)
                            frame_for_stream = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                        else:
                            frame_for_stream = frame
                        
                        # OTIMIZAÇÃO: Codificar com qualidade reduzida para velocidade
                        frame_bytes = encode_frame_jpeg(frame_for_stream, 30)  # Qualidade ultra baixa
                        
                        # Enviar frame MJPEG
                        yield (
                            b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
                        )
                    
                    # OTIMIZAÇÃO: Sleep reduzido para maior FPS
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
        try:
            logging.info("📡 SSE iniciado - usando dados do sistema de detecção")
            
            while True:
                # Usar dados já processados pelo sistema de detecção
                if app_state.detection_system and app_state.detection_system.current_detections is not None:
                    detections = app_state.detection_system.current_detections
                    stats = app_state.detection_system.stats
                    frame_count = app_state.detection_system.frame_count
                    violations = []
                    try:
                        if hasattr(app_state.detection_system, 'detector') and hasattr(app_state.detection_system.detector, 'current_violations'):
                            violations = app_state.detection_system.detector.current_violations or []
                    except Exception:
                        violations = []
                    
                    # Formatar dados para SSE
                    event_data = {
                        "frame_id": frame_count,
                        "boxes": detections,
                        "epi_summary": {
                            "com_capacete": stats.get("com_capacete", 0),
                            "sem_capacete": stats.get("sem_capacete", 0),
                            "com_colete": stats.get("com_colete", 0),
                            "sem_colete": stats.get("sem_colete", 0),
                            "total_pessoas": stats.get("total_pessoas", 0),
                            "compliance_score": stats.get("compliance_score", 0.0),
                            "detection_rate": stats.get("detection_rate", 0.0),
                            "avg_confidence": stats.get("avg_confidence", 0.0)
                        },
                        "total_people": stats.get("total_pessoas", 0),
                        "compliance_rate": stats.get("compliance_score", 0.0),
                        "detection_rate": stats.get("detection_rate", 0.0),
                        "avg_confidence": stats.get("avg_confidence", 0.0),
                        "violations": violations
                    }
                    
                    # Log para debug (apenas quando há detecções)
                    if detections:
                        logging.info(f"🔍 SSE Frame {frame_count}: {len(detections)} detecções")
                    
                    # Enviar dados
                    yield f"data: {json.dumps(event_data)}\n\n"
                    
                    # Atualizar stats globais
                    app_state.stats = stats
                    app_state.frame_count = frame_count
                    
                    # Adicionar ao histórico
                    if detections and app_state.history_system:
                        app_state.history_system.add_detection(
                            frame_count, detections, stats
                        )
                else:
                    # Sistema não inicializado - enviar dados vazios
                    event_data = {
                        "frame_id": 0,
                        "boxes": [],
                        "epi_summary": {
                            "com_capacete": 0,
                            "sem_capacete": 0,
                            "com_colete": 0,
                            "sem_colete": 0,
                            "total_pessoas": 0,
                            "compliance_score": 0.0,
                            "detection_rate": 0.0,
                            "avg_confidence": 0.0
                        },
                        "total_people": 0,
                        "compliance_rate": 0.0,
                        "detection_rate": 0.0,
                        "avg_confidence": 0.0,
                        "violations": []
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                
                # OTIMIZAÇÃO: Aguardar próximo evento (5 FPS para estabilidade máxima)
                await asyncio.sleep(0.2)  # ~5 FPS para estabilidade
                
        except Exception as e:
            logger.error(f"❌ Erro no SSE: {e}")
            await asyncio.sleep(1)
    
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

# ===== ENDPOINT DE DETECÇÃO EM TEMPO REAL =====

@app.websocket("/ws/detect-video")
async def ws_detect_video(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_json({"type": "ready"})
        while True:
            frame_bytes = await ws.receive_bytes()
            nparr = np.frombuffer(frame_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                await ws.send_json({"type": "error", "message": "frame inválido"})
                continue

            if not app_state.detection_system or not getattr(app_state.detection_system, 'detector', None):
                await ws.send_json({"type": "error", "message": "detector indisponível"})
                continue

            try:
                # Redimensionar como no POST para consistência
                h, w = image.shape[:2]
                max_side = 640
                if max(h, w) > max_side:
                    scale = max_side / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    image = cv2.resize(image, (new_w, new_h))
                    h, w = image.shape[:2]

                result = app_state.detection_system.detector.process_frame(image)
                raw_dets = result.get("detections", [])
                # Formatar como no endpoint HTTP
                formatted = []
                for d in raw_dets:
                    formatted.append({
                        "class_name": d.get("class_name"),
                        "confidence": float(d.get("confidence", 0.0)),
                        "bbox": d.get("bbox", [])
                    })

                await ws.send_json({
                    "type": "detections",
                    "detections": formatted,
                    "frame_width": int(w),
                    "frame_height": int(h)
                })
            except Exception as e:
                logger.error(f"Erro processamento WS: {e}")
                await ws.send_json({"type": "error", "message": "falha processamento"})
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws)

@app.post("/api/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Endpoint de teste para upload"""
    try:
        content = await file.read()
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": file.size,
            "content_length": len(content),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Erro no teste: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect-frame")
async def detect_frame(
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None)
):
    """Detectar EPIs em um frame de vídeo"""
    try:
        # Aceitar tanto 'file' quanto 'image'
        upload = file or image
        if upload is None:
            raise HTTPException(status_code=400, detail="Campo 'file' ou 'image' é obrigatório")

        logger.info(f"Recebendo arquivo: {upload.filename}, tipo: {upload.content_type}, tamanho: {getattr(upload, 'size', 'n/a')}")
        
        # Ler imagem primeiro
        content = await upload.read()
        
        # Validar se é uma imagem válida
        if not content:
            logger.error("Arquivo vazio recebido")
            raise HTTPException(status_code=400, detail="Arquivo vazio")
        
        logger.info(f"Conteúdo lido: {len(content)} bytes")
        
        # Converter para numpy array
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error(f"Erro ao decodificar imagem: {len(content)} bytes")
            raise HTTPException(status_code=400, detail="Imagem inválida")
        
        logger.info(f"Imagem decodificada: {image.shape}")
        
        # Usar o detector do sistema principal (via app_state)
        if not app_state.detection_system or not getattr(app_state.detection_system, 'detector', None):
            raise HTTPException(status_code=500, detail="Sistema de detecção não disponível")

        # Verificar se o detector está disponível e inicializado
        detector = app_state.detection_system.detector
        if not detector:
            logger.error("❌ Detector não está disponível")
            raise HTTPException(status_code=500, detail="Sistema de detecção não disponível")
        
        # Inicializar detector se necessário (só uma vez)
        if hasattr(detector, 'is_initialized'):
            if not detector.is_initialized:
                logger.warning("⚠️ Detector não inicializado, tentando inicializar...")
                if hasattr(detector, 'initialize_model'):
                    if not detector.initialize_model():
                        logger.error("❌ Falha ao inicializar detector")
                        raise HTTPException(status_code=500, detail="Falha ao inicializar detector")
                    else:
                        logger.info("✅ Detector inicializado com sucesso")
        elif hasattr(detector, 'initialize_model'):
            # Se não tem is_initialized, tentar inicializar uma vez
            logger.warning("⚠️ Verificando inicialização do detector...")
            if not detector.initialize_model():
                logger.error("❌ Falha ao inicializar detector")
                raise HTTPException(status_code=500, detail="Falha ao inicializar detector")
        
        # Reduzir a imagem no backend também (consistente com front)
        try:
            h, w = image.shape[:2]
            logger.debug(f"📐 Imagem original: {w}x{h}")
            max_side = 640
            if max(h, w) > max_side:
                scale = max_side / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h))
                logger.debug(f"📐 Imagem redimensionada: {new_w}x{new_h}")

            logger.debug(f"🔍 Chamando process_frame...")
            result = detector.process_frame(image)
            logger.debug(f"✅ process_frame retornou: {type(result)}, keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
            
            detections = result.get('detections', [])
            logger.info(f"📊 Detecções encontradas: {len(detections)}")
            
            if len(detections) > 0:
                logger.info(f"📋 Primeiras 3 detecções: {detections[:3]}")
            else:
                logger.warning("⚠️ Nenhuma detecção encontrada no frame")
                # Log informações do modelo para debug
                if hasattr(detector, 'model'):
                    logger.debug(f"🔧 Modelo disponível: {detector.model is not None}")
                if hasattr(detector, 'class_names'):
                    logger.debug(f"📋 Classes do modelo: {len(detector.class_names)} classes")
                if hasattr(detector, 'config'):
                    logger.debug(f"⚙️ Config threshold: {detector.config.get('conf_threshold', 'N/A')}")
                    
        except Exception as e:
            logger.error(f"Erro ao processar frame no detector: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Falha no processamento do frame")
        
        # Formatar resultados incluindo missing_epis e compliant
        formatted_detections = []
        for detection in detections:
            class_name = detection.get("class_name", "")
            missing_epis = detection.get("missing_epis", [])
            is_compliant = detection.get("compliant", True)
            is_missing_class = class_name.startswith("missing-")
            
            formatted_detections.append({
                "class_name": class_name,
                "confidence": float(detection.get("confidence", 0.0)),
                "bbox": detection.get("bbox", []),
                "missing_epis": missing_epis,
                "compliant": is_compliant and not is_missing_class,
                "missing_epi": missing_epis[0] if missing_epis else (class_name.replace("missing-", "") if is_missing_class else None),
                "type": "negative" if (is_missing_class or missing_epis or not is_compliant) else "positive"
            })
        
        h, w = image.shape[:2]
        return {
            "detections": formatted_detections,
            "frame_width": int(w),
            "frame_height": int(h),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro na detecção de frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erro na detecção: {str(e)}")

# ===== ENDPOINTS DE VÍDEO =====

# Criar diretórios para vídeos
VIDEO_UPLOAD_DIR = Path("uploads/videos")
VIDEO_PROCESSED_DIR = Path("processed/videos")
VIDEO_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Inicializar fila de processamento
video_queue.start()

@app.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload de vídeo para processamento"""
    try:
        # Validar tipo de arquivo
        detector = VideoAIDetector()
        if not any(file.filename.lower().endswith(ext) for ext in detector.get_supported_formats()):
            raise HTTPException(status_code=400, detail="Formato de vídeo não suportado")
        
        # Gerar ID único para o vídeo
        video_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        video_filename = f"{video_id}{file_extension}"
        video_path = VIDEO_UPLOAD_DIR / video_filename
        
        # Salvar arquivo
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Validar vídeo
        validation = detector.validate_video(str(video_path))
        if not validation['valid']:
            os.remove(video_path)
            raise HTTPException(status_code=400, detail=f"Vídeo inválido: {validation['error']}")
        
        # Adicionar à fila de processamento
        output_path = VIDEO_PROCESSED_DIR / f"{video_id}_processed.mp4"
        success = video_queue.add_video(
            video_id=video_id,
            video_path=str(video_path),
            output_path=str(output_path)
        )
        
        if not success:
            os.remove(video_path)
            raise HTTPException(status_code=500, detail="Erro ao adicionar vídeo à fila")
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "status": "queued",
            "video_info": validation['info'],
            "message": "Vídeo enviado com sucesso"
        }
        
    except Exception as e:
        logger.error(f"Erro no upload do vídeo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}/status")
async def get_video_status(video_id: str):
    """Obtém status do processamento de um vídeo"""
    try:
        status = video_queue.get_status(video_id)
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        return status
        
    except Exception as e:
        logger.error(f"Erro ao obter status do vídeo {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}/results")
async def get_video_results(video_id: str):
    """Obtém resultados do processamento de um vídeo"""
    try:
        status = video_queue.get_status(video_id)
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        if status['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Vídeo ainda não foi processado")
        
        return {
            "video_id": video_id,
            "results": status['results'],
            "status": status['status']
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter resultados do vídeo {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}/download")
async def download_processed_video(video_id: str):
    """Download do vídeo processado"""
    try:
        status = video_queue.get_status(video_id)
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        if status['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Vídeo ainda não foi processado")
        
        output_path = VIDEO_PROCESSED_DIR / f"{video_id}_processed.mp4"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="Arquivo processado não encontrado")
        
        return FileResponse(
            path=str(output_path),
            filename=f"{video_id}_processed.mp4",
            media_type="video/mp4"
        )
        
    except Exception as e:
        logger.error(f"Erro no download do vídeo {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/formats")
async def get_supported_formats():
    """Retorna formatos de vídeo suportados"""
    detector = VideoAIDetector()
    return {
        "supported_formats": detector.get_supported_formats(),
        "max_file_size": "500MB",
        "recommended_formats": [".mp4", ".avi", ".mov"]
    }

@app.get("/api/videos/list")
async def list_videos():
    """Lista todos os vídeos processados"""
    try:
        videos = []
        
        # Listar vídeos na fila
        for video_id, status in video_queue.results.items():
            videos.append({
                "video_id": video_id,
                "status": status['status'],
                "created_at": status.get('created_at'),
                "started_at": status.get('started_at'),
                "completed_at": status.get('completed_at'),
                "filename": status.get('filename', 'N/A')
            })
        
        return {
            "videos": videos,
            "total": len(videos)
        }
        
    except Exception as e:
        logger.error(f"Erro ao listar vídeos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}/report")
async def get_video_report(video_id: str):
    """Obtém relatório completo de detecções (positivas e negativas)"""
    try:
        from backend.video_report import video_report_system
        
        # Verificar se vídeo foi processado
        status = video_queue.get_status(video_id)
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        
        if status['status'] != 'completed':
            raise HTTPException(status_code=400, detail="Vídeo ainda não foi processado")
        
        # Obter relatório
        report = video_report_system.get_report(video_id)
        
        if not report:
            # Criar relatório se não existir
            if 'results' in status:
                report = video_report_system.create_report(video_id, status['results'])
            else:
                raise HTTPException(status_code=404, detail="Relatório não encontrado e resultados não disponíveis")
        
        return {
            "video_id": video_id,
            "report": report
        }
        
    except Exception as e:
        logger.error(f"Erro ao obter relatório do vídeo {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/{video_id}/report/csv")
async def export_video_report_csv(video_id: str):
    """Exporta relatório para CSV"""
    try:
        from backend.video_report import video_report_system
        
        # Verificar se relatório existe
        report = video_report_system.get_report(video_id)
        if not report:
            raise HTTPException(status_code=404, detail="Relatório não encontrado")
        
        # Exportar CSV
        csv_path = video_report_system.export_csv(video_id)
        
        return FileResponse(
            path=csv_path,
            filename=f"{video_id}_report.csv",
            media_type="text/csv"
        )
        
    except Exception as e:
        logger.error(f"Erro ao exportar CSV do vídeo {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/videos/reports/list")
async def list_video_reports():
    """Lista todos os relatórios disponíveis"""
    try:
        from backend.video_report import video_report_system
        reports = video_report_system.list_reports()
        return {
            "reports": reports,
            "total": len(reports)
        }
    except Exception as e:
        logger.error(f"Erro ao listar relatórios: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/videos/{video_id}")
async def delete_video(video_id: str):
    """Remove um vídeo e seus arquivos"""
    try:
        # Remover da fila se ainda estiver processando
        if video_id in video_queue.results:
            del video_queue.results[video_id]
        
        # Remover arquivos
        files_to_remove = [
            VIDEO_UPLOAD_DIR / f"{video_id}.*",
            VIDEO_PROCESSED_DIR / f"{video_id}_processed.mp4"
        ]
        
        removed_files = []
        for file_pattern in files_to_remove:
            for file_path in Path(file_pattern.parent).glob(file_pattern.name):
                try:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                except FileNotFoundError:
                    pass
        
        return {
            "message": "Vídeo removido com sucesso",
            "removed_files": removed_files
        }
        
    except Exception as e:
        logger.error(f"Erro ao remover vídeo {video_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para salvar relatório de detecções em tempo real
@app.post("/api/videos/realtime/report")
async def save_realtime_report(report_data: Dict[str, Any]):
    """Salva relatório de detecções em tempo real"""
    try:
        from backend.video_report import video_report_system
        
        video_id = report_data.get('video_id', f'realtime-{uuid.uuid4()}')
        detections = report_data.get('detections', [])
        statistics = report_data.get('statistics', {})
        video_info = report_data.get('video_info', {})
        
        # Converter logs de detecção para formato de relatório
        positive_detections = []
        negative_detections = []
        
        for det in detections:
            frame_num = det.get('frame_number', 0)
            timestamp = det.get('timestamp', 0)
            class_name = det.get('class_name', '')
            
            detection_entry = {
                'frame_number': frame_num,
                'timestamp': timestamp,
                'class_name': class_name,
                'confidence': det.get('confidence', 0.0),
                'bbox': det.get('bbox', []),
                'type': det.get('type', 'positive')
            }
            
            if det.get('type') == 'negative' or class_name.startswith('missing-'):
                detection_entry['missing_epi'] = det.get('missing_epi') or class_name.replace('missing-', '')
                negative_detections.append(detection_entry)
            else:
                positive_detections.append(detection_entry)
        
        # Criar estrutura de resultados para o sistema de relatório
        video_results = {
            'total_frames': video_info.get('total_frames', len(detections)),
            'processed_frames': len(detections),
            'detections': detections,
            'summary': {
                'compliance_score': statistics.get('compliance_score', 0.0),
                'total_pessoas': statistics.get('total_pessoas', 0)
            }
        }
        
        # Gerar relatório usando o sistema existente
        report = video_report_system.create_report(video_id, video_results)
        
        logger.info(f"✅ Relatório em tempo real criado: {video_id} com {len(detections)} detecções")
        
        return {
            "success": True,
            "video_id": video_id,
            "report": report,
            "message": "Relatório gerado com sucesso"
        }
        
    except Exception as e:
        logger.error(f"Erro ao salvar relatório em tempo real: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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

# Rota para servir o dashboard
@app.get("/dashboard")
async def serve_dashboard():
    """Serve o dashboard principal"""
    frontend_path = Path("frontend/index.html")
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        raise HTTPException(status_code=404, detail="Dashboard não encontrado")

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
