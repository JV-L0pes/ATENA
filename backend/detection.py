"""
Módulo de Detecção de EPIs Otimizado
Sistema simplificado para detecção de capacetes e coletes
"""

import cv2
import numpy as np
import threading
import time
import queue
import logging
from typing import List, Dict, Optional, Any
import torch
import sys
import os

# Adicionar yolov5 ao path
yolov5_path = os.path.join(os.path.dirname(__file__), '..', 'yolov5')
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

from .config import CONFIG

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EPIDetector:
    """Detector de EPIs simplificado usando YOLOv5"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or CONFIG.MODEL_PATH
        self.model = None
        
        # Configurações de detecção
        model_config = CONFIG.get_model_config()
        self.conf_thresh = model_config.get("conf_thresh", 0.35)
        self.iou_thresh = model_config.get("iou_thresh", 0.45)
        self.max_detections = model_config.get("max_detections", 50)
        
        # Estado do sistema
        self.is_initialized = False
        self.is_running = False
        self.frame_count = 0
        
        # Threading
        self.detection_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Armazenamento
        self.current_frame = None
        self.current_detections = []
        
        # Estatísticas básicas
        self.stats = {
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0,
            "total_pessoas": 0
        }
        
        logger.info("EPI Detector inicializado")
    
    def initialize_model(self):
        """Inicializa o modelo YOLOv5"""
        try:
            logger.info(f"Carregando modelo: {self.model_path}")
            
            # Carregar modelo YOLOv5 customizado
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                      path=self.model_path, 
                                      force_reload=False)
            
            # Configurar modelo
            self.model.conf = self.conf_thresh
            self.model.iou = self.iou_thresh
            self.model.max_det = self.max_detections
            
            # Configurar para inferência
            self.model.eval()
            
            # Warmup do modelo
            self._warmup_model()
            
            self.is_initialized = True
            logger.info("Modelo YOLOv5 carregado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _warmup_model(self):
        """Executa warmup do modelo para otimizar primeira inferência"""
        try:
            logger.info("Executando warmup do modelo...")
            dummy_input = torch.randn(1, 3, 640, 640)
            with torch.no_grad():
                _ = self.model(dummy_input)
            logger.info("Warmup concluído")
        except Exception as e:
            logger.warning(f"Warmup falhou: {e}")
    
    def start_detection_thread(self):
        """Inicia thread de detecção"""
        if not self.is_initialized:
            raise RuntimeError("Modelo não inicializado")
        
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            logger.info("Thread de detecção iniciada")
    
    def stop_detection_thread(self):
        """Para thread de detecção"""
        self.is_running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
            logger.info("Thread de detecção parada")
    
    def _detection_loop(self):
        """Loop principal de detecção"""
        logger.info("Loop de detecção iniciado")
        
        while self.is_running:
            try:
                # Aguardar frame
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Processar frame
                    detections = self._process_frame(frame)
                    
                    # Atualizar estado
                    self.current_frame = frame
                    self.current_detections = detections
                    self.frame_count += 1
                    
                    # Atualizar estatísticas
                    self._update_stats(detections)
                    
                    # Colocar resultado na fila
                    if not self.result_queue.full():
                        self.result_queue.put({
                            "frame_id": self.frame_count,
                            "detections": detections,
                            "stats": self.stats.copy()
                        })
                
                time.sleep(0.01)  # Pequena pausa para não sobrecarregar CPU
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erro no loop de detecção: {e}")
                time.sleep(0.1)
        
        logger.info("Loop de detecção finalizado")
    
    def _process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Processa frame e retorna detecções"""
        try:
            # Pré-processamento
            processed_frame = self._preprocess_frame(frame)
            
            # Inferência
            with torch.no_grad():
                results = self.model(processed_frame)
            
            # Extrair detecções
            detections = self._extract_detections(results, frame.shape)
            
            return detections
            
        except Exception as e:
            logger.error(f"Erro ao processar frame: {e}")
            return []
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Pré-processa frame para inferência"""
        # O YOLOv5 faz o pré-processamento automaticamente
        return frame
    
    def _extract_detections(self, results, original_shape) -> List[Dict]:
        """Extrai detecções com máxima precisão"""
        detections = []
        
        # YOLOv5 retorna resultados em formato específico
        if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            boxes = results.xyxy[0].cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                
                if conf >= self.conf_thresh:
                    # Converter coordenadas para formato original
                    x, y, w, h = self._convert_coordinates(x1, y1, x2, y2, original_shape)
                    
                    # Obter label (YOLOv5 usa índices numéricos)
                    label = self._get_label_name(int(cls))
                    
                    # Criar detecção
                    detection = {
                        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                        "label": label, "conf": float(conf)
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def _convert_coordinates(self, x1: float, y1: float, x2: float, y2: float, original_shape) -> tuple:
        """Converte coordenadas do modelo para formato original"""
        h, w = original_shape[:2]
        
        # Converter de (x1, y1, x2, y2) para (x, y, w, h)
        x = int(x1 * w / 640)  # 640 é o tamanho padrão do YOLOv5
        y = int(y1 * h / 640)
        w = int((x2 - x1) * w / 640)
        h = int((y2 - y1) * h / 640)
        
        return x, y, w, h
    
    def _get_label_name(self, class_index: int) -> str:
        """Mapeia índice da classe para nome"""
        # Mapeamento baseado no dataset treinado
        label_mapping = {
            0: "helmet",
            1: "no-helmet", 
            2: "no-vest",
            3: "person",
            4: "vest"
        }
        return label_mapping.get(class_index, f"class_{class_index}")
    
    def _update_stats(self, detections: List[Dict]):
        """Atualiza estatísticas básicas"""
        # Resetar contadores
        self.stats = {
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0,
            "total_pessoas": 0
        }
        
        # Contar detecções
        for detection in detections:
            label = detection["label"].lower()
            
            if label == "helmet":
                self.stats["com_capacete"] += 1
            elif label == "no-helmet":
                self.stats["sem_capacete"] += 1
            elif label == "vest":
                self.stats["com_colete"] += 1
            elif label == "no-vest":
                self.stats["sem_colete"] += 1
            elif label == "person":
                self.stats["total_pessoas"] += 1
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Retorna resumo das detecções"""
        return {
            "frame_id": self.frame_count,
            "detections": self.current_detections,
            "epi_summary": self.stats
        }
    
    # Métodos de compatibilidade
    def add_frame(self, frame: np.ndarray):
        """Adiciona frame para processamento"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def get_current_detections(self) -> List[Dict]:
        """Retorna detecções atuais"""
        return self.current_detections
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Retorna frame atual"""
        return self.current_frame
    
    def get_stats(self) -> Dict[str, int]:
        """Retorna estatísticas atuais"""
        return self.stats.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """Atualiza configurações"""
        if "conf_thresh" in config:
            self.conf_thresh = config["conf_thresh"]
        if "iou_thresh" in config:
            self.iou_thresh = config["iou_thresh"]
        if "max_detections" in config:
            self.max_detections = config["max_detections"]
        
        # Atualizar modelo se necessário
        if self.model is not None:
            self.model.conf = self.conf_thresh
            self.model.iou = self.iou_thresh
            self.model.max_det = self.max_detections
        
        logger.info("Configurações atualizadas")
    
    def cleanup(self):
        """Cleanup do sistema"""
        self.stop_detection_thread()
        logger.info("EPI Detector finalizado")

# Manter compatibilidade com código existente
EPIDetectorOptimized = EPIDetector

class EPIDetectionSystem:
    """Sistema principal de detecção de EPIs"""
    
    def __init__(self):
        self.detector = None
        self.video_source = None
        self.is_running = False
        self.video_capture = None
        
        # Threading
        self.video_thread = None
        self.running = False
        
        # Armazenar o frame atual para o stream
        self.current_frame = None
        
        logger.info("Sistema de detecção de EPIs inicializado")
    
    def initialize_system(self, model_path: str = None, video_source: int = None):
        """Inicializa o sistema completo"""
        try:
            # Usar configurações padrão se não especificadas
            video_config = CONFIG.get_video_config()
            
            # Inicializar detector
            self.detector = EPIDetector(model_path)
            self.detector.initialize_model()
            
            # Configurar fonte de vídeo
            self.video_source = video_source or video_config["source"]
            
            # Testar conexão com a webcam
            self._test_video_source()
            
            # Iniciar threads
            self.detector.start_detection_thread()
            self.start_video_thread()
            
            self.is_running = True
            logger.info("Sistema de detecção inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar sistema: {e}")
            raise
    
    def _test_video_source(self):
        """Testa se a fonte de vídeo está disponível"""
        try:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                logger.warning(f"Webcam {self.video_source} não disponível, tentando webcam padrão...")
                # Tentar webcam padrão
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    logger.warning("Webcam padrão também não disponível")
                else:
                    self.video_source = 0
                    logger.info("Usando webcam padrão")
            
            if cap.isOpened():
                # Testar leitura de um frame
                ret, frame = cap.read()
                if ret:
                    logger.info(f"Webcam {self.video_source} funcionando - Resolução: {frame.shape[1]}x{frame.shape[0]}")
                else:
                    logger.warning("Webcam não consegue ler frames")
                cap.release()
            else:
                logger.error("Nenhuma webcam disponível")
                
        except Exception as e:
            logger.error(f"Erro ao testar fonte de vídeo: {e}")
    
    def start_video_thread(self):
        """Inicia thread de captura de vídeo"""
        if self.video_thread is None or not self.video_thread.is_alive():
            self.running = True
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()
            logger.info("Thread de vídeo iniciada")
    
    def stop_video_thread(self):
        """Para thread de vídeo"""
        self.running = False
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)
            logger.info("Thread de vídeo parada")
    
    def _video_loop(self):
        """Loop principal de captura de vídeo"""
        try:
            self.video_capture = cv2.VideoCapture(self.video_source)
            
            if not self.video_capture.isOpened():
                logger.error(f"Não foi possível abrir fonte de vídeo {self.video_source}")
                return
            
            # Configurar propriedades da webcam
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.VIDEO_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.VIDEO_HEIGHT)
            self.video_capture.set(cv2.CAP_PROP_FPS, CONFIG.VIDEO_FPS)
            
            logger.info(f"Captura de vídeo iniciada - Fonte: {self.video_source}, FPS: {CONFIG.VIDEO_FPS}")
            
            while self.running:
                ret, frame = self.video_capture.read()
                if ret:
                    # Armazenar frame atual para o stream
                    self.current_frame = frame.copy()
                    
                    # Adicionar frame para processamento de detecção
                    if self.detector:
                        self.detector.add_frame(frame)
                    
                    # Aguardar próximo frame
                    time.sleep(1/CONFIG.VIDEO_FPS)
                else:
                    logger.warning("Erro ao ler frame da webcam")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Erro no loop de vídeo: {e}")
        finally:
            if self.video_capture and self.video_capture.isOpened():
                self.video_capture.release()
            logger.info("Captura de vídeo finalizada")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Retorna frame atual"""
        return self.current_frame
    
    def get_current_detections(self) -> List[Dict]:
        """Retorna detecções atuais"""
        if self.detector:
            return self.detector.get_current_detections()
        return []
    
    def get_stats(self) -> Dict[str, int]:
        """Retorna estatísticas atuais"""
        if self.detector:
            return self.detector.get_stats()
        return {}
    
    def update_config(self, config: Dict[str, Any]):
        """Atualiza configurações"""
        if self.detector:
            self.detector.update_config(config)
    
    def cleanup(self):
        """Cleanup do sistema"""
        self.stop_video_thread()
        if self.detector:
            self.detector.cleanup()
        logger.info("Sistema de detecção finalizado")
