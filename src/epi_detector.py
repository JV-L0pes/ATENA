"""
Sistema de Detecção de EPIs usando YOLOv5
Integração simplificada para a API
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import threading
from queue import Queue
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EPIDetector:
    """Detector de EPIs usando YOLOv5"""
    
    def __init__(self, model_path: str = "yolov5/yolov5n.pt", conf_thresh: float = 0.35, iou_thresh: float = 0.45, 
                 force_cpu_only: bool = False, device_preference: str = "auto"):
        self.model_path = Path(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.force_cpu_only = force_cpu_only
        self.device_preference = device_preference
        
        # Estado do sistema
        self.model = None
        self.device = None
        self.is_initialized = False
        
        # Threading
        self.detection_thread = None
        self.running = False
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        
        # Cache de resultados
        self.current_detections = []
        self.current_frame = None
        self.frame_count = 0
        
        # Estatísticas
        self.stats = {
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0
        }
        
        # Configurações
        self.config = {
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "max_detections": 50,
            "enable_tracking": True
        }
        
        logger.info("EPI Detector inicializado")
    
    def initialize_model(self):
        """Inicializa o modelo YOLOv5"""
        try:
            logger.info("Carregando modelo YOLOv5...")
            
            # Determinar dispositivo baseado nas configurações
            self.device = self._select_device()
            logger.info(f"Usando dispositivo: {self.device}")
            
            # Tentar carregar modelo com dispositivo selecionado
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=str(self.model_path), 
                                          device=self.device,
                                          force_reload=True)
            except Exception as device_error:
                logger.warning(f"Erro ao carregar com dispositivo {self.device}: {device_error}")
                logger.info("Tentando carregar com CPU...")
                
                # Fallback para CPU se houver erro com GPU
                self.device = torch.device("cpu")
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=str(self.model_path), 
                                          device=self.device,
                                          force_reload=True)
                logger.info("Modelo carregado com CPU (fallback)")
            
            # Configurar modelo
            self.model.conf = self.conf_thresh
            self.model.iou = self.iou_thresh
            self.model.max_det = self.config["max_detections"]
            
            self.is_initialized = True
            logger.info(f"Modelo YOLOv5 carregado com sucesso no dispositivo: {self.device}")
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def _select_device(self):
        """Seleciona o dispositivo baseado nas configurações"""
        if self.force_cpu_only:
            logger.info("Forçando uso de CPU (FORCE_CPU_ONLY=True)")
            return torch.device("cpu")
        
        if self.device_preference == "cpu":
            logger.info("Usando CPU (preferência do usuário)")
            return torch.device("cpu")
        elif self.device_preference == "cuda":
            if torch.cuda.is_available():
                logger.info("Usando GPU CUDA (preferência do usuário)")
                return torch.device("cuda")
            else:
                logger.warning("GPU CUDA solicitada mas não disponível, usando CPU")
                return torch.device("cpu")
        else:  # auto
            if torch.cuda.is_available():
                logger.info("Usando GPU CUDA (detecção automática)")
                return torch.device("cuda")
            else:
                logger.info("Usando CPU (CUDA não disponível)")
                return torch.device("cpu")
    
    def start_detection_thread(self):
        """Inicia thread de detecção"""
        if self.detection_thread is None or not self.detection_thread.is_alive():
            self.running = True
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            logger.info("Thread de detecção iniciada")
    
    def stop_detection_thread(self):
        """Para thread de detecção"""
        self.running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)
            logger.info("Thread de detecção parada")
    
    def _detection_loop(self):
        """Loop principal de detecção"""
        while self.running:
            try:
                # Processar frames da fila
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    detections = self._process_frame(frame)
                    
                    # Atualizar resultados
                    self.current_detections = detections
                    self.current_frame = frame
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
                
                time.sleep(0.01)  # 100 FPS
                
            except Exception as e:
                logger.error(f"Erro no loop de detecção: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Processa um frame e retorna detecções"""
        if not self.is_initialized or self.model is None:
            return []
        
        try:
            # Executar inferência
            results = self.model(frame)
            
            # Processar resultados
            detections = []
            
            if len(results.xyxy) > 0:
                boxes = results.xyxy[0].cpu().numpy()
                labels = results.names
                
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    
                    if conf >= self.conf_thresh:
                        # Converter coordenadas
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Obter label
                        label = labels[int(cls)]
                        
                        # Criar detecção
                        detection = {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "label": label,
                            "conf": float(conf),
                            "track_id": None  # Implementar tracking se necessário
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Erro ao processar frame: {e}")
            return []
    
    def _update_stats(self, detections: List[Dict]):
        """Atualiza estatísticas baseado nas detecções"""
        # Resetar contadores
        self.stats = {
            "com_capacete": 0,
            "sem_capacete": 0,
            "com_colete": 0,
            "sem_colete": 0
        }
        
        # Contar EPIs detectados
        for detection in detections:
            label = detection["label"].lower()
            
            if label == "helmet":
                self.stats["com_capacete"] += 1
            elif label == "vest":
                self.stats["com_colete"] += 1
        
        # Calcular pessoas sem EPI (simplificado)
        # Em uma implementação real, seria necessário associar pessoas com EPIs
        people_count = sum(1 for d in detections if d["label"].lower() == "person")
        
        if people_count > 0:
            # Assumir que pessoas sem capacete detectado estão sem capacete
            self.stats["sem_capacete"] = max(0, people_count - self.stats["com_capacete"])
            self.stats["sem_colete"] = max(0, people_count - self.stats["com_colete"])
    
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
    
    def add_detection_result(self, frame: np.ndarray, detections: List[Dict], epi_summary: Dict[str, int]):
        """Adiciona resultado de detecção"""
        try:
            self.current_frame = frame.copy()
            self.current_detections = detections.copy()
            
            # Atualizar estatísticas
            self.stats.update(epi_summary)
            
            # Adicionar ao histórico se sistema disponível
            if hasattr(self, 'history_system') and self.history_system:
                self.history_system.add_detection(self.frame_count, detections, epi_summary)
            
            self.frame_count += 1
            
        except Exception as e:
            logger.error(f"Erro ao adicionar resultado de detecção: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        """Retorna estatísticas atuais"""
        return self.stats.copy()
    
    def update_config(self, config: Dict[str, Any]):
        """Atualiza configurações"""
        # Verificar se as configurações de dispositivo mudaram
        old_force_cpu = self.force_cpu_only
        old_device_pref = self.device_preference
        
        # Atualizar configurações
        self.config.update(config)
        
        # Atualizar configurações de dispositivo
        self.force_cpu_only = config.get("force_cpu_only", self.force_cpu_only)
        self.device_preference = config.get("device_preference", self.device_preference)
        
        # Verificar se precisa reinicializar o modelo
        device_changed = (old_force_cpu != self.force_cpu_only or 
                         old_device_pref != self.device_preference)
        
        if device_changed:
            logger.info(f"Configurações de dispositivo alteradas: {old_device_pref} -> {self.device_preference}, CPU only: {old_force_cpu} -> {self.force_cpu_only}")
            
            # Parar detecção se estiver rodando
            was_running = self.running
            if was_running:
                self.stop_detection_thread()
            
            # Reinicializar modelo com novo dispositivo
            try:
                self.initialize_model()
                logger.info(f"Modelo reinicializado com dispositivo: {self.device}")
            except Exception as e:
                logger.error(f"Erro ao reinicializar modelo: {e}")
                # Restaurar configurações antigas em caso de erro
                self.force_cpu_only = old_force_cpu
                self.device_preference = old_device_pref
                raise
        
        # Atualizar configurações do modelo
        if self.model is not None:
            self.model.conf = config.get("conf_thresh", self.conf_thresh)
            self.model.iou = config.get("iou_thresh", self.iou_thresh)
            self.model.max_det = config.get("max_detections", 50)
        
        logger.info("Configurações atualizadas com sucesso")
    
    def cleanup(self):
        """Cleanup do sistema"""
        self.stop_detection_thread()
        logger.info("EPI Detector finalizado")

class EPIDetectionSystem:
    """Sistema principal de detecção de EPIs"""
    
    def __init__(self):
        self.detector = None
        self.video_source = None
        self.running = False
        
        # Threading
        self.video_thread = None
        self.running = False
        
        logger.info("Sistema de detecção de EPIs inicializado")
    
    def initialize_system(self, model_path: str = "yolov5/yolov5n.pt", video_source: int = 0, 
                         force_cpu_only: bool = False, device_preference: str = "auto"):
        """Inicializa o sistema completo"""
        try:
            # Inicializar detector
            self.detector = EPIDetector(model_path, force_cpu_only=force_cpu_only, 
                                      device_preference=device_preference)
            self.detector.initialize_model()
            
            # Configurar fonte de vídeo
            self.video_source = video_source
            
            # Iniciar threads
            self.detector.start_detection_thread()
            self.start_video_thread()
            
            self.running = True
            logger.info("Sistema de detecção inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar sistema: {e}")
            raise
    
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
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2)
            logger.info("Thread de vídeo parada")
    

    
    def _video_loop(self):
        """Loop principal de captura de vídeo"""
        try:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                logger.error(f"Câmera {self.video_source} não disponível")
                return
        except Exception as e:
            logger.error(f"Erro ao abrir câmera: {e}")
            return
        
        if not cap.isOpened():
            logger.error("Não foi possível abrir fonte de vídeo")
            return
        
        try:
            while self.running:
                ret, frame = cap.read()
                if ret:
                    # Adicionar frame para processamento
                    if self.detector:
                        self.detector.add_frame(frame)
                    
                    # Aguardar próximo frame
                    time.sleep(1/30)  # 30 FPS
                else:
                    logger.warning("Erro ao ler frame")
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Erro no loop de vídeo: {e}")
        finally:
            cap.release()
            logger.info("Captura de vídeo finalizada")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Retorna frame atual"""
        if self.detector:
            return self.detector.get_current_frame()
        return None
    
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

# Função de teste
def test_detector():
    """Função para testar o detector"""
    try:
        # Inicializar sistema
        system = EPIDetectionSystem()
        system.initialize_system()
        
        print("Sistema inicializado. Pressione Ctrl+C para parar...")
        
        # Loop de teste
        while True:
            detections = system.get_current_detections()
            stats = system.get_stats()
            
            if detections:
                print(f"Detecções: {len(detections)}, Stats: {stats}")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nParando sistema...")
        system.cleanup()
    except Exception as e:
        print(f"Erro: {e}")
        system.cleanup()

if __name__ == "__main__":
    test_detector()
