#!/usr/bin/env python3
"""
ATHENA - Sistema de Detecção em Tempo Real Otimizado
===================================================
Versão ultra-otimizada para detecção em tempo real sem travamentos
"""

import cv2
import numpy as np
import time
import logging
import threading
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO
import torch
from queue import Queue, Empty
from collections import deque

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Importar sistema de recuperação de webcam
import sys
sys.path.append(str(Path(__file__).parent))
try:
    from webcam_recovery import StableWebcamCapture
    logger.info("✅ Sistema de recuperação de webcam disponível")
except ImportError:
    StableWebcamCapture = None
    logger.warning("⚠️ Sistema de recuperação não disponível")

class AthenaRealtimeDetector:
    """Detector ultra-otimizado para tempo real"""
    
    def __init__(self, model_path: str = None, video_source: str = None):
        # Usar modelo da Fase 1 por padrão
        if model_path is None:
            model_path = "athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt"
        
        self.model_path = Path(model_path)
        self.video_source = video_source or os.getenv("RTSP_URL", "0")  # Usar RTSP_URL se disponível
        self.model = None
        self.device = None
        self.is_initialized = False
        
        # Configurações ultra-otimizadas para tempo real
        self.config = {
            'conf_threshold': 0.25,  # Threshold compatível com modelo best.pt
            'iou_threshold': 0.45,
            'max_detections': 100,   # Limite maior para modelo de 17 classes
            'frame_skip': 2,        # Processar apenas 1 a cada 2 frames
            'resize_factor': 0.5,   # Reduzir resolução para velocidade
            'batch_size': 1,        # Processar 1 frame por vez
            'warmup_frames': 5       # Frames de aquecimento
        }
        
        # Classes do modelo best.pt (17 classes)
        self.class_names = [
            'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask-medical', 
            'foot', 'tools', 'glasses', 'gloves', 'helmet', 'hands', 'head', 
            'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
        ]
        
        # Classes principais para EPIs (mapeamento para o modelo)
        self.main_classes = ['person', 'helmet', 'safety-vest', 'gloves', 'glasses']
        
        # Estado do sistema
        self.current_frame = None
        self.processed_frame = None
        self.current_detections = []
        self.frame_count = 0
        self.last_process_time = 0
        self.fps_counter = deque(maxlen=30)
        
        # Threading otimizado
        self.frame_queue = Queue(maxsize=3)  # Fila pequena para evitar acúmulo
        self.detection_thread = None
        self.running = False
        
        # Estatísticas
        self.stats = {
            'total_detections': 0,
            'fps': 0.0,
            'avg_processing_time': 0.0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_frames': 0
        }
        
        logger.info("🚀 Detector ATHENA Tempo Real inicializado")
    
    def initialize_model(self):
        """Inicializa modelo com configurações ultra-otimizadas"""
        try:
            if not self.model_path.exists():
                logger.error(f"❌ Modelo não encontrado: {self.model_path}")
                return False
            
            logger.info(f"🎯 Carregando modelo: {self.model_path}")
            
            # Carregar modelo
            self.model = YOLO(str(self.model_path))
            
            # Verificar se as classes estão corretas
            logger.info(f"📋 Classes carregadas do modelo: {len(self.model.names)}")
            logger.info(f"📋 Primeiras 5 classes: {list(self.model.names.values())[:5]}")
            
            # Atualizar class_names com as classes reais do modelo
            self.class_names = list(self.model.names.values())
            
            # Configurar dispositivo
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info("🚀 Usando GPU CUDA para inferência ultra-rápida")
            else:
                self.device = torch.device('cpu')
                logger.info("💻 Usando CPU")
            
            # Configurar modelo para velocidade máxima
            self.model.to(self.device)
            
            # Configurações de inferência otimizadas
            self.model.overrides = {
                'conf': self.config['conf_threshold'],
                'iou': self.config['iou_threshold'],
                'max_det': self.config['max_detections'],
                'verbose': False,
                'half': True if self.device.type == 'cuda' else False,  # FP16 para GPU
                'device': self.device
            }
            
            # Aquecimento do modelo
            logger.info("🔥 Aquecendo modelo...")
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(self.config['warmup_frames']):
                with torch.no_grad():
                    _ = self.model(dummy_frame, verbose=False)
            
            self.is_initialized = True
            logger.info("✅ Modelo inicializado com configurações ultra-otimizadas!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar modelo: {e}")
            return False
    
    def start_detection(self):
        """Inicia thread de detecção"""
        if not self.is_initialized:
            logger.error("❌ Modelo não inicializado")
            return False
        
        self.running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        logger.info("🚀 Thread de detecção iniciada")
        return True
    
    def _detection_loop(self):
        """Loop principal de detecção ultra-otimizado"""
        logger.info("🔄 Iniciando loop de detecção ultra-otimizado")
        
        while self.running:
            try:
                # Tentar obter frame da fila (não bloqueante)
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except Empty:
                    time.sleep(0.01)
                    continue
                
                # Pular frames para performance
                if self.frame_count % self.config['frame_skip'] != 0:
                    self.stats['frames_skipped'] += 1
                    continue
                
                # Processar frame
                start_time = time.time()
                
                # Reduzir resolução para velocidade
                if self.config['resize_factor'] < 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * self.config['resize_factor']), int(w * self.config['resize_factor'])
                    frame_small = cv2.resize(frame, (new_w, new_h))
                else:
                    frame_small = frame
                
                # Detecção ultra-rápida
                with torch.no_grad():
                    results = self.model(frame_small, verbose=False)
                
                # Processar resultados
                detections = self._process_results(results, frame.shape)
                
                # Escalar coordenadas de volta se necessário
                if self.config['resize_factor'] < 1.0:
                    scale_factor = 1.0 / self.config['resize_factor']
                    for detection in detections:
                        detection['bbox'] = [int(x * scale_factor) for x in detection['bbox']]
                
                # Desenhar resultados
                processed_frame = self._draw_detections(frame, detections)
                
                # Atualizar estado
                self.current_frame = frame.copy()
                self.processed_frame = processed_frame
                self.current_detections = detections
                self.frame_count += 1
                
                # Calcular FPS
                processing_time = time.time() - start_time
                self.fps_counter.append(processing_time)
                
                if len(self.fps_counter) > 0:
                    avg_time = np.mean(self.fps_counter)
                    self.stats['fps'] = 1.0 / avg_time if avg_time > 0 else 0.0
                    self.stats['avg_processing_time'] = avg_time
                
                self.stats['frames_processed'] += 1
                self.stats['total_detections'] += len(detections)
                
                # Log de performance a cada 30 frames
                if self.frame_count % 30 == 0:
                    logger.info(f"📊 FPS: {self.stats['fps']:.1f}, Detecções: {len(detections)}, Tempo: {processing_time*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de detecção: {e}")
                time.sleep(0.01)
    
    def _process_results(self, results, original_shape):
        """Processa resultados do modelo"""
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i, (box, conf, cls) in enumerate(zip(boxes, confidences, class_ids)):
                    if 0 <= int(cls) < len(self.class_names):
                        x1, y1, x2, y2 = box
                        class_name = self.class_names[int(cls)]
                        
                        # Filtrar detecções baseado no threshold de confiança
                        if conf >= self.config['conf_threshold']:
                            detection = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_name': class_name,
                                'class_id': int(cls),
                                'frame_id': self.frame_count,
                                'timestamp': time.time()
                            }
                            detections.append(detection)
                            
                            # Log detalhado para debug
                            if self.frame_count % 30 == 0:  # Log a cada 30 frames
                                logger.info(f"🔍 Detecção: {class_name} (ID: {int(cls)}, conf: {conf:.3f})")
        
        return detections
    
    def _draw_detections(self, frame, detections):
        """Desenha detecções no frame"""
        frame_copy = frame.copy()
        
        # Cores otimizadas para modelo best.pt (17 classes)
        colors = {
            'person': (0, 255, 0),        # Verde para pessoas
            'helmet': (255, 0, 0),        # Vermelho para capacetes
            'safety-vest': (0, 0, 255),    # Azul para coletes de segurança
            'gloves': (255, 165, 0),      # Laranja para luvas
            'glasses': (128, 0, 128),     # Roxo para óculos
            'hands': (255, 192, 203),     # Rosa para mãos
            'head': (255, 255, 0),        # Amarelo para cabeça
            'face': (0, 255, 255),        # Ciano para rosto
            'foot': (165, 42, 42),        # Marrom para pés
            'shoes': (64, 224, 208),      # Turquesa para sapatos
            'tools': (255, 20, 147),      # Rosa choque para ferramentas
            'ear': (50, 205, 50),         # Verde lima para orelhas
            'ear-mufs': (255, 69, 0),     # Vermelho laranja para protetores
            'face-guard': (138, 43, 226), # Azul violeta para protetor facial
            'face-mask-medical': (0, 191, 255), # Azul profundo para máscara médica
            'medical-suit': (255, 105, 180), # Rosa quente para macacão médico
            'safety-suit': (34, 139, 34), # Verde floresta para macacão de segurança
            'other': (255, 255, 255)      # Branco para outros
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Cor baseada na classe
            color = colors.get(class_name, colors['other'])
            
            # Desenhar bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar label (apenas se confiança alta)
            if confidence > 0.6:
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame_copy, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame_copy
    
    def add_frame(self, frame):
        """Adiciona frame para processamento"""
        if not self.running:
            return
        
        # Não bloquear se a fila estiver cheia
        try:
            self.frame_queue.put_nowait(frame)
        except:
            pass  # Ignorar frames se a fila estiver cheia
    
    def get_current_detections(self):
        """Retorna detecções atuais"""
        return self.current_detections.copy()
    
    def get_current_frame(self):
        """Retorna frame processado atual"""
        return self.processed_frame if self.processed_frame is not None else self.current_frame
    
    def get_stats(self):
        """Retorna estatísticas"""
        return self.stats.copy()
    
    def process_frame(self, frame):
        """Processa frame individual (método de compatibilidade)"""
        if not self.is_initialized:
            logger.warning("⚠️ Sistema não inicializado, inicializando agora...")
            if not self.initialize_model():
                return {"detections": [], "processed_frame": frame, "summary": {}}
        
        try:
            # Processar frame diretamente para RTSP
            start_time = time.time()
            
            # Reduzir resolução para velocidade
            if self.config['resize_factor'] < 1.0:
                h, w = frame.shape[:2]
                new_h, new_w = int(h * self.config['resize_factor']), int(w * self.config['resize_factor'])
                frame_small = cv2.resize(frame, (new_w, new_h))
            else:
                frame_small = frame
            
            # Detecção ultra-rápida
            with torch.no_grad():
                results = self.model(frame_small, verbose=False)
            
            # Processar resultados
            detections = self._process_results(results, frame.shape)
            
            # Escalar coordenadas de volta se necessário
            if self.config['resize_factor'] < 1.0:
                scale_factor = 1.0 / self.config['resize_factor']
                for detection in detections:
                    detection['bbox'] = [int(x * scale_factor) for x in detection['bbox']]
            
            # Desenhar resultados
            processed_frame = self._draw_detections(frame, detections)
            
            # Atualizar estado
            self.current_frame = frame.copy()
            self.processed_frame = processed_frame
            self.current_detections = detections
            self.frame_count += 1
            
            # Atualizar estatísticas
            processing_time = time.time() - start_time
            self.stats['avg_processing_time'] = processing_time
            self.stats['total_frames'] += 1
            
            # Log para debug
            if self.frame_count % 30 == 0:
                logger.info(f"🔍 Frame {self.frame_count}: {len(detections)} detecções")
            
            return {
                "detections": detections,
                "processed_frame": processed_frame,
                "summary": {
                    "total_detections": len(detections),
                    "fps": self.stats['fps'],
                    "processing_time": processing_time
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erro ao processar frame: {e}")
            return {"detections": [], "processed_frame": frame, "summary": {}}
    
    def cleanup(self):
        """Cleanup do sistema"""
        self.running = False
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1)
        
        logger.info("🔄 Sistema de detecção finalizado")

# Classe principal para compatibilidade
class AthenaDetectionSystemOptimized:
    """Sistema principal otimizado"""
    
    def __init__(self, model_path: str = None, video_source: str = None):
        self.detector = AthenaRealtimeDetector(model_path, video_source)
        logger.info("🎯 Sistema ATHENA Tempo Real inicializado")
    
    def setup_detector(self):
        """Configura o detector"""
        return self.detector.initialize_model()
    
    def start_detection(self):
        """Inicia detecção"""
        return self.detector.start_detection()
    
    def add_frame(self, frame):
        """Adiciona frame"""
        self.detector.add_frame(frame)
    
    def get_current_detections(self):
        """Retorna detecções"""
        return self.detector.get_current_detections()
    
    def get_current_frame(self):
        """Retorna frame"""
        return self.detector.get_current_frame()
    
    def get_stats(self):
        """Retorna estatísticas"""
        return self.detector.get_stats()
    
    def process_frame(self, frame):
        """Processa frame individual"""
        return self.detector.process_frame(frame)
    
    def cleanup(self):
        """Cleanup"""
        self.detector.cleanup()

# Alias para compatibilidade
AthenaPhase1Detector = AthenaDetectionSystemOptimized
