"""
Módulo de Detecção de IA em Vídeos - Athena Dashboard
Sistema para processar vídeos e detectar EPIs frame por frame
"""

import cv2
import numpy as np
import torch
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
import json
from datetime import datetime
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)

class VideoAIDetector:
    """Detector de IA para processamento de vídeos"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):
        """
        Inicializa o detector de vídeos
        
        Args:
            model_path: Caminho para o modelo YOLOv5
            confidence_threshold: Limite de confiança para detecções [[memory:7420670]]
        """
        self.model_path = model_path or self._get_latest_model()
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configurações de processamento
        self.frame_skip = 1  # Processar todos os frames
        self.max_resolution = 1920  # Resolução máxima para processamento
        self.batch_size = 1  # Processar um frame por vez para vídeos
        
        # Estatísticas
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections_found': 0,
            'processing_time': 0.0,
            'fps': 0.0
        }
        
        logger.info(f"Video AI Detector inicializado com threshold {confidence_threshold}")
    
    def _get_latest_model(self) -> str:
        """Obtém o modelo mais recente disponível"""
        model_paths = [
            "athena_model_latest.pt",
            "athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt"
        ]
        
        for path in model_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError("Nenhum modelo encontrado")
    
    def load_model(self) -> bool:
        """Carrega o modelo YOLOv11"""
        try:
            logger.info(f"Carregando modelo: {self.model_path}")
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            logger.info(f"✅ Modelo YOLOv11 carregado com sucesso")
            logger.info(f"📋 Classes disponíveis: {len(self.model.names)}")
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def detect_in_video(self, video_path: str, output_path: str = None, 
                       progress_callback=None) -> Dict[str, Any]:
        """
        Processa um vídeo completo e detecta EPIs
        
        Args:
            video_path: Caminho para o vídeo de entrada
            output_path: Caminho para salvar o vídeo processado (opcional)
            progress_callback: Função para callback de progresso
            
        Returns:
            Dict com resultados da detecção
        """
        if not self.model:
            if not self.load_model():
                raise RuntimeError("Falha ao carregar modelo")
        
        start_time = time.time()
        results = {
            'video_path': video_path,
            'total_frames': 0,
            'processed_frames': 0,
            'detections': [],
            'summary': {
                'com_capacete': 0,
                'sem_capacete': 0,
                'com_colete': 0,
                'sem_colete': 0,
                'total_pessoas': 0,
                'compliance_score': 0.0
            },
            'processing_time': 0.0,
            'fps': 0.0,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Abrir vídeo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
            
            # Obter informações do vídeo
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            results['total_frames'] = total_frames
            results['video_info'] = {
                'fps': fps,
                'width': width,
                'height': height,
                'duration': total_frames / fps if fps > 0 else 0
            }
            
            logger.info(f"Processando vídeo: {width}x{height}, {total_frames} frames, {fps:.2f} FPS")
            
            # Configurar writer de saída se especificado
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Processar frame
                if frame_count % self.frame_skip == 0:
                    frame_detections = self._detect_in_frame(frame, frame_count)
                    results['detections'].extend(frame_detections)
                    processed_count += 1
                    
                    # Atualizar estatísticas
                    self._update_summary_stats(results['summary'], frame_detections)
                    
                    # Callback de progresso
                    if progress_callback:
                        progress = (frame_count / total_frames) * 100
                        progress_callback(progress, frame_count, total_frames)
                
                # Salvar frame processado se necessário
                if writer:
                    processed_frame = self._draw_detections(frame, results['detections'][-len(frame_detections):] if frame_count % self.frame_skip == 0 else [])
                    writer.write(processed_frame)
            
            # Finalizar
            cap.release()
            if writer:
                writer.release()
            
            # Calcular estatísticas finais
            processing_time = time.time() - start_time
            results['processed_frames'] = processed_count
            results['processing_time'] = processing_time
            results['fps'] = processed_count / processing_time if processing_time > 0 else 0
            results['summary']['compliance_score'] = self._calculate_compliance_score(results['summary'])
            
            logger.info(f"✅ Processamento concluído: {processed_count} frames em {processing_time:.2f}s ({results['fps']:.2f} FPS)")
            
        except Exception as e:
            logger.error(f"❌ Erro no processamento do vídeo: {e}")
            results['error'] = str(e)
        
        return results
    
    def _detect_in_frame(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """Detecta EPIs em um frame específico"""
        detections = []
        
        try:
            # Redimensionar se necessário
            if max(frame.shape[:2]) > self.max_resolution:
                scale = self.max_resolution / max(frame.shape[:2])
                new_h, new_w = int(frame.shape[0] * scale), int(frame.shape[1] * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
            else:
                frame_resized = frame
                scale = 1.0
            
                # Detecção usando YOLOv11 (mesmo sistema do projeto principal)
                results = self.model(frame_resized, verbose=False)
                
                # Processar resultados (mesma lógica do sistema principal)
                if results and len(results) > 0:
                    for result in results:
                        if result.boxes is not None and len(result.boxes) > 0:
                            boxes = result.boxes
                            for i in range(len(boxes)):
                                confidence = float(boxes.conf[i])
                                
                                if confidence >= self.confidence_threshold:
                                    # Escalar coordenadas de volta
                                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                                    x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
                                    
                                    class_id = int(boxes.cls[i])
                                    class_name = self.model.names[class_id]
                                    
                                    detection_data = {
                                        'frame_number': frame_number,
                                        'timestamp': frame_number / 30.0,  # Assumindo 30 FPS
                                        'bbox': [x1, y1, x2, y2],
                                        'confidence': confidence,
                                        'class_id': class_id,
                                        'class_name': class_name,
                                        'area': (x2 - x1) * (y2 - y1)
                                    }
                                    
                                    detections.append(detection_data)
        
        except Exception as e:
            logger.error(f"Erro na detecção do frame {frame_number}: {e}")
        
        return detections
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Desenha as detecções no frame"""
        processed_frame = frame.copy()
        
        # Cores para diferentes classes (mesmo padrão do sistema principal)
        colors = {
            'person': (0, 255, 0),      # Verde
            'helmet': (255, 0, 0),      # Azul
            'vest': (0, 0, 255),        # Vermelho
            'safety_helmet': (255, 0, 0), # Azul
            'safety_vest': (0, 0, 255),   # Vermelho
            'ear': (255, 255, 0),       # Amarelo
            'ear-mufs': (255, 165, 0),  # Laranja
            'face': (128, 0, 128),      # Roxo
            'face-guard': (0, 255, 255), # Ciano
            'face-mask-medical': (255, 192, 203), # Rosa
            'foot': (165, 42, 42),      # Marrom
            'tools': (128, 128, 128),   # Cinza
            'glasses': (0, 128, 0),     # Verde escuro
            'gloves': (255, 20, 147),   # Rosa escuro
            'hands': (255, 69, 0),      # Vermelho laranja
            'head': (75, 0, 130),       # Índigo
            'medical-suit': (0, 100, 0), # Verde floresta
            'shoes': (139, 69, 19),     # Marrom sela
            'safety-suit': (0, 0, 139), # Azul escuro
        }
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Cor baseada na classe
            color = colors.get(class_name, (255, 255, 0))
            
            # Desenhar bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(processed_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return processed_frame
    
    def _update_summary_stats(self, summary: Dict[str, Any], detections: List[Dict[str, Any]]):
        """Atualiza estatísticas resumidas (mesma lógica do sistema principal)"""
        for detection in detections:
            class_name = detection['class_name']
            
            if class_name == 'person':
                summary['total_pessoas'] += 1
            elif class_name in ['helmet', 'safety_helmet', 'ear', 'ear-mufs']:
                summary['com_capacete'] += 1
            elif class_name in ['vest', 'safety_vest', 'safety-suit', 'medical-suit']:
                summary['com_colete'] += 1
    
    def _calculate_compliance_score(self, summary: Dict[str, Any]) -> float:
        """Calcula score de compliance"""
        total_pessoas = summary['total_pessoas']
        if total_pessoas == 0:
            return 0.0
        
        # Assumir que pessoas sem capacete/colete são violações
        sem_capacete = max(0, total_pessoas - summary['com_capacete'])
        sem_colete = max(0, total_pessoas - summary['com_colete'])
        
        total_violations = sem_capacete + sem_colete
        max_possible_violations = total_pessoas * 2  # Capacete + Colete
        
        if max_possible_violations == 0:
            return 100.0
        
        compliance_score = ((max_possible_violations - total_violations) / max_possible_violations) * 100
        return max(0.0, min(100.0, compliance_score))
    
    def get_supported_formats(self) -> List[str]:
        """Retorna formatos de vídeo suportados"""
        return ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    def validate_video(self, video_path: str) -> Dict[str, Any]:
        """Valida se o vídeo pode ser processado"""
        result = {
            'valid': False,
            'error': None,
            'info': {}
        }
        
        try:
            # Verificar se arquivo existe
            if not Path(video_path).exists():
                result['error'] = "Arquivo não encontrado"
                return result
            
            # Verificar extensão
            ext = Path(video_path).suffix.lower()
            if ext not in self.get_supported_formats():
                result['error'] = f"Formato não suportado: {ext}"
                return result
            
            # Tentar abrir vídeo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result['error'] = "Não foi possível abrir o vídeo"
                return result
            
            # Obter informações
            result['info'] = {
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
            }
            
            cap.release()
            result['valid'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result


class VideoProcessingQueue:
    """Fila de processamento de vídeos para processamento assíncrono"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.queue = Queue()
        self.workers = []
        self.running = False
        self.results = {}
        
    def start(self):
        """Inicia os workers de processamento"""
        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Fila de processamento iniciada com {self.max_workers} workers")
    
    def stop(self):
        """Para os workers"""
        self.running = False
        for worker in self.workers:
            worker.join()
        
        logger.info("Fila de processamento parada")
    
    def add_video(self, video_id: str, video_path: str, output_path: str = None, 
                  progress_callback=None) -> bool:
        """Adiciona vídeo à fila de processamento"""
        try:
            task = {
                'video_id': video_id,
                'video_path': video_path,
                'output_path': output_path,
                'progress_callback': progress_callback,
                'status': 'queued',
                'created_at': datetime.now().isoformat()
            }
            
            self.queue.put(task)
            self.results[video_id] = task
            logger.info(f"Vídeo {video_id} adicionado à fila")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar vídeo à fila: {e}")
            return False
    
    def get_status(self, video_id: str) -> Dict[str, Any]:
        """Obtém status do processamento de um vídeo"""
        return self.results.get(video_id, {'error': 'Vídeo não encontrado'})
    
    def _worker(self):
        """Worker thread para processamento"""
        detector = VideoAIDetector()
        
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                video_id = task['video_id']
                
                logger.info(f"Processando vídeo {video_id}")
                task['status'] = 'processing'
                task['started_at'] = datetime.now().isoformat()
                
                try:
                    # Processar vídeo
                    results = detector.detect_in_video(
                        task['video_path'],
                        task['output_path'],
                        task['progress_callback']
                    )
                    
                    task['status'] = 'completed'
                    task['completed_at'] = datetime.now().isoformat()
                    task['results'] = results
                    
                    logger.info(f"✅ Vídeo {video_id} processado com sucesso")
                    
                except Exception as e:
                    task['status'] = 'error'
                    task['error'] = str(e)
                    task['completed_at'] = datetime.now().isoformat()
                    logger.error(f"❌ Erro no processamento do vídeo {video_id}: {e}")
                
                self.queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Erro no worker: {e}")


# Instância global da fila de processamento
video_queue = VideoProcessingQueue(max_workers=2)
