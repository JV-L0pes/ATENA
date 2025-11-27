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
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.20):
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
        
        # Configurações de processamento - OTIMIZADAS PARA MÁXIMA DETECÇÃO
        self.frame_skip = 1  # Processar TODOS os frames (100% de cobertura)
        self.max_resolution = None  # SEM limite de resolução (manter original para máxima precisão)
        self.batch_size = 1  # Processar um frame por vez (mais preciso que batch)
        
        # Estatísticas
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'detections_found': 0,
            'processing_time': 0.0,
            'fps': 0.0
        }
        
        logger.info(f"🎬 Video AI Detector inicializado com threshold {confidence_threshold} (OTIMIZADO PARA MÁXIMA DETECÇÃO)")
        logger.info(f"   ✅ Processando TODOS os frames (100% de cobertura)")
        logger.info(f"   ✅ Resolução original mantida (sem limite)")
        logger.info(f"   ✅ Threshold otimizado para capturar mais detecções")
    
    def _get_latest_model(self) -> str:
        """Obtém o modelo best.pt da Fase 1"""
        from backend.config import CONFIG
        model_path = CONFIG.MODEL_PATH
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        return model_path
    
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
        """Detecta EPIs em um frame específico (positivos e negativos) - MÁXIMA PRECISÃO"""
        detections = []
        
        try:
            # MELHORIA: Manter resolução original (sem redimensionamento) para máxima precisão
            # Só redimensionar se realmente necessário (muito grande) e manter proporção
            scale = 1.0
            frame_resized = frame
            
            # Limite apenas para frames extremamente grandes (4K+) para evitar OOM
            # Se max_resolution for None, nunca redimensiona (máxima precisão)
            if self.max_resolution is not None and max(frame.shape[:2]) > self.max_resolution:
                scale = self.max_resolution / max(frame.shape[:2])
                new_h, new_w = int(frame.shape[0] * scale), int(frame.shape[1] * scale)
                # Usar INTER_LINEAR para melhor qualidade ao redimensionar
                frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                logger.debug(f"Frame {frame_number}: Redimensionado de {frame.shape[:2]} para {frame_resized.shape[:2]}")
            else:
                # Manter resolução original - máxima precisão
                frame_resized = frame
                scale = 1.0
            
            # MELHORIA: Detecção com threshold mais baixo e confiança explícita
            # Usar conf=0.20 para capturar mais detecções válidas
            results = self.model(
                frame_resized, 
                verbose=False,
                conf=0.20,  # Threshold mais baixo para não perder detecções
                iou=0.45,   # IoU padrão
                imgsz=640   # Tamanho de entrada do modelo
            )
            
            # Processar resultados (mesma lógica do sistema principal)
            if results and len(results) > 0:
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            confidence = float(boxes.conf[i])
                            
                            # MELHORIA: Usar threshold por classe (mais preciso)
                            # Threshold mais baixo para capturar mais detecções válidas
                            class_threshold = self.confidence_threshold
                            
                            # Thresholds específicos por classe (mais baixos para não perder detecções)
                            class_thresholds = {
                                'person': 0.20,  # Pessoas são críticas - threshold mais baixo
                                'helmet': 0.25,
                                'safety-vest': 0.25,
                                'gloves': 0.20,
                                'glasses': 0.20,
                            }
                            
                            # Usar threshold específico da classe se disponível
                            if class_name in class_thresholds:
                                class_threshold = class_thresholds[class_name]
                            
                            if confidence >= class_threshold:
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
            
            # Analisar pessoas para detectar EPIs faltando (negativos)
            people_detections = [d for d in detections if d.get('class_name') == 'person']
            epi_detections = [d for d in detections if d.get('class_name') != 'person']
            
            # FILTRAR EPIs soltos antes de processar
            epi_detections = self._filter_orphan_epis(epi_detections, people_detections)
            
            # Atualizar detecções: remover EPIs soltos
            detections = people_detections + epi_detections
            
            # Para cada pessoa, verificar EPIs faltando
            for person in people_detections:
                person_bbox = person['bbox']
                missing_epis = self._analyze_missing_epis(person, epi_detections, person_bbox)
                
                if missing_epis:
                    person['missing_epis'] = missing_epis
                    person['compliant'] = False
                    
                    # Criar detecções virtuais "missing-*" para cada EPI faltando
                    for missing_epi in missing_epis:
                        missing_bbox = self._get_missing_epi_bbox(person_bbox, missing_epi)
                        detections.append({
                            'frame_number': frame_number,
                            'timestamp': frame_number / 30.0,
                            'bbox': missing_bbox,
                            'confidence': 1.0,  # Confiança alta para detecções virtuais
                            'class_id': -1,  # ID especial para detecções virtuais
                            'class_name': f'missing-{missing_epi}',
                            'area': (missing_bbox[2] - missing_bbox[0]) * (missing_bbox[3] - missing_bbox[1]),
                            'type': 'negative',
                            'missing_epi': missing_epi,
                            'person_bbox': person_bbox
                        })
                else:
                    person['missing_epis'] = []
                    person['compliant'] = True
        
        except Exception as e:
            logger.error(f"Erro na detecção do frame {frame_number}: {e}")
        
        return detections
    
    def _filter_orphan_epis(self, epi_detections: List[Dict], people_detections: List[Dict]) -> List[Dict]:
        """
        Filtra EPIs soltos (não associados a pessoas).
        Só mantém EPIs que estão dentro ou próximos de pessoas.
        """
        if not epi_detections or not people_detections:
            return epi_detections if people_detections else []
        
        def bbox_center(b):
            x1, y1, x2, y2 = b
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        
        def center_inside(center, box):
            cx, cy = center
            x1, y1, x2, y2 = box
            return x1 <= cx <= x2 and y1 <= cy <= y2
        
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0
        
        person_boxes = [p['bbox'] for p in people_detections]
        filtered = []
        
        for epi in epi_detections:
            epi_bbox = epi.get('bbox', [])
            if len(epi_bbox) < 4:
                continue
            
            epi_center = bbox_center(epi_bbox)
            associated = False
            
            for person_box in person_boxes:
                # Verificar se o centro do EPI está dentro da pessoa
                if center_inside(epi_center, person_box):
                    associated = True
                    break
                # Ou verificar se há overlap significativo (IoU > 0.1)
                if iou(epi_bbox, person_box) > 0.1:
                    associated = True
                    break
            
            if associated:
                filtered.append(epi)
            else:
                logger.debug(f"🚫 EPI solto filtrado: {epi.get('class_name')} (não associado a pessoa)")
        
        return filtered
    
    def _analyze_missing_epis(self, person: Dict, epi_detections: List[Dict], person_bbox: List[int]) -> List[str]:
        """Analisa quais EPIs estão faltando para uma pessoa"""
        missing = []
        
        # EPIs requeridos
        required_epis = ['helmet', 'safety-vest']  # Pode ser configurável
        
        # Verificar quais EPIs estão presentes perto da pessoa
        px1, py1, px2, py2 = person_bbox
        person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
        
        present_epis = set()
        for epi in epi_detections:
            epi_bbox = epi['bbox']
            epi_center = ((epi_bbox[0] + epi_bbox[2]) / 2, (epi_bbox[1] + epi_bbox[3]) / 2)
            
            # Verificar se EPI está dentro ou próximo da pessoa
            if (px1 <= epi_center[0] <= px2 and py1 <= epi_center[1] <= py2):
                class_name = epi['class_name']
                # Mapear classes do modelo para EPIs
                if class_name in ['helmet', 'ear', 'ear-mufs']:
                    present_epis.add('helmet')
                elif class_name in ['safety-vest', 'vest', 'safety-suit']:
                    present_epis.add('safety-vest')
        
        # Verificar quais EPIs estão faltando
        for required in required_epis:
            if required not in present_epis:
                missing.append(required)
        
        return missing
    
    def _get_missing_epi_bbox(self, person_bbox: List[int], missing_epi: str) -> List[int]:
        """Gera bbox para detecção virtual de EPI faltando"""
        px1, py1, px2, py2 = person_bbox
        ph = py2 - py1
        pw = px2 - px1
        
        if missing_epi == 'helmet':
            # Região da cabeça (topo da pessoa)
            vx1 = px1 + int(0.15 * pw)
            vx2 = px2 - int(0.15 * pw)
            vy1 = py1
            vy2 = py1 + int(0.30 * ph)
        elif missing_epi == 'safety-vest':
            # Região do tronco
            vx1 = px1 + int(0.10 * pw)
            vx2 = px2 - int(0.10 * pw)
            vy1 = py1 + int(0.25 * ph)
            vy2 = py1 + int(0.70 * ph)
        else:
            # Região genérica
            vx1, vy1, vx2, vy2 = px1, py1, px2, py2
        
        return [max(0, vx1), max(0, vy1), max(0, vx2), max(0, vy2)]
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Desenha as detecções no frame - Verde = tem EPI, Vermelho = sem EPI"""
        processed_frame = frame.copy()
        
        # Cores simplificadas: Verde = tem EPI, Vermelho = sem EPI
        COLOR_GREEN = (0, 255, 0)   # BGR: Verde = tem EPI
        COLOR_RED = (0, 0, 255)     # BGR: Vermelho = sem EPI
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Determinar cor: Verde ou Vermelho
            has_missing_epi = detection.get('missing_epis', []) or detection.get('type') == 'negative'
            is_missing_class = class_name.startswith('missing-')
            is_compliant = detection.get('compliant', True) and not has_missing_epi
            
            if is_missing_class or has_missing_epi or not is_compliant:
                # SEM EPI = VERMELHO
                color = COLOR_RED
                missing_epi = detection.get('missing_epi', 'EPI')
                label = f"Sem {missing_epi}"
            else:
                # COM EPI = VERDE
                color = COLOR_GREEN
                label = f"{class_name}: {confidence:.2f}"
            
            # Desenhar bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar label
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
                    # Callback de progresso para atualizar status
                    def progress_callback(progress, frame_count, total_frames):
                        task['progress'] = progress
                        task['current_frame'] = frame_count
                        task['total_frames'] = total_frames
                    
                    # Processar vídeo
                    results = detector.detect_in_video(
                        task['video_path'],
                        task['output_path'],
                        progress_callback
                    )
                    
                    task['status'] = 'completed'
                    task['completed_at'] = datetime.now().isoformat()
                    task['results'] = results
                    task['progress'] = 100.0
                    
                    # Criar relatório automaticamente após processamento
                    try:
                        from backend.video_report import video_report_system
                        report = video_report_system.create_report(video_id, results)
                        task['report'] = report
                        logger.info(f"✅ Relatório criado para vídeo {video_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Erro ao criar relatório: {e}")
                    
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
