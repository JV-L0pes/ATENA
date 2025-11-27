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

class AthenaDetector:
    """Detector consolidado para tempo real e processamento de vídeos"""
    
    def __init__(self, model_path: str = None, video_source: str = None, video_mode: bool = False):
        """
        Inicializa o detector
        
        Args:
            model_path: Caminho para o modelo
            video_source: Fonte de vídeo (RTSP, webcam, etc.)
            video_mode: Se True, otimiza para processamento completo de vídeos (sem frame skip)
        """
        # Usar modelo da configuração ou padrão
        if model_path is None:
            from .config import Config
            model_path = Config.MODEL_PATH
            # Se não existe, tentar caminho alternativo
            if not Path(model_path).exists():
                alt_path = "athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt"
                if Path(alt_path).exists():
                    model_path = alt_path
                else:
                    # Tentar models/best.pt
                    if Path("models/best.pt").exists():
                        model_path = "models/best.pt"
        
        self.model_path = Path(model_path)
        self.video_source = video_source or os.getenv("RTSP_URL", "0")
        self.video_mode = video_mode  # Modo vídeo = processar todos os frames
        self.model = None
        self.device = None
        self.is_initialized = False
        
        # Configurações otimizadas - diferentes para tempo real vs vídeo
        if video_mode:
            # MODO VÍDEO: Máxima precisão, processar todos os frames
            self.config = {
                'conf_threshold': 0.20,  # Threshold mais baixo para capturar mais detecções
                'iou_threshold': 0.45,
                'max_detections': 500,   # Limite maior para não perder detecções
                'frame_skip': 1,         # Processar TODOS os frames (100% de cobertura)
                'resize_factor': 1.0,    # Resolução original (sem perda de detalhes)
                'batch_size': 4,         # Batch processing para melhor performance
                'warmup_frames': 5,
                'max_resolution': None,   # Sem limite de resolução
                'use_temporal_smoothing': True,  # Suavização temporal para melhor precisão
                'multi_scale': False      # Desabilitado para consistência
            }
        else:
            # MODO TEMPO REAL: Balance entre performance e precisão
            self.config = {
                'conf_threshold': 0.25,
                'iou_threshold': 0.45,
                'max_detections': 300,
                'frame_skip': 2,         # Processar 1 a cada 2 frames para performance
                'resize_factor': 1.0,
                'batch_size': 1,
                'warmup_frames': 5,
                'max_resolution': 1920,  # Limite para performance
                'use_temporal_smoothing': True,
                'multi_scale': False
            }
        
        # Limiares por classe (thresholds ajustados para precisão)
        self.class_thresholds: Dict[str, float] = {
            'person': 0.20,         # Threshold mais baixo para não perder pessoas
            'helmet': 0.25,         # EPIs críticos
            'glasses': 0.20,
            'safety-vest': 0.25,
            'gloves': 0.20,
            'ear-mufs': 0.20,
            'ear': 0.20,
            'face': 0.25,
            'face-guard': 0.20,
            'face-mask-medical': 0.20,
            'foot': 0.20,
            'tools': 0.25,
            'hands': 0.20,
            'head': 0.25,
            'medical-suit': 0.20,
            'shoes': 0.20,
            'safety-suit': 0.20,
        }
        
        # EPIs requeridos
        required_epis_env = os.getenv("REQUIRED_EPIS", "helmet,safety-vest,gloves,glasses")
        self.required_epis = set([e.strip() for e in required_epis_env.split(',') if e.strip()])
        
        # Classes do modelo
        self.class_names = [
            'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask-medical', 
            'foot', 'tools', 'glasses', 'gloves', 'helmet', 'hands', 'head', 
            'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
        ]
        
        self.enabled_classes = set(self.class_names)
        self.main_classes = ['person', 'helmet', 'safety-vest', 'gloves', 'glasses']
        
        # Estado do sistema
        self.current_frame = None
        self.processed_frame = None
        self.current_detections = []
        self.current_violations = []
        self.frame_count = 0
        self.last_process_time = 0
        self.fps_counter = deque(maxlen=30)
        
        # Suavização temporal
        self.temporal_params = {
            'missing_frames_required': 3,
            'present_frames_clear': 2
        }
        self.missing_counters: Dict[Tuple[str, str], int] = {}
        
        # Cache de detecções para suavização temporal (modo vídeo)
        self.detection_cache = deque(maxlen=5) if video_mode else None
        
        # Threading
        self.frame_queue = Queue(maxsize=1)
        self.detection_thread = None
        self.running = False
        
        # Estatísticas
        self.stats = {
            'total_detections': 0,
            'fps': 0.0,
            'avg_processing_time': 0.0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_frames': 0,
            'total_pessoas': 0,
            'compliance_score': 0.0,
            'detection_rate': 0.0,
            'avg_confidence': 0.0,
            'violations': []
        }
        
        logger.info(f"🚀 Detector ATHENA inicializado (modo: {'vídeo' if video_mode else 'tempo real'})")
    
    def initialize_model(self):
        """Inicializa modelo com configurações otimizadas"""
        try:
            if not self.model_path.exists():
                logger.error(f"❌ Modelo não encontrado: {self.model_path}")
                return False
            
            logger.info(f"🎯 Carregando modelo: {self.model_path}")
            
            self.model = YOLO(str(self.model_path))
            
            logger.info(f"📋 Classes carregadas: {len(self.model.names)}")
            
            self.class_names = list(self.model.names.values())
            self.enabled_classes = set(self.class_names)
            
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info("🚀 Usando GPU CUDA")
            else:
                self.device = torch.device('cpu')
                logger.info("💻 Usando CPU")
            
            self.model.to(self.device)
            
            self.model.overrides = {
                'verbose': False,
                'max_det': self.config['max_detections'],
                'half': True if self.device.type == 'cuda' else False,
                'device': self.device,
                'imgsz': 640
            }
            
            # Aquecimento
            logger.info("🔥 Aquecendo modelo...")
            dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(self.config['warmup_frames']):
                with torch.no_grad():
                    _ = self.model(dummy_frame, verbose=False)
            
            self.is_initialized = True
            logger.info("✅ Modelo inicializado!")
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
                
                # Detecção com threshold de confiança aplicado
                with torch.no_grad():
                    results = self.model(
                        frame_small,
                        conf=self.config['conf_threshold'],
                        iou=self.config['iou_threshold'],
                        verbose=False,
                        imgsz=640
                    )
                
                # Processar resultados
                detections = self._process_results(results, frame.shape)
                
                # Escalar coordenadas de volta se necessário
                if self.config['resize_factor'] < 1.0:
                    scale_factor = 1.0 / self.config['resize_factor']
                    for detection in detections:
                        detection['bbox'] = [int(x * scale_factor) for x in detection['bbox']]

                # Avaliar compliance (associar EPIs a pessoas e inferir ausências)
                detections, violations = self._evaluate_compliance(detections)
                
                # FILTRAR EPIs soltos - só manter EPIs associados a pessoas
                detections = self._filter_orphan_epis(detections)
                
                # Desenhar resultados
                processed_frame = self._draw_detections(frame, detections)
                
                # Atualizar estado
                self.current_frame = frame.copy()
                self.processed_frame = processed_frame
                self.current_detections = detections
                self.current_violations = violations
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
        
        if not results or len(results) == 0:
            logger.debug("⚠️ Nenhum resultado do modelo")
            return detections
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            logger.debug("⚠️ Nenhuma caixa detectada pelo modelo")
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        # Log sempre (não só em debug) para diagnóstico
        if len(boxes) > 0:
            logger.info(f"📦 Boxes brutas do modelo: {len(boxes)}")
            logger.info(f"📊 Confianças: min={confidences.min():.3f}, max={confidences.max():.3f}, mean={confidences.mean():.3f}")
            logger.info(f"🏷️ Classes detectadas: {[self.class_names[int(c)] for c in set(class_ids)]}")
        else:
            logger.warning("⚠️ Modelo não retornou nenhuma detecção bruta")
        
        filtered_count = 0
        for i, (box, conf, cls) in enumerate(zip(boxes, confidences, class_ids)):
            if 0 <= int(cls) < len(self.class_names):
                x1, y1, x2, y2 = box
                class_name = self.class_names[int(cls)]
                # Filtrar por threshold por classe
                thr = self.class_thresholds.get(class_name, self.config['conf_threshold'])
                
                if conf < thr:
                    filtered_count += 1
                    if self.frame_count % 30 == 0:
                        logger.debug(f"🚫 Filtrado: {class_name} (conf={conf:.3f} < thr={thr:.3f})")
                    continue
                
                if class_name not in self.enabled_classes:
                    filtered_count += 1
                    if self.frame_count % 30 == 0:
                        logger.debug(f"🚫 Classe desabilitada: {class_name}")
                    continue
                
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
                if self.frame_count % 30 == 0:
                    logger.info(f"✅ Detecção: {class_name} (conf={conf:.3f}, thr={thr:.3f})")
            else:
                logger.warning(f"⚠️ Class ID {int(cls)} fora do range [0, {len(self.class_names)})")
        
        if filtered_count > 0 and self.frame_count % 30 == 0:
            logger.debug(f"🚫 Total filtrado: {filtered_count}/{len(boxes)}")
        
        logger.debug(f"✅ Detecções finais: {len(detections)}")
        return detections
    
    def _filter_orphan_epis(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtra EPIs soltos (não associados a pessoas).
        Só mantém: pessoas, EPIs associados a pessoas, e detecções virtuais missing-*
        """
        if not detections:
            return detections
        
        # Identificar pessoas
        persons = [d for d in detections if d.get('class_name') == 'person']
        if not persons:
            # Se não há pessoas, retornar apenas detecções virtuais missing-* (se houver)
            return [d for d in detections if d.get('class_name', '').startswith('missing-')]
        
        # Criar bounding boxes de pessoas para verificação de overlap
        person_boxes = [p['bbox'] for p in persons]
        
        def bbox_center(b):
            x1, y1, x2, y2 = b
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
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
        
        # Filtrar detecções
        filtered = []
        
        for detection in detections:
            class_name = detection.get('class_name', '')
            
            # Sempre manter: pessoas e detecções virtuais missing-*
            if class_name == 'person' or class_name.startswith('missing-'):
                filtered.append(detection)
                continue
            
            # Para outras classes (EPIs, partes do corpo, etc.):
            # Só manter se estiver associado a uma pessoa
            epi_bbox = detection.get('bbox', [])
            if len(epi_bbox) < 4:
                continue
            
            # Verificar se o EPI está dentro ou próximo de alguma pessoa
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
                filtered.append(detection)
            else:
                # EPI solto - não adicionar
                logger.debug(f"🚫 EPI solto filtrado: {class_name} (não associado a pessoa)")
        
        return filtered
    
    def _draw_detections(self, frame, detections):
        """Desenha detecções no frame - Verde = tem EPI, Vermelho = sem EPI"""
        frame_copy = frame.copy()
        
        # Cores simplificadas: Verde = tem EPI, Vermelho = sem EPI
        COLOR_GREEN = (0, 255, 0)   # BGR: Verde = tem EPI
        COLOR_RED = (0, 0, 255)     # BGR: Vermelho = sem EPI
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Determinar cor: Verde ou Vermelho
            has_missing_epi = detection.get('missing_epis', []) or detection.get('type') == 'negative'
            is_missing_class = isinstance(class_name, str) and class_name.startswith('missing-')
            is_compliant = detection.get('compliant', True) and not has_missing_epi
            
            if is_missing_class or has_missing_epi or not is_compliant:
                # SEM EPI = VERMELHO
                color = COLOR_RED
                if is_missing_class:
                    missing_epi = class_name.replace('missing-', '').replace('-', ' ')
                    label = f"Sem {missing_epi}"
                else:
                    missing_epi = detection.get('missing_epi', 'EPI')
                    label = f"Sem {missing_epi}"
            else:
                # COM EPI = VERDE
                color = COLOR_GREEN
                label = f"{class_name}: {confidence:.2f}"
            
            # Desenhar bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar label
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy

    def _evaluate_compliance(self, detections: List[Dict[str, Any]]):
        """Associa EPIs às pessoas e infere ausências.
        Retorna (detections_atualizadas, violations).
        """
        if not detections:
            # reset stats de pessoas
            self._update_compliance_stats(total_pessoas=0, per_person=[])
            return detections, []

        persons: List[Dict[str, Any]] = [d for d in detections if d.get('class_name') == 'person']
        
        # DINÂMICO: Identificar todas as classes de EPI detectadas (exceto person e partes do corpo)
        # Partes do corpo que não são EPIs: person, face, hands, head, foot, ear (sem proteção)
        body_parts = {'person', 'face', 'hands', 'head', 'foot', 'ear'}
        
        # Todas as detecções que não são pessoa nem parte do corpo são consideradas EPIs
        epis_raw: List[Dict[str, Any]] = [
            d for d in detections 
            if d.get('class_name') not in body_parts and not d.get('class_name', '').startswith('missing-')
        ]
        
        # EPIs únicos detectados (dinâmico)
        detected_epi_classes = set(d.get('class_name') for d in epis_raw)
        
        # EPIs requeridos = usar required_epis da config (não só os detectados)
        # Se não detectar EPIs, ainda devemos verificar se estão faltando
        active_required = set()
        # Primeiro, adicionar todos os EPIs requeridos que estão habilitados
        for req_epi in self.required_epis:
            if req_epi in self.enabled_classes:
                active_required.add(req_epi)
        # Também adicionar EPIs detectados que estão nos requeridos
        for epi_class in detected_epi_classes:
            if epi_class in self.enabled_classes:
                if epi_class in self.required_epis:
                    active_required.add(epi_class)
                # Mapear aliases comuns
                alias_map = {
                    'ear-mufs': 'ear-plugs',
                    'ear': 'ear-plugs'
                }
                if epi_class in alias_map and alias_map[epi_class] in self.required_epis:
                    active_required.add(alias_map[epi_class])
        # Suportes anatômicos
        supports_head: List[Dict[str, Any]] = [d for d in detections if d.get('class_name') in ('head', 'face')]
        supports_hands: List[Dict[str, Any]] = [d for d in detections if d.get('class_name') == 'hands']
        supports_ears: List[Dict[str, Any]] = [d for d in detections if d.get('class_name') == 'ear']

        def bbox_center(b):
            x1, y1, x2, y2 = b
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))

        def center_inside(center, box):
            cx, cy = center
            x1, y1, x2, y2 = box
            return x1 <= cx <= x2 and y1 <= cy <= y2

        def clip_box(x1, y1, x2, y2):
            return [max(0, int(x1)), max(0, int(y1)), max(0, int(x2)), max(0, int(y2))]

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        def find_support_in(person_box, support_list):
            best = None
            best_iou = 0.0
            for s in support_list:
                ov = iou(person_box, s['bbox'])
                if ov > best_iou:
                    best_iou = ov
                    best = s
            return best

        # Mapeamento de EPIs presentes por pessoa
        per_person_present = []
        violations = []
        virtual_missing_detections: List[Dict[str, Any]] = []
        for person in persons:
            px1, py1, px2, py2 = person['bbox']
            ph = max(1, py2 - py1)
            # região do topo para capacete/óculos
            top_region_y = py1 + int(0.35 * ph)
            present = set()
            person_box = person['bbox']

            # Associação DINÂMICA de EPIs a pessoas
            # Para cada EPI detectado, verificar se está associado a esta pessoa
            for epi in epis_raw:
                epi_name = epi['class_name']
                epi_bbox = epi['bbox']
                epi_center = bbox_center(epi_bbox)
                
                # Verificar se o EPI está dentro da bounding box da pessoa
                if not center_inside(epi_center, person_box):
                    continue
                
                # EPIs que precisam estar na região superior (cabeça)
                head_epis = {'helmet', 'glasses', 'face-guard', 'face-mask-medical', 'ear', 'ear-mufs'}
                if epi_name in head_epis:
                    # Verificar se está na região superior da pessoa
                    if epi_center[1] <= top_region_y:
                        present.add(epi_name)
                # EPIs que precisam estar no tronco
                elif epi_name in {'safety-vest', 'medical-suit', 'safety-suit'}:
                    present.add(epi_name)
                # EPIs que precisam estar nas mãos
                elif epi_name in {'gloves'}:
                    # Verificar se está próximo a mãos detectadas
                    matched = False
                    for h in supports_hands:
                        if center_inside(bbox_center(h['bbox']), person_box) and iou(epi_bbox, h['bbox']) > 0.3:
                            matched = True
                            break
                    if matched:
                        present.add(epi_name)
                # EPIs que precisam estar nos pés
                elif epi_name in {'shoes', 'foot'}:
                    # Verificar se está na região inferior da pessoa
                    bottom_region_y = py2 - int(0.25 * ph)
                    if epi_center[1] >= bottom_region_y:
                        present.add(epi_name)
                # Outros EPIs: se está dentro da pessoa, considerar presente
                else:
                    present.add(epi_name)

            # Gating por visibilidade: só considerar ausências se a região/suporte estiver visível
            # DINÂMICO: Para cada EPI requerido, verificar se está presente ou faltando
            missing = []
            for req_epi in active_required:
                if req_epi not in present:
                    # Verificar se há suporte anatômico visível para este EPI
                    # EPIs de cabeça precisam de head/face visível
                    head_epis = {'helmet', 'glasses', 'face-guard', 'face-mask-medical', 'ear', 'ear-mufs', 'ear-plugs'}
                    if req_epi in head_epis:
                        if any(center_inside(bbox_center(s['bbox']), person_box) for s in supports_head):
                            missing.append(req_epi)
                    # EPIs de mão precisam de hands visível
                    elif req_epi == 'gloves':
                        if any(center_inside(bbox_center(s['bbox']), person_box) for s in supports_hands):
                            missing.append(req_epi)
                    # Outros EPIs (safety-vest, etc.) não precisam de suporte anatômico
                    else:
                        missing.append(req_epi)
            person['missing_epis'] = missing
            person['compliant'] = len(missing) == 0
            per_person_present.append({
                'present': present,
                'missing': missing
            })
            if missing:
                violations.append({
                    'person_bbox': person['bbox'],
                    'missing_epis': missing
                })

            # Criar detecções virtuais "missing-*" DINÂMICAS para todos os EPIs faltando
            for m in missing:
                # Determinar região do EPI faltando baseado no tipo
                head_epis = {'helmet', 'glasses', 'face-guard', 'face-mask-medical', 'ear', 'ear-mufs', 'ear-plugs'}
                torso_epis = {'safety-vest', 'medical-suit', 'safety-suit'}
                hand_epis = {'gloves'}
                foot_epis = {'shoes', 'foot'}
                
                roi = None
                
                if m in head_epis:
                    # EPI de cabeça: usar região superior da pessoa ou suporte de cabeça
                    support = find_support_in(person['bbox'], supports_head)
                    if support is not None:
                        sx1, sy1, sx2, sy2 = support['bbox']
                        sh = max(1, sy2 - sy1)
                        vy1 = sy1 + int(0.20 * sh)
                        vy2 = sy1 + int(0.50 * sh)
                        vx1 = sx1 + int(0.20 * (sx2 - sx1))
                        vx2 = sx2 - int(0.20 * (sx2 - sx1))
                        roi = clip_box(vx1, vy1, vx2, vy2)
                    else:
                        vx1 = px1 + int(0.20 * (px2 - px1))
                        vx2 = px2 - int(0.20 * (px2 - px1))
                        vy1 = py1
                        vy2 = py1 + int(0.35 * ph)
                        roi = clip_box(vx1, vy1, vx2, vy2)
                elif m in torso_epis:
                    # EPI de tronco: região central da pessoa
                    vy1 = py1 + int(0.30 * ph)
                    vy2 = py1 + int(0.80 * ph)
                    vx1 = px1 + int(0.15 * (px2 - px1))
                    vx2 = px2 - int(0.15 * (px2 - px1))
                    roi = clip_box(vx1, vy1, vx2, vy2)
                elif m in hand_epis:
                    # EPI de mão: usar mãos detectadas
                    hands_in_person = [h['bbox'] for h in supports_hands if center_inside(bbox_center(h['bbox']), person_box)]
                    if hands_in_person:
                        # Criar uma detecção missing por mão sem luva
                        for hb in hands_in_person:
                            has_epi = False
                            for epi in epis_raw:
                                if epi['class_name'] == m and iou(epi['bbox'], hb) > 0.3:
                                    has_epi = True
                                    break
                            if not has_epi:
                                vx1, vy1, vx2, vy2 = hb
                                roi = clip_box(vx1, vy1, vx2, vy2)
                                virtual_missing_detections.append({
                                    'bbox': roi,
                                    'confidence': 1.0,
                                    'class_name': f'missing-{m}',
                                    'class_id': -1,
                                    'frame_id': self.frame_count,
                                    'timestamp': time.time()
                                })
                        continue  # Já adicionou, pular para próximo
                    else:
                        # Se não há mãos detectadas, usar região lateral da pessoa
                        vx1 = px1
                        vx2 = px1 + int(0.25 * (px2 - px1))
                        vy1 = py1 + int(0.50 * ph)
                        vy2 = py1 + int(0.75 * ph)
                        roi = clip_box(vx1, vy1, vx2, vy2)
                elif m in foot_epis:
                    # EPI de pé: região inferior da pessoa
                    vy1 = py2 - int(0.30 * ph)
                    vy2 = py2
                    vx1 = px1 + int(0.25 * (px2 - px1))
                    vx2 = px2 - int(0.25 * (px2 - px1))
                    roi = clip_box(vx1, vy1, vx2, vy2)
                else:
                    # EPI genérico: usar região central da pessoa
                    vy1 = py1 + int(0.30 * ph)
                    vy2 = py1 + int(0.70 * ph)
                    vx1 = px1 + int(0.20 * (px2 - px1))
                    vx2 = px2 - int(0.20 * (px2 - px1))
                    roi = clip_box(vx1, vy1, vx2, vy2)
                
                # Adicionar detecção virtual missing (exceto para hand_epis que já foram adicionadas)
                if roi:
                    virtual_missing_detections.append({
                        'bbox': roi,
                        'confidence': 1.0,
                        'class_name': f'missing-{m}',
                        'class_id': -1,
                        'frame_id': self.frame_count,
                        'timestamp': time.time()
                    })

        # Suavização temporal de ausências por pessoa (anti-flicker)
        def person_key(box):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            # quantizar para reduzir jitter
            return f"{(cx//50)*50}:{(cy//50)*50}:{(x2-x1)//50}:{(y2-y1)//50}"

        stabilized_missing = []
        for p in persons:
            key = person_key(p['bbox'])
            miss = [m for m in p.get('missing_epis', [])]
            # atualizar contadores
            for epi in miss:
                ck = (key, epi)
                self.missing_counters[ck] = self.missing_counters.get(ck, 0) + 1
                if self.missing_counters[ck] >= self.temporal_params['missing_frames_required']:
                    stabilized_missing.append((key, epi))
            # limpar quando presente
            for epi in active_required:
                if epi not in miss:
                    ck = (key, epi)
                    cval = self.missing_counters.get(ck, 0)
                    if cval > 0:
                        self.missing_counters[ck] = max(0, cval - self.temporal_params['present_frames_clear'])

        # filtrar virtual_missing por estabilização
        if stabilized_missing:
            keep = []
            for v in virtual_missing_detections:
                k = person_key(v['bbox'])  # aproximação
                epi_name = v.get('class_name', '').replace('missing-', '')
                if (k, epi_name) in stabilized_missing:
                    keep.append(v)
            virtual_missing_detections = keep

        # Atualizar estatísticas agregadas
        self._update_compliance_stats(total_pessoas=len(persons), per_person=per_person_present)

        # Anexar virtual_missing_detections para visualização/SSE
        if virtual_missing_detections:
            detections = detections + virtual_missing_detections

        return detections, violations

    def _update_compliance_stats(self, total_pessoas: int, per_person: List[Dict[str, Any]]):
        com_capacete = sum(1 for p in per_person if 'helmet' in p['present'])
        sem_capacete = total_pessoas - com_capacete
        com_colete = sum(1 for p in per_person if 'safety-vest' in p['present'])
        sem_colete = total_pessoas - com_colete
        com_luvas = sum(1 for p in per_person if 'gloves' in p['present'])
        sem_luvas = total_pessoas - com_luvas
        com_oculos = sum(1 for p in per_person if 'glasses' in p['present'])
        sem_oculos = total_pessoas - com_oculos
        compliant = sum(1 for p in per_person if not p['missing'])
        compliance_score = (compliant / total_pessoas * 100.0) if total_pessoas > 0 else 0.0

        self.stats.update({
            'total_pessoas': total_pessoas,
            'com_capacete': com_capacete,
            'sem_capacete': sem_capacete,
            'com_colete': com_colete,
            'sem_colete': sem_colete,
            'com_luvas': com_luvas,
            'sem_luvas': sem_luvas,
            'com_oculos': com_oculos,
            'sem_oculos': sem_oculos,
            'compliance_score': compliance_score,
        })

    def get_enabled_classes(self) -> List[str]:
        return sorted(list(self.enabled_classes))

    def set_enabled_classes(self, enabled: List[str]):
        if not enabled:
            # garantir que pelo menos 'person' fique habilitada para compliance básico
            self.enabled_classes = {'person'} if 'person' in self.class_names else set()
            return
        valid = set(c for c in enabled if c in self.class_names)
        if 'person' in self.class_names and 'person' not in valid:
            # manter 'person' sempre habilitada
            valid.add('person')
        self.enabled_classes = valid
    
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
            
            # Detecção com threshold de confiança aplicado
            with torch.no_grad():
                # Aplicar threshold de confiança no modelo para filtrar falsos positivos
                results = self.model(
                    frame_small,
                    conf=self.config['conf_threshold'],  # Threshold base aplicado no modelo
                    iou=self.config['iou_threshold'],
                    verbose=False,
                    imgsz=640
                )
            
            # Processar resultados
            detections = self._process_results(results, frame.shape)
            
            # Escalar coordenadas de volta se necessário
            if self.config['resize_factor'] < 1.0:
                scale_factor = 1.0 / self.config['resize_factor']
                for detection in detections:
                    detection['bbox'] = [int(x * scale_factor) for x in detection['bbox']]
            
            # Avaliar compliance (associar EPIs a pessoas e inferir ausências)
            detections, violations = self._evaluate_compliance(detections)
            
            # FILTRAR EPIs soltos - só manter EPIs associados a pessoas
            detections = self._filter_orphan_epis(detections)

            # Desenhar resultados
            processed_frame = self._draw_detections(frame, detections)
            
            # Atualizar estado
            self.current_frame = frame.copy()
            self.processed_frame = processed_frame
            self.current_detections = detections
            self.current_violations = violations
            self.frame_count += 1
            
            # Atualizar estatísticas
            processing_time = time.time() - start_time
            self.stats['avg_processing_time'] = processing_time
            self.stats['total_frames'] += 1
            
            # Log para debug
            if self.frame_count % 30 == 0:
                logger.info(f"🔍 Frame {self.frame_count}: {len(detections)} detecções")
            elif len(detections) > 0:
                logger.info(f"🔍 Frame {self.frame_count}: {len(detections)} detecções encontradas!")
            
            # Log detalhado se não houver detecções
            if len(detections) == 0 and self.frame_count % 60 == 0:
                logger.warning(f"⚠️ Frame {self.frame_count}: Nenhuma detecção. Modelo: {self.model is not None}, Inicializado: {self.is_initialized}")
            
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
    """Sistema principal otimizado (compatibilidade)"""
    
    def __init__(self, model_path: str = None, video_source: str = None):
        self.detector = AthenaDetector(model_path, video_source)
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
