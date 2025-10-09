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
            'conf_threshold': 0.5,  # Threshold aumentado para maior precisão
            'iou_threshold': 0.45,
            'max_detections': 100,   # Limite maior para modelo de 17 classes
            'frame_skip': 2,        # Processar apenas 1 a cada 2 frames
            'resize_factor': 0.5,   # Reduzir resolução para velocidade
            'batch_size': 1,        # Processar 1 frame por vez
            'warmup_frames': 5       # Frames de aquecimento
        }
        
        # EPIs requeridos (configurável via env: REQUIRED_EPIS=helmet,safety-vest,gloves,glasses)
        required_epis_env = os.getenv("REQUIRED_EPIS", "helmet,safety-vest,gloves,glasses")
        self.required_epis = set([e.strip() for e in required_epis_env.split(',') if e.strip()])

        # Classes do modelo best.pt (17 classes)
        self.class_names = [
            'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask-medical', 
            'foot', 'tools', 'glasses', 'gloves', 'helmet', 'hands', 'head', 
            'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
        ]

        # Classes habilitadas para detecção (por padrão todas ativas)
        self.enabled_classes = set(self.class_names)
        
        # Classes principais para EPIs (mapeamento para o modelo)
        self.main_classes = ['person', 'helmet', 'safety-vest', 'gloves', 'glasses']
        
        # Estado do sistema
        self.current_frame = None
        self.processed_frame = None
        self.current_detections = []
        self.current_violations = []
        self.frame_count = 0
        self.last_process_time = 0
        self.fps_counter = deque(maxlen=30)
        
        # Threading otimizado
        self.frame_queue = Queue(maxsize=1)  # Fila mínima: sempre o frame mais novo
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
            # Estatísticas de compliance
            'total_pessoas': 0,
            'com_capacete': 0,
            'sem_capacete': 0,
            'com_colete': 0,
            'sem_colete': 0,
            'com_luvas': 0,
            'sem_luvas': 0,
            'com_oculos': 0,
            'sem_oculos': 0,
            'compliance_score': 0.0,
            'detection_rate': 0.0,
            'avg_confidence': 0.0,
            'violations': []
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
            # Inicializar enabled_classes com todas as classes do modelo
            self.enabled_classes = set(self.class_names)
            
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
                'device': self.device,
                'imgsz': 640
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
                    results = self.model(frame_small, verbose=False, imgsz=640)
                
                # Processar resultados
                detections = self._process_results(results, frame.shape)
                
                # Escalar coordenadas de volta se necessário
                if self.config['resize_factor'] < 1.0:
                    scale_factor = 1.0 / self.config['resize_factor']
                    for detection in detections:
                        detection['bbox'] = [int(x * scale_factor) for x in detection['bbox']]

                # Avaliar compliance (associar EPIs a pessoas e inferir ausências)
                detections, violations = self._evaluate_compliance(detections)
                
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
                        
                        # Filtrar por threshold e classes habilitadas
                        if conf >= self.config['conf_threshold'] and class_name in self.enabled_classes:
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
            
            # Exibir apenas boxes cirúrgicos (missing-*) no stream
            if not (isinstance(class_name, str) and class_name.startswith('missing-')):
                continue
            
            # Cor fixa vermelha para ausências
            color = (0, 0, 255)
            
            # Desenhar bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Desenhar label (apenas para missing-*)
            if confidence > 0.6:
                nice = class_name.replace('missing-', '').replace('-', ' ')
                label = f"Faltando: {nice}"
                cv2.putText(frame_copy, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
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
        # EPIs considerados para compliance = interseção entre requeridos e habilitados
        active_required = self.required_epis.intersection(self.enabled_classes)
        epis: List[Dict[str, Any]] = [d for d in detections if d.get('class_name') in active_required]
        supports_head: List[Dict[str, Any]] = [d for d in detections if d.get('class_name') in ('head', 'face')]
        supports_hands: List[Dict[str, Any]] = [d for d in detections if d.get('class_name') == 'hands']

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

            for epi in epis:
                c = bbox_center(epi['bbox'])
                if not center_inside(c, person['bbox']):
                    continue
                ename = epi['class_name']
                # Regras rápidas por região
                if ename == 'helmet':
                    if c[1] <= top_region_y:
                        present.add('helmet')
                elif ename == 'glasses':
                    if c[1] <= top_region_y:
                        present.add('glasses')
                elif ename == 'safety-vest':
                    present.add('safety-vest')
                elif ename == 'gloves':
                    present.add('gloves')

            missing = [e for e in active_required if e not in present]
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

                # Criar detecções virtuais "missing-*" com ROIs específicas por EPI ausente
                for m in missing:
                    if m == 'helmet':
                        support = find_support_in(person['bbox'], supports_head)
                        if support is not None:
                            vx1, vy1, vx2, vy2 = support['bbox']
                            roi = clip_box(vx1, vy1, vx2, vy2)
                        else:
                            vx1 = px1 + int(0.15 * (px2 - px1))
                            vx2 = px2 - int(0.15 * (px2 - px1))
                            vy1 = py1
                            vy2 = py1 + int(0.30 * ph)
                            roi = clip_box(vx1, vy1, vx2, vy2)
                        virtual_missing_detections.append({
                            'bbox': roi,
                            'confidence': 1.0,
                            'class_name': 'missing-helmet',
                            'class_id': -1,
                            'frame_id': self.frame_count,
                            'timestamp': time.time()
                        })
                    elif m == 'glasses':
                        support = find_support_in(person['bbox'], supports_head)
                        if support is not None:
                            sx1, sy1, sx2, sy2 = support['bbox']
                            sh = max(1, sy2 - sy1)
                            vy1 = sy1 + int(0.25 * sh)
                            vy2 = sy1 + int(0.45 * sh)
                            vx1 = sx1 + int(0.20 * (sx2 - sx1))
                            vx2 = sx2 - int(0.20 * (sx2 - sx1))
                            roi = clip_box(vx1, vy1, vx2, vy2)
                        else:
                            vy1 = py1 + int(0.18 * ph)
                            vy2 = py1 + int(0.35 * ph)
                            vx1 = px1 + int(0.25 * (px2 - px1))
                            vx2 = px2 - int(0.25 * (px2 - px1))
                            roi = clip_box(vx1, vy1, vx2, vy2)
                        virtual_missing_detections.append({
                            'bbox': roi,
                            'confidence': 1.0,
                            'class_name': 'missing-glasses',
                            'class_id': -1,
                            'frame_id': self.frame_count,
                            'timestamp': time.time()
                        })
                    elif m == 'safety-vest':
                        vy1 = py1 + int(0.35 * ph)
                        vy2 = py1 + int(0.75 * ph)
                        vx1 = px1 + int(0.15 * (px2 - px1))
                        vx2 = px2 - int(0.15 * (px2 - px1))
                        roi = clip_box(vx1, vy1, vx2, vy2)
                        virtual_missing_detections.append({
                            'bbox': roi,
                            'confidence': 1.0,
                            'class_name': 'missing-safety-vest',
                            'class_id': -1,
                            'frame_id': self.frame_count,
                            'timestamp': time.time()
                        })
                    elif m == 'gloves':
                        # Preferir ROIs de mãos detectadas
                        matched_hands = []
                        for h in supports_hands:
                            c = bbox_center(h['bbox'])
                            if center_inside(c, person['bbox']):
                                matched_hands.append(h['bbox'])
                        if matched_hands:
                            for hb in matched_hands:
                                vx1, vy1, vx2, vy2 = hb
                                roi = clip_box(vx1, vy1, vx2, vy2)
                                virtual_missing_detections.append({
                                    'bbox': roi,
                                    'confidence': 1.0,
                                    'class_name': 'missing-gloves',
                                    'class_id': -1,
                                    'frame_id': self.frame_count,
                                    'timestamp': time.time()
                                })
                        else:
                            # fallback: duas regiões inferiores laterais
                            bw = px2 - px1
                            vy1 = py2 - int(0.25 * ph)
                            vy2 = py2 - int(0.05 * ph)
                            # esquerda
                            vx1 = px1 + int(0.05 * bw)
                            vx2 = px1 + int(0.35 * bw)
                            roi_l = clip_box(vx1, vy1, vx2, vy2)
                            virtual_missing_detections.append({
                                'bbox': roi_l,
                                'confidence': 1.0,
                                'class_name': 'missing-gloves',
                                'class_id': -1,
                                'frame_id': self.frame_count,
                                'timestamp': time.time()
                            })
                            # direita
                            vx1 = px2 - int(0.35 * bw)
                            vx2 = px2 - int(0.05 * bw)
                            roi_r = clip_box(vx1, vy1, vx2, vy2)
                            virtual_missing_detections.append({
                                'bbox': roi_r,
                                'confidence': 1.0,
                                'class_name': 'missing-gloves',
                                'class_id': -1,
                                'frame_id': self.frame_count,
                                'timestamp': time.time()
                            })

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

    # ===== API auxiliar para habilitar/desabilitar classes =====
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
            
            # Detecção ultra-rápida
            with torch.no_grad():
                results = self.model(frame_small, verbose=False, imgsz=640)
            
            # Processar resultados
            detections = self._process_results(results, frame.shape)
            
            # Escalar coordenadas de volta se necessário
            if self.config['resize_factor'] < 1.0:
                scale_factor = 1.0 / self.config['resize_factor']
                for detection in detections:
                    detection['bbox'] = [int(x * scale_factor) for x in detection['bbox']]
            
            # Avaliar compliance (associar EPIs a pessoas e inferir ausências)
            detections, violations = self._evaluate_compliance(detections)

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
