#!/usr/bin/env python3
"""
INTERFACE COMPLETA E FUNCIONAL - Detecção de EPIs Ultra-Otimizada
Sistema anti-veículo + análise anatômica + fullscreen + performance adaptativa
"""

import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import queue
from PIL import Image, ImageTk
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque
import json

# ============================================================================
# SISTEMA ANTI-VEÍCULO ROBUSTO
# ============================================================================

class ObjectType(Enum):
    HUMAN = "human"
    MOTORCYCLE = "motorcycle"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"

@dataclass
class ObjectGeometry:
    width: int
    height: int
    area: int
    aspect_ratio: float
    width_height_ratio: float
    
    @classmethod
    def from_bbox(cls, bbox: List[int]) -> 'ObjectGeometry':
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = height / width if width > 0 else 0
        width_height_ratio = width / height if height > 0 else 0
        
        return cls(width, height, area, aspect_ratio, width_height_ratio)

class AdvancedHumanValidator:
    """Validador avançado com múltiplas camadas de análise"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações otimizadas para canteiro de obra
        self.config = {
            'human_constraints': {
                'min_height_px': 80,            # Mais flexível para pessoas distantes
                'max_height_px': 1200,
                'min_aspect_ratio': 1.2,        # Mais flexível para poses
                'max_aspect_ratio': 6.0,
                'min_width_px': 15,
                'max_width_px': 400
            },
            
            'motorcycle_detection': {
                'critical_indicators': {
                    'width_height_ratio': (0.6, 1.5),  # Motos são mais largas
                    'min_area': 3000,
                    'max_area': 60000,
                    'min_width': 60
                },
                'suspicious_patterns': {
                    'nearly_square': (0.7, 1.3),       # Muito suspeito
                    'too_wide': 2.0,                    # Muito largo
                    'high_symmetry_threshold': 0.8      # Muito simétrico
                }
            },
            
            'confidence_strategy': {
                'high_conf': {'threshold': 0.6, 'validation': 'basic'},
                'medium_conf': {'threshold': 0.35, 'validation': 'rigorous'},
                'low_conf': {'threshold': 0.15, 'validation': 'extreme'}
            }
        }
    
    def is_valid_human(self, detection_data: Dict) -> Tuple[bool, str, float]:
        """
        Validação principal que retorna resultado + razão + confiança
        """
        geometry = ObjectGeometry.from_bbox(detection_data['bbox'])
        confidence = detection_data['confidence']
        
        # Análise em camadas baseada na confiança
        if confidence >= self.config['confidence_strategy']['high_conf']['threshold']:
            return self._validate_high_confidence(geometry, confidence)
        elif confidence >= self.config['confidence_strategy']['medium_conf']['threshold']:
            return self._validate_medium_confidence(geometry, confidence)
        else:
            return self._validate_low_confidence(geometry, confidence)
    
    def _validate_high_confidence(self, geometry: ObjectGeometry, confidence: float) -> Tuple[bool, str, float]:
        """Validação para alta confiança - mais permissiva"""
        # Rejeita apenas casos óbvios
        if self._is_obviously_vehicle(geometry):
            return False, f"Veículo óbvio (ratio: {geometry.width_height_ratio:.2f})", 0.1
        
        return True, f"Alta confiança aprovada ({confidence:.2f})", confidence
    
    def _validate_medium_confidence(self, geometry: ObjectGeometry, confidence: float) -> Tuple[bool, str, float]:
        """Validação rigorosa para confiança média"""
        # Validações dimensionais
        if not self._meets_size_constraints(geometry):
            return False, f"Dimensões inválidas ({geometry.width}x{geometry.height})", 0.2
        
        # Validações de proporção
        if not self._meets_proportion_constraints(geometry):
            return False, f"Proporções inválidas (ratio: {geometry.aspect_ratio:.2f})", 0.2
        
        # Detecção específica de moto
        is_moto, moto_reason = self._detect_motorcycle_signature(geometry)
        if is_moto:
            return False, f"Moto detectada: {moto_reason}", 0.1
        
        return True, f"Validação rigorosa passou ({confidence:.2f})", confidence * 0.9
    
    def _validate_low_confidence(self, geometry: ObjectGeometry, confidence: float) -> Tuple[bool, str, float]:
        """Validação extrema para baixa confiança"""
        # Todas as validações anteriores + extras
        valid, reason, _ = self._validate_medium_confidence(geometry, confidence)
        if not valid:
            return False, reason, 0.1
        
        # Validações extras para baixa confiança
        if geometry.height < 100:
            return False, f"Muito baixo para pessoa ({geometry.height}px)", 0.1
        
        if geometry.aspect_ratio < 1.4:
            return False, f"Muito largo para pessoa (ratio: {geometry.aspect_ratio:.2f})", 0.1
        
        return True, f"Validação extrema passou ({confidence:.2f})", confidence * 0.8
    
    def _is_obviously_vehicle(self, geometry: ObjectGeometry) -> bool:
        """Detecta veículos óbvios"""
        ratio = geometry.width_height_ratio
        
        # Casos óbvios de veículos
        return (ratio > 2.5 or  # Muito largo
                geometry.aspect_ratio > 8.0 or  # Muito estreito
                geometry.height < 50)  # Muito baixo
    
    def _meets_size_constraints(self, geometry: ObjectGeometry) -> bool:
        """Valida restrições de tamanho"""
        constraints = self.config['human_constraints']
        return (constraints['min_height_px'] <= geometry.height <= constraints['max_height_px'] and
                constraints['min_width_px'] <= geometry.width <= constraints['max_width_px'])
    
    def _meets_proportion_constraints(self, geometry: ObjectGeometry) -> bool:
        """Valida restrições de proporção"""
        constraints = self.config['human_constraints']
        return (constraints['min_aspect_ratio'] <= geometry.aspect_ratio <= constraints['max_aspect_ratio'])
    
    def _detect_motorcycle_signature(self, geometry: ObjectGeometry) -> Tuple[bool, str]:
        """Detecta assinatura específica de motocicleta"""
        moto_config = self.config['motorcycle_detection']
        
        # Indicadores críticos
        indicators = []
        critical_indicators = moto_config['critical_indicators']
        
        # Indicador 1: Proporção típica de moto
        ratio = geometry.width_height_ratio
        if critical_indicators['width_height_ratio'][0] <= ratio <= critical_indicators['width_height_ratio'][1]:
            indicators.append(f"proporção_moto({ratio:.2f})")
        
        # Indicador 2: Área típica
        if critical_indicators['min_area'] <= geometry.area <= critical_indicators['max_area']:
            indicators.append(f"área_moto({geometry.area})")
        
        # Indicador 3: Largura mínima
        if geometry.width >= critical_indicators['min_width']:
            indicators.append(f"largura_moto({geometry.width})")
        
        # Indicador 4: Padrão suspeito (quase quadrado)
        suspicious = moto_config['suspicious_patterns']
        if suspicious['nearly_square'][0] <= ratio <= suspicious['nearly_square'][1]:
            indicators.append(f"quase_quadrado({ratio:.2f})")
        
        # Se 3+ indicadores, é moto
        is_motorcycle = len(indicators) >= 3
        reason = " + ".join(indicators) if indicators else "sem_indicadores"
        
        return is_motorcycle, reason

# ============================================================================
# VALIDADOR AVANÇADO DE EPIs
# ============================================================================

class SmartEPIValidator:
    """Validador inteligente de EPIs com análise anatômica"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações anatômicas
        self.anatomy = {
            'head_zone': 0.30,      # 30% superior = região da cabeça
            'torso_start': 0.25,    # 25% = início do torso
            'torso_end': 0.75,      # 75% = fim do torso
            
            # Tolerâncias de posicionamento
            'helmet_tolerance': 40,  # pixels de tolerância para capacete
            'vest_tolerance': 60,    # pixels de tolerância para colete
            
            # Cobertura mínima
            'helmet_min_coverage': 0.3,  # 30% da região da cabeça
            'vest_min_coverage': 0.4     # 40% da região do torso
        }
    
    def validate_epis_smart(self, people: List[Dict], helmets: List[Dict], 
                           vests: List[Dict]) -> List[Dict]:
        """Validação inteligente de EPIs com análise anatômica"""
        validated_people = []
        
        for person in people:
            px1, py1, px2, py2 = person['bbox']
            person_height = py2 - py1
            person_width = px2 - px1
            
            # Calcula regiões anatômicas
            head_region = self._calculate_head_region(person['bbox'])
            torso_region = self._calculate_torso_region(person['bbox'])
            
            # Validação avançada de capacete
            helmet_analysis = self._analyze_helmet_placement(person, helmets, head_region)
            
            # Validação avançada de colete
            vest_analysis = self._analyze_vest_placement(person, vests, torso_region)
            
            # Status final
            compliance_score = self._calculate_compliance_score(helmet_analysis, vest_analysis)
            final_status = self._determine_status(helmet_analysis, vest_analysis, compliance_score)
            
            validated_people.append({
                'person': person,
                'helmet_analysis': helmet_analysis,
                'vest_analysis': vest_analysis,
                'compliance_score': compliance_score,
                'status': final_status['status'],
                'color': final_status['color'],
                'recommendations': final_status['recommendations'],
                'head_region': head_region,
                'torso_region': torso_region
            })
        
        return validated_people
    
    def _calculate_head_region(self, person_bbox: List[int]) -> Dict:
        """Calcula região precisa da cabeça"""
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        width = x2 - x1
        
        # Região da cabeça (30% superior, mais estreita)
        head_height = int(height * self.anatomy['head_zone'])
        head_width = int(width * 0.7)  # Cabeça é mais estreita
        
        center_x = (x1 + x2) // 2
        head_x1 = center_x - head_width // 2
        head_x2 = center_x + head_width // 2
        
        return {
            'bbox': [head_x1, y1, head_x2, y1 + head_height],
            'center': (center_x, y1 + head_height // 2),
            'area': head_width * head_height
        }
    
    def _calculate_torso_region(self, person_bbox: List[int]) -> Dict:
        """Calcula região precisa do torso"""
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        
        # Região do torso (25% a 75% da altura)
        torso_start = int(height * self.anatomy['torso_start'])
        torso_end = int(height * self.anatomy['torso_end'])
        torso_height = torso_end - torso_start
        
        return {
            'bbox': [x1, y1 + torso_start, x2, y1 + torso_end],
            'center': ((x1 + x2) // 2, y1 + torso_start + torso_height // 2),
            'area': (x2 - x1) * torso_height
        }
    
    def _analyze_helmet_placement(self, person: Dict, helmets: List[Dict], head_region: Dict) -> Dict:
        """Análise avançada de posicionamento do capacete"""
        best_helmet = None
        best_score = 0.0
        analysis_details = []
        
        head_center = head_region['center']
        head_bbox = head_region['bbox']
        
        for helmet in helmets:
            hx1, hy1, hx2, hy2 = helmet['bbox']
            helmet_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)
            
            score = 0.0
            details = []
            
            # 1. Distância do centro da cabeça (40% do score)
            distance = np.sqrt((helmet_center[0] - head_center[0])**2 + 
                             (helmet_center[1] - head_center[1])**2)
            
            if distance <= self.anatomy['helmet_tolerance']:
                distance_score = 1.0 - (distance / self.anatomy['helmet_tolerance'])
                score += 0.4 * distance_score
                details.append(f"distância_ok({distance:.0f}px)")
            
            # 2. Sobreposição com região da cabeça (35% do score)
            overlap = self._calculate_iou(helmet['bbox'], head_bbox)
            if overlap >= self.anatomy['helmet_min_coverage']:
                score += 0.35 * min(1.0, overlap / self.anatomy['helmet_min_coverage'])
                details.append(f"cobertura_ok({overlap:.2f})")
            
            # 3. Posicionamento vertical (25% do score)
            if hy2 <= head_bbox[3]:  # Capacete não ultrapassa região da cabeça
                score += 0.25
                details.append("posição_vertical_ok")
            
            if score > best_score:
                best_score = score
                best_helmet = helmet
                analysis_details = details
        
        # Determina status
        if best_score > 0.7:
            status = "CORRECTLY_POSITIONED"
            is_compliant = True
        elif best_score > 0.4:
            status = "PARTIALLY_CORRECT"
            is_compliant = False
        else:
            status = "NOT_DETECTED_OR_INCORRECT"
            is_compliant = False
        
        return {
            'helmet': best_helmet,
            'score': best_score,
            'status': status,
            'is_compliant': is_compliant,
            'details': analysis_details,
            'confidence_level': 'high' if best_score > 0.8 else 'medium' if best_score > 0.5 else 'low'
        }
    
    def _analyze_vest_placement(self, person: Dict, vests: List[Dict], torso_region: Dict) -> Dict:
        """Análise avançada de posicionamento do colete"""
        best_vest = None
        best_score = 0.0
        analysis_details = []
        
        torso_center = torso_region['center']
        torso_bbox = torso_region['bbox']
        
        for vest in vests:
            vx1, vy1, vx2, vy2 = vest['bbox']
            vest_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
            
            score = 0.0
            details = []
            
            # 1. Distância do centro do torso (35% do score)
            distance = np.sqrt((vest_center[0] - torso_center[0])**2 + 
                             (vest_center[1] - torso_center[1])**2)
            
            if distance <= self.anatomy['vest_tolerance']:
                distance_score = 1.0 - (distance / self.anatomy['vest_tolerance'])
                score += 0.35 * distance_score
                details.append(f"distância_ok({distance:.0f}px)")
            
            # 2. Cobertura do torso (40% do score)
            overlap = self._calculate_iou(vest['bbox'], torso_bbox)
            if overlap >= self.anatomy['vest_min_coverage']:
                score += 0.40 * min(1.0, overlap / self.anatomy['vest_min_coverage'])
                details.append(f"cobertura_ok({overlap:.2f})")
            
            # 3. Posicionamento no torso (25% do score)
            if torso_bbox[1] <= vy1 and vy2 <= torso_bbox[3]:
                score += 0.25
                details.append("posição_torso_ok")
            
            if score > best_score:
                best_score = score
                best_vest = vest
                analysis_details = details
        
        # Determina status
        if best_score > 0.8:
            status = "CORRECTLY_POSITIONED"
            is_compliant = True
        elif best_score > 0.5:
            status = "PARTIALLY_CORRECT"
            is_compliant = False
        else:
            status = "NOT_DETECTED_OR_INCORRECT"
            is_compliant = False
        
        return {
            'vest': best_vest,
            'score': best_score,
            'status': status,
            'is_compliant': is_compliant,
            'details': analysis_details,
            'confidence_level': 'high' if best_score > 0.8 else 'medium' if best_score > 0.5 else 'low'
        }
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calcula Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Interseção
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        return intersection / (area1 + area2 - intersection)
    
    def _calculate_compliance_score(self, helmet_analysis: Dict, vest_analysis: Dict) -> float:
        """Calcula score de compliance combinado"""
        helmet_score = helmet_analysis['score'] if helmet_analysis['is_compliant'] else 0
        vest_score = vest_analysis['score'] if vest_analysis['is_compliant'] else 0
        
        # Capacete tem peso maior (60%) por ser mais crítico
        return helmet_score * 0.6 + vest_score * 0.4
    
    def _determine_status(self, helmet_analysis: Dict, vest_analysis: Dict, 
                         compliance_score: float) -> Dict:
        """Determina status final com recomendações"""
        helmet_ok = helmet_analysis['is_compliant']
        vest_ok = vest_analysis['is_compliant']
        
        recommendations = []
        
        if helmet_ok and vest_ok:
            status = "FULL_COMPLIANCE"
            color = (0, 255, 0)
        elif helmet_ok or vest_ok:
            status = "PARTIAL_COMPLIANCE"
            color = (0, 255, 255)
            
            if not helmet_ok:
                recommendations.append(f"Capacete: {helmet_analysis['status']}")
            if not vest_ok:
                recommendations.append(f"Colete: {vest_analysis['status']}")
        else:
            status = "CRITICAL_VIOLATION"
            color = (0, 0, 255)
            recommendations.extend(["CAPACETE OBRIGATÓRIO", "COLETE OBRIGATÓRIO"])
        
        return {
            'status': status,
            'color': color,
            'recommendations': recommendations
        }

# ============================================================================
# INTERFACE PRINCIPAL ULTRA-OTIMIZADA
# ============================================================================

class UltraOptimizedEPIInterface:
    """Interface completa ultra-otimizada"""
    
    def __init__(self):
        """Inicializa interface ultra-otimizada"""
        # Logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Componentes otimizados
        self.human_validator = AdvancedHumanValidator()
        self.epi_validator = SmartEPIValidator()
        
        # Modelo
        self.model = None
        self.load_model()
        
        # Interface
        self.cap = None
        self.is_camera_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Fullscreen
        self.is_fullscreen = False
        self.fullscreen_window = None
        
        # Estatísticas
        self.frame_count = 0
        self.start_time = time.time()
        self.rejected_objects = 0
        self.processing_times = deque(maxlen=30)
        
        # Vídeo
        self.is_video_playing = False
        self.video_paused = False
        self.current_frame = 0
        self.current_display_frame = None
        
        # Classes
        self.classes = ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']
        
        # Cores
        self.colors = {
            'person': (255, 255, 0),
            'helmet': (0, 255, 0),
            'vest': (0, 255, 0),
            'no-helmet': (0, 0, 255),
            'no-vest': (0, 0, 255)
        }
        
        self.create_interface()
    
    def load_model(self):
        """Carrega modelo otimizado"""
        try:
            weights_paths = [
                "yolov5/runs/train/epi_safe_fine_tuned/weights/best.pt",
                "yolov5/runs/train/epi_detection_v24/weights/best.pt"
            ]
            
            weights_path = None
            for path in weights_paths:
                if Path(path).exists():
                    weights_path = path
                    break
            
            if not weights_path:
                self.logger.error("Nenhum modelo encontrado")
                return False
            
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
            
            # Configurações ultra-otimizadas
            self.model.conf = 0.12
            self.model.iou = 0.20
            
            self.logger.info(f"Modelo carregado: {weights_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {e}")
            return False
    
    def create_interface(self):
        """Cria interface gráfica"""
        self.root = tk.Tk()
        self.root.title("EPI Detection - Ultra-Optimized System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Teclas globais
        self.root.bind('<Escape>', self.exit_fullscreen)
        self.root.bind('<F11>', self.toggle_fullscreen)
        self.root.bind('<KeyPress>', self.handle_keypress)
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title = tk.Label(main_frame, text="DETECÇÃO DE EPIs - SISTEMA ULTRA-OTIMIZADO", 
                        font=('Arial', 18, 'bold'), fg='white', bg='#2b2b2b')
        title.pack(pady=(0, 10))
        
        # Controles
        self.create_controls(main_frame)
        
        # Visualização
        self.create_display_area(main_frame)
        
        # Status
        self.create_status_area(main_frame)
        
        # Configurações
        self.create_config_area(main_frame)
        
        self.update_interface()
    
    def create_controls(self, parent):
        """Cria controles principais"""
        controls_frame = tk.Frame(parent, bg='#3b3b3b', relief='raised', bd=2)
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        
        btn_frame = tk.Frame(controls_frame, bg='#3b3b3b')
        btn_frame.pack(pady=10)
        
        # Botões principais
        buttons = [
            ("📸 IMAGEM", self.detect_image, '#4CAF50'),
            ("🎥 VÍDEO", self.detect_video, '#2196F3'),
            ("📹 CÂMERA", self.toggle_camera, '#FF9800'),
            ("🖥️ FULLSCREEN", self.toggle_fullscreen, '#9C27B0'),
            ("⏹️ PARAR", self.stop_all, '#f44336')
        ]
        
        self.buttons = {}
        for text, command, color in buttons:
            btn = tk.Button(btn_frame, text=text, command=command,
                           font=('Arial', 11, 'bold'), bg=color, fg='white',
                           relief='raised', bd=3, padx=15, pady=8)
            btn.pack(side=tk.LEFT, padx=8)
            self.buttons[text.split()[0]] = btn
    
    def create_display_area(self, parent):
        """Cria área de visualização"""
        view_frame = tk.Frame(parent, bg='#1b1b1b', relief='sunken', bd=2)
        view_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(view_frame, bg='#1b1b1b', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_status_area(self, parent):
        """Cria área de status"""
        status_frame = tk.Frame(parent, bg='#3b3b3b', relief='raised', bd=2)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.status_label = tk.Label(status_frame, text="Status: Sistema inicializado", 
                                   font=('Arial', 11), fg='white', bg='#3b3b3b')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0", 
                                font=('Arial', 11), fg='white', bg='#3b3b3b')
        self.fps_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        self.detection_label = tk.Label(status_frame, text="Detecções: 0", 
                                      font=('Arial', 11), fg='white', bg='#3b3b3b')
        self.detection_label.pack(side=tk.LEFT, padx=10, pady=8)
    
    def create_config_area(self, parent):
        """Cria área de configurações"""
        config_frame = tk.Frame(parent, bg='#3b3b3b', relief='raised', bd=2)
        config_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Confidence
        tk.Label(config_frame, text="Confidence:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(10, 5), pady=8)
        
        self.conf_var = tk.DoubleVar(value=0.12)
        self.conf_scale = tk.Scale(config_frame, from_=0.05, to=0.8, resolution=0.05,
                                  orient=tk.HORIZONTAL, variable=self.conf_var,
                                  bg='#3b3b3b', fg='white')
        self.conf_scale.pack(side=tk.LEFT, padx=(0, 15), pady=8)
        
        # IoU
        tk.Label(config_frame, text="IoU:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(0, 5), pady=8)
        
        self.iou_var = tk.DoubleVar(value=0.20)
        self.iou_scale = tk.Scale(config_frame, from_=0.1, to=0.8, resolution=0.05,
                                 orient=tk.HORIZONTAL, variable=self.iou_var,
                                 bg='#3b3b3b', fg='white')
        self.iou_scale.pack(side=tk.LEFT, padx=(0, 15), pady=8)
        
        # Aplicar
        apply_btn = tk.Button(config_frame, text="APLICAR", command=self.apply_config,
                             font=('Arial', 10, 'bold'), bg='#4CAF50', fg='white',
                             relief='raised', bd=2, padx=12, pady=5)
        apply_btn.pack(side=tk.LEFT, padx=(0, 10), pady=8)
        
        # Bind eventos
        self.conf_scale.bind("<ButtonRelease-1>", self.on_config_change)
        self.iou_scale.bind("<ButtonRelease-1>", self.on_config_change)
    
    def on_config_change(self, event):
        """Atualiza configurações em tempo real"""
        if self.model:
            self.model.conf = self.conf_var.get()
            self.model.iou = self.iou_var.get()
    
    def apply_config(self):
        """Aplica configurações"""
        if self.model:
            self.model.conf = self.conf_var.get()
            self.model.iou = self.iou_var.get()
            messagebox.showinfo("Config", f"Conf: {self.model.conf:.2f} | IoU: {self.model.iou:.2f}")
    
    def detect_image(self):
        """Detecta EPIs em imagem"""
        if not self.model:
            messagebox.showerror("Erro", "Modelo não carregado")
            return
        
        file_path = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Erro", "Imagem inválida")
                return
            
            self.rejected_objects = 0
            result_frame, analysis = self.process_frame_optimized(image)
            
            self.display_image(result_frame)
            self.current_display_frame = result_frame
            
            # Status
            people_count = len([p for p in analysis.get('people_analysis', [])])
            try:
                if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                    self.status_label.config(text=f"Imagem processada - {people_count} pessoas")
                if hasattr(self, 'detection_label') and self.detection_label.winfo_exists():
                    self.detection_label.config(text=f"Humanos: {people_count} | Rejeitados: {self.rejected_objects}")
            except tk.TclError:
                # Widgets já foram destruídos, ignora o erro
                pass
            
            self.save_result(result_frame, "image")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro: {e}")
    
    def detect_video(self):
        """Detecta EPIs em vídeo"""
        if not self.model:
            messagebox.showerror("Erro", "Modelo não carregado")
            return
        
        file_path = filedialog.askopenfilename(
            title="Selecionar vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov"), ("Todos", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                messagebox.showerror("Erro", "Vídeo inválido")
                return
            
            self.process_video(cap, file_path)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro: {e}")
    
    def process_video(self, cap, file_path):
        """Processa vídeo"""
        self.is_video_playing = True
        self.video_paused = False
        self.current_frame = 0
        self.video_cap = cap
        self.rejected_objects = 0
        
        self.create_video_controls()
        
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def create_video_controls(self):
        """Cria controles de vídeo"""
        self.video_controls = tk.Frame(self.root, bg='#3b3b3b', relief='raised', bd=2)
        self.video_controls.pack(fill=tk.X, pady=(10, 0))
        
        btn_frame = tk.Frame(self.video_controls, bg='#3b3b3b')
        btn_frame.pack(pady=8)
        
        self.play_btn = tk.Button(btn_frame, text="PAUSAR", command=self.toggle_video_pause,
                                 font=('Arial', 10, 'bold'), bg='#FF9800', fg='white')
        self.play_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_video_btn = tk.Button(btn_frame, text="PARAR VIDEO", command=self.stop_video,
                                       font=('Arial', 10, 'bold'), bg='#f44336', fg='white')
        self.stop_video_btn.pack(side=tk.LEFT, padx=10)
        
        self.video_status = tk.Label(btn_frame, text="Reproduzindo...", 
                                    font=('Arial', 10), fg='white', bg='#3b3b3b')
        self.video_status.pack(side=tk.LEFT, padx=20)
    
    def video_loop(self):
        """Loop de processamento de vídeo"""
        try:
            target_fps = 30
            frame_delay = 1.0 / target_fps
            
            while self.is_video_playing and self.video_cap.isOpened():
                if not self.video_paused:
                    ret, frame = self.video_cap.read()
                    if not ret:
                        break
                    
                    result_frame, analysis = self.process_frame_optimized(frame)
                    self.current_display_frame = result_frame
                    
                    self.root.after(0, self.update_video_display, result_frame, analysis)
                    
                    time.sleep(frame_delay)
                    self.current_frame += 1
                else:
                    time.sleep(0.1)
        
        except Exception as e:
            self.logger.error(f"Erro no vídeo: {e}")
        finally:
            if hasattr(self, 'video_cap'):
                self.video_cap.release()
            self.is_video_playing = False
            self.root.after(0, self.cleanup_video)
    
    def update_video_display(self, frame, analysis):
        """Atualiza display do vídeo"""
        try:
            self.display_image(frame)
            
            if self.is_fullscreen and hasattr(self, 'fullscreen_canvas'):
                self.display_image_fullscreen(frame)
            
            people_count = len(analysis.get('people_analysis', []))
            try:
                if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                    self.status_label.config(text=f"Frame {self.current_frame}")
                if hasattr(self, 'detection_label') and self.detection_label.winfo_exists():
                    self.detection_label.config(text=f"Pessoas: {people_count} | Rejeitados: {self.rejected_objects}")
            except tk.TclError:
                # Widgets já foram destruídos, ignora o erro
                pass
            
        except Exception as e:
            self.logger.error(f"Erro ao atualizar display: {e}")
    
    def toggle_video_pause(self):
        """Pausa/resume vídeo"""
        self.video_paused = not self.video_paused
        self.play_btn.config(text="CONTINUAR" if self.video_paused else "PAUSAR")
    
    def stop_video(self):
        """Para vídeo"""
        self.is_video_playing = False
        if self.is_fullscreen:
            self.exit_fullscreen()
    
    def cleanup_video(self):
        """Remove controles de vídeo"""
        try:
            if hasattr(self, 'video_controls'):
                self.video_controls.destroy()
        except:
            pass
    
    def toggle_camera(self):
        """Liga/desliga câmera"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Inicia câmera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Câmera não disponível")
                return
            
            # Configurações
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_camera_running = True
            # Verifica se os widgets ainda existem antes de configurá-los
            try:
                if hasattr(self, 'buttons') and "📹" in self.buttons and self.buttons["📹"].winfo_exists():
                    self.buttons["📹"].config(text="📹 PARAR CÂMERA", bg='#f44336')
            except tk.TclError:
                # Widgets já foram destruídos, ignora o erro
                pass
            
            self.frame_count = 0
            self.rejected_objects = 0
            self.start_time = time.time()
            
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            try:
                if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                    self.status_label.config(text="Câmera ativa")
            except tk.TclError:
                # Widgets já foram destruídos, ignora o erro
                pass
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na câmera: {e}")
    
    def camera_loop(self):
        """Loop da câmera"""
        while self.is_camera_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                result_frame, analysis = self.process_frame_optimized(frame)
                self.current_display_frame = result_frame
                
                if not self.frame_queue.full():
                    self.frame_queue.put((result_frame, analysis))
                
                self.frame_count += 1
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                people_count = len(analysis.get('people_analysis', []))
                self.root.after(0, self.update_camera_display, fps, people_count, result_frame)
            
            time.sleep(0.033)
    
    def update_camera_display(self, fps, people_count, frame):
        """Atualiza display da câmera"""
        try:
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.detection_label.config(text=f"Pessoas: {people_count} | Rejeitados: {self.rejected_objects}")
            
            if not self.frame_queue.empty():
                display_frame, analysis = self.frame_queue.get_nowait()
                self.display_image(display_frame)
                
                if self.is_fullscreen:
                    self.display_image_fullscreen(display_frame)
        except:
            pass
    
    def stop_camera(self):
        """Para câmera"""
        self.is_camera_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.is_fullscreen:
            self.exit_fullscreen()
        
        # Verifica se os widgets ainda existem antes de configurá-los
        try:
            if hasattr(self, 'buttons') and "📹" in self.buttons and self.buttons["📹"].winfo_exists():
                self.buttons["📹"].config(text="📹 CÂMERA", bg='#FF9800')
            if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                self.status_label.config(text="Câmera parada")
        except tk.TclError:
            # Widgets já foram destruídos, ignora o erro
            pass
    
    def stop_all(self):
        """Para todos os processos"""
        self.stop_camera()
        self.stop_video()
        if self.is_fullscreen:
            self.exit_fullscreen()
    
    def toggle_fullscreen(self, event=None):
        """Liga/desliga fullscreen"""
        if not self.is_fullscreen:
            self.enter_fullscreen()
        else:
            self.exit_fullscreen()
    
    def enter_fullscreen(self):
        """Entra em fullscreen"""
        if self.is_fullscreen:
            return
        
        self.is_fullscreen = True
        
        self.fullscreen_window = tk.Toplevel(self.root)
        self.fullscreen_window.title("EPI Detection - Fullscreen")
        self.fullscreen_window.configure(bg='black')
        self.fullscreen_window.attributes('-fullscreen', True)
        
        self.fullscreen_window.bind('<Escape>', self.exit_fullscreen)
        self.fullscreen_window.bind('<F11>', self.exit_fullscreen)
        self.fullscreen_window.focus_set()
        
        self.fullscreen_canvas = tk.Canvas(self.fullscreen_window, bg='black')
        self.fullscreen_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Verifica se os widgets ainda existem antes de configurá-los
        try:
            if hasattr(self, 'buttons') and "🖥️" in self.buttons and self.buttons["🖥️"].winfo_exists():
                self.buttons["🖥️"].config(text="🖥️ SAIR FULL", bg='#E91E63')
        except tk.TclError:
            # Widgets já foram destruídos, ignora o erro
            pass
        
        instructions = tk.Label(self.fullscreen_window, 
                               text="ESC ou F11: Sair Fullscreen", 
                               font=('Arial', 14), fg='white', bg='black')
        instructions.place(relx=0.5, y=30, anchor='center')
    
    def exit_fullscreen(self, event=None):
        """Sai do fullscreen"""
        if not self.is_fullscreen:
            return
        
        self.is_fullscreen = False
        
        if self.fullscreen_window:
            self.fullscreen_window.destroy()
            self.fullscreen_window = None
            self.fullscreen_canvas = None
        
        # Verifica se os widgets ainda existem antes de configurá-los
        try:
            if hasattr(self, 'buttons') and "🖥️" in self.buttons and self.buttons["🖥️"].winfo_exists():
                self.buttons["🖥️"].config(text="🖥️ FULLSCREEN", bg='#9C27B0')
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.focus_set()
        except tk.TclError:
            # Widgets já foram destruídos, ignora o erro
            pass
    
    def process_frame_optimized(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Processamento otimizado com TODAS as detecções visíveis"""
        start_time = time.time()
        
        # Detecção YOLO - ARMAZENA TODAS as detecções
        raw_detections = self.detect_yolo_objects(frame)
        self._current_raw_detections = raw_detections  # Armazena para visualização
        
        # Log das detecções brutas
        self.logger.debug(f"Detecções YOLO brutas: {len(raw_detections)}")
        for det in raw_detections:
            self.logger.debug(f"  {det['class_name']}: conf={det['confidence']:.2f}, bbox={det['bbox']}")
        
        # Filtro anti-veículo APENAS para pessoas
        validated_detections = self.apply_vehicle_filter(raw_detections)
        
        # Log das detecções validadas
        self.logger.debug(f"Detecções após filtro: {len(validated_detections)}")
        
        # Separação por classe APÓS filtro
        people = [d for d in validated_detections if d['class_name'] == 'person']
        helmets = [d for d in validated_detections if d['class_name'] == 'helmet']
        vests = [d for d in validated_detections if d['class_name'] == 'vest']
        no_helmets = [d for d in validated_detections if d['class_name'] == 'no-helmet']
        no_vests = [d for d in validated_detections if d['class_name'] == 'no-vest']
        
        self.logger.debug(f"Pessoas: {len(people)}, Capacetes: {len(helmets)}, Coletes: {len(vests)}")
        self.logger.debug(f"Infrações - No-helmet: {len(no_helmets)}, No-vest: {len(no_vests)}")
        
        # Validação avançada de EPIs
        epi_analysis = self.epi_validator.validate_epis_smart(people, helmets, vests)
        
        # Desenha TODOS os resultados
        result_frame = self.draw_advanced_results(frame, epi_analysis)
        
        # Performance
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return result_frame, {
            'people_analysis': epi_analysis, 
            'processing_time': processing_time,
            'raw_detections_count': len(raw_detections),
            'validated_detections_count': len(validated_detections),
            'detection_breakdown': {
                'people': len(people),
                'helmets': len(helmets),
                'vests': len(vests),
                'no_helmets': len(no_helmets),
                'no_vests': len(no_vests)
            }
        }
    
    def detect_yolo_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detecção YOLO otimizada com logging detalhado"""
        if not self.model:
            return []
        
        try:
            # Executa detecção YOLO
            results = self.model(frame)
            detections = []
            
            self.logger.debug(f"YOLO executado. Verificando resultados...")
            
            if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                detections_array = results.xyxy[0].cpu().numpy()
                self.logger.debug(f"Detecções brutas encontradas: {len(detections_array)}")
                
                for i, detection in enumerate(detections_array):
                    x1, y1, x2, y2, conf, cls = detection
                    class_id = int(cls)
                    
                    if 0 <= class_id < len(self.classes):
                        class_name = self.classes[class_id]
                        
                        detection_dict = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_id': class_id,
                            'class_name': class_name
                        }
                        
                        detections.append(detection_dict)
                        
                        # Log detalhado
                        bbox_str = f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
                        self.logger.debug(f"  Detecção {i}: {class_name} conf={conf:.3f} bbox={bbox_str}")
                    else:
                        self.logger.warning(f"Classe inválida detectada: {class_id}")
            else:
                self.logger.debug("Nenhuma detecção YOLO encontrada")
            
            self.logger.info(f"TOTAL de detecções YOLO: {len(detections)}")
            
            # Breakdown por classe
            class_counts = {}
            for det in detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            self.logger.info(f"Breakdown: {class_counts}")
            
            return detections
        
        except Exception as e:
            self.logger.error(f"Erro na detecção YOLO: {e}")
            return []
    
    def apply_vehicle_filter(self, detections: List[Dict]) -> List[Dict]:
        """Aplica filtro anti-veículo APENAS para pessoas, mantém todos os EPIs"""
        validated = []
        
        for detection in detections:
            if detection['class_name'] == 'person':
                is_valid, reason, confidence = self.human_validator.is_valid_human(detection)
                
                if is_valid:
                    detection['validation_reason'] = reason
                    detection['adjusted_confidence'] = confidence
                    validated.append(detection)
                    self.logger.debug(f"Pessoa validada: {reason}")
                else:
                    self.rejected_objects += 1
                    self.logger.info(f"Pessoa rejeitada: {reason}")
            else:
                # TODOS os EPIs passam direto (helmet, vest, no-helmet, no-vest)
                validated.append(detection)
                self.logger.debug(f"EPI detectado: {detection['class_name']} conf={detection['confidence']:.2f}")
        
        return validated
    
    def draw_advanced_results(self, frame: np.ndarray, epi_analysis: List[Dict]) -> np.ndarray:
        """Desenha resultados avançados com TODOS os EPIs detectados"""
        result = frame.copy()
        
        # PRIMEIRO: Desenha todas as detecções YOLO individuais
        if hasattr(self, '_current_raw_detections'):
            for detection in self._current_raw_detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                conf = detection['confidence']
                
                # Define cor por classe
                if class_name == 'person':
                    color = (255, 255, 0)  # Amarelo para pessoa
                elif class_name == 'helmet':
                    color = (0, 255, 0)    # Verde para capacete
                elif class_name == 'vest':
                    color = (0, 255, 0)    # Verde para colete
                elif class_name == 'no-helmet':
                    color = (0, 0, 255)    # Vermelho para infração capacete
                elif class_name == 'no-vest':
                    color = (0, 0, 255)    # Vermelho para infração colete
                else:
                    color = (128, 128, 128)  # Cinza para outros
                
                # Desenha bounding box do EPI/objeto
                thickness = 3 if class_name in ['no-helmet', 'no-vest'] else 2
                cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
                
                # Label do EPI/objeto
                label = f"{class_name.upper()}: {conf:.2f}"
                
                # Background para o texto
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result, (x1, y1 - text_height - 10), (x1 + text_width + 5, y1), color, -1)
                
                # Texto do label
                cv2.putText(result, label, (x1 + 2, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Adiciona ícone de alerta para infrações
                if class_name in ['no-helmet', 'no-vest']:
                    cv2.putText(result, "⚠", (x2 - 25, y1 + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # SEGUNDO: Desenha análise das pessoas validadas
        for person_data in epi_analysis:
            person = person_data['person']
            px1, py1, px2, py2 = person['bbox']
            
            # Desenha pessoa com borda mais espessa
            cv2.rectangle(result, (px1, py1), (px2, py2), (255, 255, 0), 3)
            
            # DESENHA REGIÕES ANATÔMICAS (debug visual)
            if 'head_region' in person_data:
                head_bbox = person_data['head_region']['bbox']
                cv2.rectangle(result, (head_bbox[0], head_bbox[1]), (head_bbox[2], head_bbox[3]), 
                             (255, 0, 255), 1)  # Magenta para região da cabeça
                cv2.putText(result, "HEAD", (head_bbox[0], head_bbox[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            if 'torso_region' in person_data:
                torso_bbox = person_data['torso_region']['bbox']
                cv2.rectangle(result, (torso_bbox[0], torso_bbox[1]), (torso_bbox[2], torso_bbox[3]), 
                             (0, 255, 255), 1)  # Ciano para região do torso
                cv2.putText(result, "TORSO", (torso_bbox[0], torso_bbox[1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # DESTACA EPIs ESPECÍFICOS desta pessoa
            helmet_info = person_data['helmet_analysis']
            if helmet_info['helmet']:
                hx1, hy1, hx2, hy2 = helmet_info['helmet']['bbox']
                helmet_color = (0, 255, 0) if helmet_info['is_compliant'] else (255, 0, 0)
                cv2.rectangle(result, (hx1, hy1), (hx2, hy2), helmet_color, 3)  # Borda espessa
                
                # Conecta capacete à pessoa com linha
                person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                helmet_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)
                cv2.line(result, person_center, helmet_center, helmet_color, 2)
            
            vest_info = person_data['vest_analysis']
            if vest_info['vest']:
                vx1, vy1, vx2, vy2 = vest_info['vest']['bbox']
                vest_color = (0, 255, 0) if vest_info['is_compliant'] else (255, 0, 0)
                cv2.rectangle(result, (vx1, vy1), (vx2, vy2), vest_color, 3)  # Borda espessa
                
                # Conecta colete à pessoa com linha
                person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
                vest_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
                cv2.line(result, person_center, vest_center, vest_color, 2)
            
            # Status da pessoa
            status = person_data['status']
            color = person_data['color']
            compliance = person_data['compliance_score']
            
            # Label principal da pessoa
            cv2.putText(result, f"PESSOA: {status} ({compliance:.0%})", 
                       (px1, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Detalhes dos EPIs na parte inferior
            y_offset = py2 + 25
            
            # Status do capacete
            helmet_text = f"Capacete: {helmet_info['status']} (score: {helmet_info['score']:.2f})"
            helmet_color = (0, 255, 0) if helmet_info['is_compliant'] else (0, 0, 255)
            cv2.putText(result, helmet_text, (px1, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, helmet_color, 2)
            
            # Status do colete
            vest_text = f"Colete: {vest_info['status']} (score: {vest_info['score']:.2f})"
            vest_color = (0, 255, 0) if vest_info['is_compliant'] else (0, 0, 255)
            cv2.putText(result, vest_text, (px1, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, vest_color, 2)
            
            # Recomendações
            recommendations = person_data.get('recommendations', [])
            for i, rec in enumerate(recommendations[:2]):
                cv2.putText(result, f"• {rec}", (px1, y_offset + 45 + i*18), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Sistema info
        self.add_system_overlay(result, epi_analysis)
        
        return result
    
    def add_system_overlay(self, frame: np.ndarray, analysis: List[Dict]):
        """Adiciona overlay do sistema"""
        # Calcula estatísticas
        total = len(analysis)
        compliant = len([p for p in analysis if p['status'] == 'FULL_COMPLIANCE'])
        partial = len([p for p in analysis if p['status'] == 'PARTIAL_COMPLIANCE'])
        violations = total - compliant - partial
        
        compliance_rate = (compliant / total * 100) if total > 0 else 0
        avg_fps = len(self.processing_times) / sum(self.processing_times) if self.processing_times else 0
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Informações
        info_lines = [
            "SISTEMA ULTRA-OTIMIZADO DE EPIs",
            f"Pessoas: {total} | Compliance: {compliance_rate:.1f}%",
            f"Compliant: {compliant} | Parcial: {partial} | Violações: {violations}",
            f"FPS: {avg_fps:.1f} | Rejeitados: {self.rejected_objects}",
            "F11: Fullscreen | S: Screenshot | Q: Sair"
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if i == 0 else (255, 255, 255) if i < 4 else (0, 255, 255)
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, line, (15, 35 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    
    def display_image(self, image):
        """Exibe imagem no canvas"""
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                h, w = image_rgb.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                image_resized = cv2.resize(image_rgb, (new_w, new_h))
                pil_image = Image.fromarray(image_resized)
                self.current_image = ImageTk.PhotoImage(pil_image)
                
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                       image=self.current_image, anchor=tk.CENTER)
        except Exception as e:
            self.logger.error(f"Erro ao exibir imagem: {e}")
    
    def display_image_fullscreen(self, image):
        """Exibe imagem em fullscreen"""
        if not hasattr(self, 'fullscreen_canvas') or not self.fullscreen_canvas:
            return
        
        try:
            screen_width = self.fullscreen_window.winfo_screenwidth()
            screen_height = self.fullscreen_window.winfo_screenheight()
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            scale = min(screen_width / w, screen_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
            pil_image = Image.fromarray(image_resized)
            self.fullscreen_image = ImageTk.PhotoImage(pil_image)
            
            self.fullscreen_canvas.delete("all")
            self.fullscreen_canvas.create_image(screen_width//2, screen_height//2, 
                                               image=self.fullscreen_image, anchor=tk.CENTER)
        except Exception as e:
            self.logger.error(f"Erro fullscreen: {e}")
    
    def save_result(self, image, source_type):
        """Salva resultado"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"epi_detection_{source_type}_{timestamp}.jpg"
        
        Path("output").mkdir(exist_ok=True)
        output_path = Path("output") / filename
        
        cv2.imwrite(str(output_path), image)
        self.logger.info(f"Salvo: {output_path}")
    
    def handle_keypress(self, event):
        """Manipula teclas"""
        key = event.keysym.lower()
        
        if key == 'f11':
            self.toggle_fullscreen()
        elif key == 'escape':
            if self.is_fullscreen:
                self.exit_fullscreen()
        elif key == 's':
            if self.current_display_frame is not None:
                self.save_result(self.current_display_frame, "screenshot")
        elif key == 'q':
            if self.is_fullscreen:
                self.exit_fullscreen()
            else:
                self.root.quit()
    
    def update_interface(self):
        """Atualização periódica da interface"""
        try:
            if self.root.winfo_exists():
                self.root.after(100, self.update_interface)
        except:
            pass
    
    def run(self):
        """Executa interface"""
        try:
            self.root.focus_set()
            self.root.mainloop()
        except KeyboardInterrupt:
            self.logger.info("Interface interrompida")
        finally:
            self.stop_all()

def main():
    """Função principal"""
    print("SISTEMA FUNCIONAL DE DETECÇÃO DE EPIs - ULTRA-OTIMIZADO")
    print("=" * 60)
    print("Funcionalidades:")
    print("  - Sistema anti-veículo robusto")
    print("  - Validação anatômica de EPIs")
    print("  - Fullscreen (F11)")
    print("  - Performance adaptativa")
    print("  - Tracking inteligente")
    print("=" * 60)
    print("Configurações otimizadas:")
    print("  - Confidence: 0.12 (ultra-sensível)")
    print("  - IoU: 0.20 (permite sobreposições)")
    print("  - Anti-moto: ativo")
    print("=" * 60)
    
    interface = UltraOptimizedEPIInterface()
    interface.run()

if __name__ == "__main__":
    main()