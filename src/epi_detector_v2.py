#!/usr/bin/env python3
"""
Sistema de Detecção Automática de EPIs em Canteiros de Obra
Detecta e valida se capacete e colete estão fisicamente na pessoa
"""

import cv2
import numpy as np
import torch
import torch.hub
import yaml
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

class EPIStatus(Enum):
    """Status dos EPIs"""
    CORRETO = "CORRETO"
    AUSENTE = "AUSENTE"
    INCORRETO = "INCORRETO"

@dataclass
class Detection:
    """Estrutura para detecções"""
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    status: EPIStatus = EPIStatus.CORRETO

@dataclass
class Person:
    """Estrutura para pessoa detectada"""
    bbox: List[int]
    confidence: float
    helmet_status: EPIStatus = EPIStatus.AUSENTE
    vest_status: EPIStatus = EPIStatus.AUSENTE
    helmet_detection: Detection = None
    vest_detection: Detection = None

class EPIDetectorV2:
    """
    Detector de EPIs versão 2.0
    Foco: Validação física dos equipamentos na pessoa
    """
    
    def __init__(self, config_path: str = "config/epi_config_v2.yaml"):
        """Inicializa o detector de EPIs"""
        # Carrega configurações
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Configura logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Carrega modelo YOLOv5
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.config['model']['weights'])
        self.device = self.config['model']['device']
        
        # Configurações de detecção
        self.conf_threshold = self.config['model']['conf_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        
        # Aplica configurações ao modelo YOLOv5
        self.model.conf = self.conf_threshold
        self.model.iou = self.iou_threshold
        
        # Classes para detecção
        self.classes = ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']
        
        # Cores para visualização
        self.colors = {
            'person': (255, 255, 0),        # Amarelo
            'helmet_correct': (0, 255, 0),   # Verde
            'helmet_violation': (0, 0, 255), # Vermelho
            'vest_correct': (0, 255, 0),     # Verde
            'vest_violation': (0, 0, 255),   # Vermelho
            'no_helmet': (0, 0, 255),        # Vermelho
            'no_vest': (0, 0, 255)           # Vermelho
        }
        
        self.logger.info(f"Detector de EPIs V2 inicializado com dispositivo: {self.device}")
    
    def detect_objects(self, image: np.ndarray) -> List[Detection]:
        """Detecta todos os objetos na imagem"""
        # Executa detecção YOLOv5
        results = self.model(image)
        
        detections = []
        if len(results) > 0:
            result = results[0]  # Primeiro resultado
            if result is not None:
                for detection in result:
                    x1, y1, x2, y2, conf, cls = detection
                    
                    detections.append(Detection(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=float(conf),
                        class_id=int(cls),
                        class_name=self.classes[int(cls)]
                    ))
        
        return detections
    
    def validate_epi_placement(self, detections: List[Detection]) -> List[Person]:
        """
        Valida se os EPIs estão fisicamente na pessoa
        Regras fundamentais:
        - Capacete deve estar na cabeça (topo 25% da pessoa)
        - Colete deve estar no torso (30-70% da altura da pessoa)
        """
        # Separa detecções por tipo
        people = [d for d in detections if d.class_name == 'person']
        helmets = [d for d in detections if d.class_name == 'helmet']
        vests = [d for d in detections if d.class_name == 'vest']
        no_helmets = [d for d in detections if d.class_name == 'no-helmet']
        no_vests = [d for d in detections if d.class_name == 'no-vest']
        
        validated_people = []
        
        for person in people:
            person_obj = Person(
                bbox=person.bbox,
                confidence=person.confidence
            )
            
            # Valida capacete
            helmet_status, helmet_det = self._validate_helmet_placement(person, helmets)
            person_obj.helmet_status = helmet_status
            person_obj.helmet_detection = helmet_det
            
            # Valida colete
            vest_status, vest_det = self._validate_vest_placement(person, vests)
            person_obj.vest_status = vest_status
            person_obj.vest_detection = vest_det
            
            validated_people.append(person_obj)
        
        return validated_people
    
    def _validate_helmet_placement(self, person: Detection, helmets: List[Detection]) -> Tuple[EPIStatus, Detection]:
        """Valida se capacete está na cabeça da pessoa"""
        px1, py1, px2, py2 = person.bbox
        person_height = py2 - py1
        
        # Zona da cabeça: topo 25% da pessoa
        head_zone_top = py1
        head_zone_bottom = py1 + int(person_height * 0.25)
        
        best_helmet = None
        best_overlap = 0
        
        for helmet in helmets:
            hx1, hy1, hx2, hy2 = helmet.bbox
            
            # Verifica se capacete está na zona da cabeça
            if hy2 <= head_zone_bottom:
                # Calcula sobreposição com a zona da cabeça
                overlap = self._calculate_overlap(
                    [hx1, hy1, hx2, hy2],
                    [px1, head_zone_top, px2, head_zone_bottom]
                )
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_helmet = helmet
        
        if best_helmet and best_overlap > self.config['validation']['helmet_overlap_threshold']:
            return EPIStatus.CORRETO, best_helmet
        else:
            return EPIStatus.AUSENTE, None
    
    def _validate_vest_placement(self, person: Detection, vests: List[Detection]) -> Tuple[EPIStatus, Detection]:
        """Valida se colete está no torso da pessoa"""
        px1, py1, px2, py2 = person.bbox
        person_height = py2 - py1
        
        # Zona do torso: 30-70% da altura da pessoa
        torso_zone_top = py1 + int(person_height * 0.30)
        torso_zone_bottom = py1 + int(person_height * 0.70)
        
        best_vest = None
        best_overlap = 0
        
        for vest in vests:
            vx1, vy1, vx2, vy2 = vest.bbox
            
            # Verifica se colete está na zona do torso
            if vy1 >= torso_zone_top and vy2 <= torso_zone_bottom:
                # Calcula sobreposição com a zona do torso
                overlap = self._calculate_overlap(
                    [vx1, vy1, vx2, vy2],
                    [px1, torso_zone_top, px2, torso_zone_bottom]
                )
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_vest = vest
        
        if best_vest and best_overlap > self.config['validation']['vest_overlap_threshold']:
            return EPIStatus.CORRETO, best_vest
        else:
            return EPIStatus.AUSENTE, None
    
    def _calculate_overlap(self, box1: List[int], box2: List[int]) -> float:
        """Calcula sobreposição entre duas bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calcula interseção
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcula união
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Person]]:
        """Processa imagem completa e retorna resultado visualizado"""
        # Detecta objetos
        detections = self.detect_objects(image)
        
        # Valida posicionamento dos EPIs
        validated_people = self.validate_epi_placement(detections)
        
        # Desenha resultados
        result_image = self._draw_results(image, validated_people)
        
        return result_image, validated_people
    
    def _draw_results(self, image: np.ndarray, people: List[Person]) -> np.ndarray:
        """Desenha resultados na imagem"""
        result_image = image.copy()
        
        for person in people:
            px1, py1, px2, py2 = person.bbox
            
            # Desenha pessoa
            cv2.rectangle(result_image, (px1, py1), (px2, py2), self.colors['person'], 2)
            
            # Desenha capacete se detectado
            if person.helmet_detection:
                hx1, hy1, hx2, hy2 = person.helmet_detection.bbox
                color = self.colors['helmet_correct'] if person.helmet_status == EPIStatus.CORRETO else self.colors['helmet_violation']
                cv2.rectangle(result_image, (hx1, hy1), (hx2, hy2), color, 2)
                
                # Label do capacete
                label = f"CAPACETE: {'CORRETO' if person.helmet_status == EPIStatus.CORRETO else 'INCORRETO'}"
                cv2.putText(result_image, label, (hx1, hy1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Desenha colete se detectado
            if person.vest_detection:
                vx1, vy1, vx2, vy2 = person.vest_detection.bbox
                color = self.colors['vest_correct'] if person.vest_status == EPIStatus.CORRETO else self.colors['vest_violation']
                cv2.rectangle(result_image, (vx1, vy1), (vx2, vy2), color, 2)
                
                # Label do colete
                label = f"COLETE: {'CORRETO' if person.vest_status == EPIStatus.CORRETO else 'INCORRETO'}"
                cv2.putText(result_image, label, (vx1, vy1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Status geral da pessoa
            if person.helmet_status == EPIStatus.CORRETO and person.vest_status == EPIStatus.CORRETO:
                status = "COMPLIANT"
                color = (0, 255, 0)  # Verde
            elif person.helmet_status == EPIStatus.CORRETO or person.vest_status == EPIStatus.CORRETO:
                status = "PARCIAL"
                color = (0, 255, 255)  # Amarelo
            else:
                status = "VIOLATION"
                color = (0, 0, 255)  # Vermelho
            
            # Label da pessoa
            cv2.putText(result_image, f"PESSOA: {status}", (px1, py2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return result_image
    
    def get_compliance_summary(self, people: List[Person]) -> Dict[str, Any]:
        """Retorna resumo de compliance"""
        total_people = len(people)
        compliant = 0
        partial = 0
        violations = 0
        
        for person in people:
            if person.helmet_status == EPIStatus.CORRETO and person.vest_status == EPIStatus.CORRETO:
                compliant += 1
            elif person.helmet_status == EPIStatus.CORRETO or person.vest_status == EPIStatus.CORRETO:
                partial += 1
            else:
                violations += 1
        
        return {
            'total_people': total_people,
            'compliant': compliant,
            'partial': partial,
            'violations': violations,
            'compliance_rate': (compliant / total_people * 100) if total_people > 0 else 0
        }
