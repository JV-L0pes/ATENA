"""
Detector de EPIs - Classe principal para detecção e validação
Valida se capacete está na cabeça e colete no tronco da pessoa
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml
from typing import List, Tuple, Dict, Any
import logging

class EPIDetector:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa o detector de EPIs
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        # Carrega configurações
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Configura logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Carrega modelo YOLO
        self.model = YOLO(self.config['model']['weights'])
        self.device = self.config['model']['device']
        
        # Configurações de detecção
        self.conf_threshold = self.config['model']['conf_threshold']
        self.iou_threshold = self.config['model']['iou_threshold']
        
        # Configurações de validação
        self.head_region = self.config['validation']['head_region']
        self.torso_region = self.config['validation']['torso_region']
        self.overlap_threshold = self.config['validation']['overlap_threshold']
        
        # Cores para visualização
        self.colors = self.config['visualization']['colors']
        
        self.logger.info(f"Detector de EPIs inicializado com dispositivo: {self.device}")
    
    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detecta pessoas e EPIs na imagem
        
        Args:
            image: Imagem de entrada (BGR)
            
        Returns:
            Lista de detecções com bounding boxes e classes
        """
        # Executa detecção YOLO
        results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, device=self.device)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': cls,
                        'class_name': self._get_class_name(cls)
                    })
        
        return detections
    
    def validate_epi_positioning(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Valida o posicionamento dos EPIs em relação às pessoas
        
        Args:
            detections: Lista de detecções brutas
            
        Returns:
            Lista de detecções validadas com status de compliance
        """
        # Separa pessoas, EPIs e infrações (usando nomes do novo dataset)
        people = [d for d in detections if d['class_name'] == 'person']
        helmets = [d for d in detections if d['class_name'] == 'helmet']
        vests = [d for d in detections if d['class_name'] == 'vest']
        no_helmets = [d for d in detections if d['class_name'] == 'no-helmet']
        no_vests = [d for d in detections if d['class_name'] == 'no-vest']
        
        validated_detections = []
        
        # Para cada pessoa, valida EPIs com lógica híbrida
        for person in people:
            person_bbox = person['bbox']
            validated_detections.append({
                **person,
                'status': 'person',
                'violations': []
            })
            
            # LÓGICA HÍBRIDA: Validação de capacete
            helmet_status = self._validate_helmet_hybrid(person_bbox, helmets, no_helmets)
            validated_detections.append(helmet_status)
            
            # LÓGICA HÍBRIDA: Validação de colete  
            vest_status = self._validate_vest_hybrid(person_bbox, vests, no_vests)
            validated_detections.append(vest_status)
        
        return validated_detections
    
    def _validate_helmet_hybrid(self, person_bbox: List[int], helmets: List[Dict], no_helmets: List[Dict]) -> Dict[str, Any]:
        """
        Validação híbrida de capacete: IA + lógica Python
        
        Args:
            person_bbox: Bounding box da pessoa
            helmets: Lista de capacetes detectados
            no_helmets: Lista de infrações de capacete detectadas pela IA
            
        Returns:
            Status de validação do capacete
        """
        head_region = self._calculate_head_region(person_bbox)
        
        # Método 1: IA detectou infração diretamente
        for no_helmet in no_helmets:
            if self._calculate_overlap(head_region, no_helmet['bbox']) > self.overlap_threshold:
                return {
                    'class_name': 'no-helmet',
                    'bbox': no_helmet['bbox'],
                    'confidence': no_helmet['confidence'],
                    'status': 'violation',
                    'violation_type': 'missing_helmet'
                }
        
        # Método 2: Verifica se há capacete na região da cabeça
        for helmet in helmets:
            if self._calculate_overlap(head_region, helmet['bbox']) > self.overlap_threshold:
                return {
                    'class_name': 'helmet',
                    'bbox': helmet['bbox'],
                    'confidence': helmet['confidence'],
                    'status': 'compliant',
                    'violation_type': None
                }
        
        # Método 3: Lógica Python - se não há capacete na cabeça = infração
        return {
            'class_name': 'no-helmet',
            'bbox': head_region,
            'confidence': 0.8,  # Confiança da lógica Python
            'status': 'violation',
            'violation_type': 'missing_helmet'
        }
    
    def _validate_vest_hybrid(self, person_bbox: List[int], vests: List[Dict], no_vests: List[Dict]) -> Dict[str, Any]:
        """
        Validação híbrida de colete: IA + lógica Python
        
        Args:
            person_bbox: Bounding box da pessoa
            vests: Lista de coletes detectados
            no_vests: Lista de infrações de colete detectadas pela IA
            
        Returns:
            Status de validação do colete
        """
        torso_region = self._calculate_torso_region(person_bbox)
        
        # Método 1: IA detectou infração diretamente
        for no_vest in no_vests:
            if self._calculate_overlap(torso_region, no_vest['bbox']) > self.overlap_threshold:
                return {
                    'class_name': 'no-vest',
                    'bbox': no_vest['bbox'],
                    'confidence': no_vest['confidence'],
                    'status': 'violation',
                    'violation_type': 'missing_vest'
                }
        
        # Método 2: Verifica se há colete na região do torso
        for vest in vests:
            if self._calculate_overlap(torso_region, vest['bbox']) > self.overlap_threshold:
                return {
                    'class_name': 'vest',
                    'bbox': vest['bbox'],
                    'confidence': vest['confidence'],
                    'status': 'compliant',
                    'violation_type': None
                }
        
        # Método 3: Lógica Python - se não há colete no torso = infração
        return {
            'class_name': 'no-vest',
            'bbox': torso_region,
            'confidence': 0.8,  # Confiança da lógica Python
            'status': 'violation',
            'violation_type': 'missing_vest'
        }
    
    def _calculate_head_region(self, person_bbox: List[int]) -> List[int]:
        """Calcula região da cabeça baseada na pessoa"""
        x1, y1, x2, y2 = person_bbox
        person_height = y2 - y1
        head_height = int(person_height * self.head_region['height'])
        head_y1 = y1
        head_y2 = y1 + head_height
        return [x1, head_y1, x2, head_y2]
    
    def _calculate_torso_region(self, person_bbox: List[int]) -> List[int]:
        """Calcula região do torso baseada na pessoa"""
        x1, y1, x2, y2 = person_bbox
        person_height = y2 - y1
        torso_start = int(person_height * self.torso_region['start'])
        torso_end = int(person_height * self.torso_region['end'])
        torso_y1 = y1 + torso_start
        torso_y2 = y1 + torso_end
        return [x1, torso_y1, x2, torso_y2]

    def _validate_helmet_position(self, person_bbox: List[int], helmets: List[Dict]) -> Dict[str, Any]:
        """
        Valida se o capacete está na região da cabeça da pessoa
        """
        person_x1, person_y1, person_x2, person_y2 = person_bbox
        person_height = person_y2 - person_y1
        
        # Define região da cabeça
        head_y1 = person_y1 + int(person_height * self.head_region['top'])
        head_y2 = person_y1 + int(person_height * self.head_region['bottom'])
        
        for helmet in helmets:
            helmet_bbox = helmet['bbox']
            helmet_x1, helmet_y1, helmet_x2, helmet_y2 = helmet_bbox
            
            # Verifica sobreposição com região da cabeça
            overlap = self._calculate_overlap(
                [helmet_x1, helmet_y1, helmet_x2, helmet_y2],
                [person_x1, head_y1, person_x2, head_y2]
            )
            
            if overlap > self.overlap_threshold:
                return {
                    **helmet,
                    'status': 'helmet_correct',
                    'position': 'head',
                    'overlap': overlap
                }
        
        return None
    
    def _validate_vest_position(self, person_bbox: List[int], vests: List[Dict]) -> Dict[str, Any]:
        """
        Valida se o colete está na região do tronco da pessoa
        """
        person_x1, person_y1, person_x2, person_y2 = person_bbox
        person_height = person_y2 - person_y1
        
        # Define região do tronco
        torso_y1 = person_y1 + int(person_height * self.torso_region['top'])
        torso_y2 = person_y1 + int(person_height * self.torso_region['bottom'])
        
        for vest in vests:
            vest_bbox = vest['bbox']
            vest_x1, vest_y1, vest_x2, vest_y2 = vest_bbox
            
            # Verifica sobreposição com região do tronco
            overlap = self._calculate_overlap(
                [vest_x1, vest_y1, vest_x2, vest_y2],
                [person_x1, torso_y1, person_x2, torso_y2]
            )
            
            if overlap > self.overlap_threshold:
                return {
                    **vest,
                    'status': 'vest_correct',
                    'position': 'torso',
                    'overlap': overlap
                }
        
        return None
    
    def _check_violations(self, person_bbox: List[int], helmet_status: Dict, vest_status: Dict) -> List[Dict]:
        """
        Cria boxes de violação para EPIs ausentes ou mal posicionados
        """
        violations = []
        person_x1, person_y1, person_x2, person_y2 = person_bbox
        person_height = person_y2 - person_y1
        
        # Violação de capacete ausente
        if not helmet_status:
            head_y1 = person_y1 + int(person_height * self.head_region['top'])
            head_y2 = person_y1 + int(person_height * self.head_region['bottom'])
            
            violations.append({
                'bbox': [person_x1, head_y1, person_x2, head_y2],
                'status': 'helmet_violation',
                'class_name': 'helmet_missing',
                'confidence': 1.0,
                'violation_type': 'missing_helmet'
            })
        
        # Violação de colete ausente
        if not vest_status:
            torso_y1 = person_y1 + int(person_height * self.torso_region['top'])
            torso_y2 = person_y1 + int(person_height * self.torso_region['bottom'])
            
            violations.append({
                'bbox': [person_x1, torso_y1, person_x2, torso_y2],
                'status': 'vest_violation',
                'class_name': 'vest_missing',
                'confidence': 1.0,
                'violation_type': 'missing_vest'
            })
        
        return violations
    
    def _calculate_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calcula a sobreposição entre dois bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calcula interseção
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Retorna IoU (Intersection over Union)
        return intersection / (area1 + area2 - intersection)
    
    def _get_class_name(self, class_id: int) -> str:
        """
        Converte ID da classe para nome (baseado no dataset Roboflow)
        """
        class_names = {
            0: 'helmet',
            1: 'no-helmet',
            2: 'no-vest', 
            3: 'person',
            4: 'vest'
        }
        return class_names.get(class_id, 'unknown')
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Processa imagem completa: detecta, valida e retorna resultado
        
        Args:
            image: Imagem de entrada (BGR)
            
        Returns:
            Tupla com imagem processada e detecções validadas
        """
        # Detecta objetos
        detections = self.detect_objects(image)
        
        # Valida posicionamento dos EPIs
        validated_detections = self.validate_epi_positioning(detections)
        
        # Desenha resultados na imagem
        processed_image = self._draw_detections(image, validated_detections)
        
        return processed_image, validated_detections
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Desenha boxes e labels na imagem baseado no status
        """
        img_copy = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            status = detection.get('status', 'unknown')
            class_name = detection.get('class_name', 'unknown')
            
            # Define cor baseada no status e classe
            if status == 'person':
                color = self.colors['person']
            elif status == 'compliant':
                if 'helmet' in class_name:
                    color = self.colors['helmet_correct']
                elif 'vest' in class_name:
                    color = self.colors['vest_correct']
                else:
                    color = [0, 255, 0]  # Verde padrão
            elif status == 'violation':
                if 'helmet' in class_name:
                    color = self.colors['helmet_violation']
                elif 'vest' in class_name:
                    color = self.colors['vest_violation']
                else:
                    color = [0, 0, 255]  # Vermelho padrão
            else:
                color = [128, 128, 128]  # Cinza para desconhecido
            
            # Desenha box
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 
                         self.config['visualization']['line_thickness'])
            
            # Adiciona label
            label = f"{class_name}: {status}"
            font_scale = self.config['visualization']['font_scale']
            thickness = self.config['visualization']['line_thickness']
            
            # Calcula tamanho do texto
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                          font_scale, thickness)
            
            # Desenha background do texto
            cv2.rectangle(img_copy, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            
            # Desenha texto
            cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (255, 255, 255), thickness)
        
        return img_copy
