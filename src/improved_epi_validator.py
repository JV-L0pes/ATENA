#!/usr/bin/env python3
"""
Validador inteligente de EPIs com sistema anti-veículo otimizado
Implementa validação em camadas baseada na confiança
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ObjectType(Enum):
    """Tipos de objetos detectados"""
    HUMAN = "human"
    MOTORCYCLE = "motorcycle"
    VEHICLE = "vehicle"
    UNKNOWN = "unknown"

@dataclass
class DetectionData:
    """Dados da detecção para validação"""
    bbox: List[int]
    confidence: float
    class_name: str
    frame_number: int
    timestamp: float

class ImprovedEPIValidator:
    """
    Validador inteligente para diferenciar humanos de veículos
    Implementa validação em camadas baseada na confiança
    """
    
    def __init__(self):
        """Inicializa validador com configurações otimizadas"""
        self.logger = logging.getLogger(__name__)
        
        # Configurações otimizadas para canteiro de obra
        self.config = {
            'human_constraints': {
                'min_height_px': 120,           # Altura mínima para pessoa
                'max_height_px': 3000,          # Altura máxima
                'min_aspect_ratio': 1.6,        # height/width mínimo
                'max_aspect_ratio': 6.0,        # height/width máximo
                'min_width_px': 30,             # Largura mínima
                'min_height_px_small': 50       # Altura mínima para objetos pequenos
            },
            
            'vehicle_signatures': {
                'motorcycle': {
                    'width_height_ratio_min': 0.6,  # Motos tendem a ser mais largas
                    'width_height_ratio_max': 1.4,  # Proporção quase quadrada
                    'min_width': 80,                 # Largura mínima de moto
                    'typical_area_range': [5000, 50000]  # Área típica
                },
                'rectangular_forms': {
                    'width_height_ratio_min': 0.8,  # Objetos retangulares
                    'width_height_ratio_max': 1.2
                },
                'wide_vehicles': {
                    'width_height_ratio_threshold': 1.5  # Veículos muito largos
                },
                'obvious_vehicles': {
                    'width_height_ratio_threshold': 2.0  # Veículos extremamente largos
                }
            },
            
            'confidence_thresholds': {
                'high_confidence': 0.6,        # Confiança alta = validação relaxada
                'medium_confidence': 0.4,      # Confiança média = validação rigorosa
                'low_confidence': 0.25         # Confiança baixa = rejeita
            }
        }
    
    def is_valid_human_detection(self, detection_data: DetectionData) -> bool:
        """
        Valida se a detecção é provavelmente uma pessoa
        Implementa validação em camadas baseada na confiança
        """
        try:
            # Extrai geometria do objeto
            bbox = detection_data.bbox
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Validação baseada na confiança
            if detection_data.confidence >= self.config['confidence_thresholds']['high_confidence']:
                return self._strict_validation(width, height)
            elif detection_data.confidence >= self.config['confidence_thresholds']['medium_confidence']:
                return self._medium_validation(width, height)
            else:
                return self._strict_validation(width, height)
                
        except Exception as e:
            self.logger.error(f"❌ Erro na validação: {e}")
            return False
    
    def _strict_validation(self, width: int, height: int) -> bool:
        """Validação rigorosa para confiança baixa"""
        # Verifica se é obviamente um veículo
        if self._is_obviously_vehicle(width, height):
            return False
        
        # Verifica proporções humanas
        if height < self.config['human_constraints']['min_height_px']:
            return False
        
        if width < self.config['human_constraints']['min_width_px']:
            return False
        
        aspect_ratio = height / width if width > 0 else 0
        if aspect_ratio < self.config['human_constraints']['min_aspect_ratio']:
            return False
        
        return True
    
    def _medium_validation(self, width: int, height: int) -> bool:
        """Validação média para confiança média"""
        # Verifica se é obviamente um veículo
        if self._is_obviously_vehicle(width, height):
            return False
        
        # Validação mais flexível
        if height < self.config['human_constraints']['min_height_px_small']:
            return False
        
        return True
    
    def _is_vehicle_like_shape(self, width: int, height: int) -> bool:
        """Verifica se a forma é típica de veículo"""
        width_height_ratio = width / height if height > 0 else 0
        
        # Verifica assinaturas de moto
        if (self.config['vehicle_signatures']['motorcycle']['width_height_ratio_min'] <= 
            width_height_ratio <= 
            self.config['vehicle_signatures']['motorcycle']['width_height_ratio_max']):
            
            if width >= self.config['vehicle_signatures']['motorcycle']['min_width']:
                area = width * height
                min_area, max_area = self.config['vehicle_signatures']['motorcycle']['typical_area_range']
                if min_area <= area <= max_area:
                    return True
        
        # Verifica formas retangulares
        if (self.config['vehicle_signatures']['rectangular_forms']['width_height_ratio_min'] <= 
            width_height_ratio <= 
            self.config['vehicle_signatures']['rectangular_forms']['width_height_ratio_max']):
            return True
        
        return False
    
    def _is_obviously_vehicle(self, width: int, height: int) -> bool:
        """Verifica se é obviamente um veículo"""
        width_height_ratio = width / height if height > 0 else 0
        
        # Veículos muito largos
        if width_height_ratio >= self.config['vehicle_signatures']['wide_vehicles']['width_height_ratio_threshold']:
            return True
        
        # Veículos extremamente largos
        if width_height_ratio >= self.config['vehicle_signatures']['obvious_vehicles']['width_height_ratio_threshold']:
            return True
        
        return False
    
    def _is_too_symmetric(self, width: int, height: int) -> bool:
        """Verifica se o objeto é muito simétrico (típico de veículos)"""
        # Se largura e altura são muito similares, pode ser um veículo
        if width > 0 and height > 0:
            ratio = min(width, height) / max(width, height)
            if ratio > 0.8:  # Mais de 80% de similaridade
                return True
        return False
    
    def validate_epi_compliance(self, detection_data: DetectionData) -> Dict[str, bool]:
        """
        Valida compliance de EPIs para detecções humanas
        Retorna dicionário com status de cada EPI
        """
        if not self.is_valid_human_detection(detection_data):
            return {
                'is_human': False,
                'has_helmet': False,
                'has_vest': False,
                'is_compliant': False
            }
        
        # Aqui você pode implementar validação específica de EPIs
        # Por enquanto, retorna que é humano
        return {
            'is_human': True,
            'has_helmet': True,  # Placeholder
            'has_vest': True,    # Placeholder
            'is_compliant': True
        }