#!/usr/bin/env python3
"""
Sistema de Câmera para Canteiros de Obra
Configurações otimizadas para detecção de EPIs em ambiente de construção
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple
import logging

class ConstructionCameraSystem:
    """
    Sistema de câmera otimizado para canteiros de obra
    Configurações específicas para detecção de EPIs
    """
    
    def __init__(self, camera_id: int = 0):
        """Inicializa o sistema de câmera"""
        self.camera_id = camera_id
        self.cap = None
        self.logger = logging.getLogger(__name__)
        
        # Configurações padrão para canteiros de obra
        self.default_settings = {
            'resolution': (1280, 720),      # HD otimizado para detecção
            'fps': 30,                      # FPS para análise em tempo real
            'brightness': 60,               # Brilho aumentado para ambientes externos
            'contrast': 45,                 # Contraste reduzido para evitar sombras duras
            'exposure': 1,                  # Exposição ligeiramente positiva
            'gain': 0,                      # Ganho neutro
            'saturation': 55,               # Saturação aumentada para cores dos EPIs
            'autofocus': 0,                 # Foco fixo para estabilidade
            'auto_exposure': 0.75           # Exposição automática com controle
        }
        
        # Configurações específicas para diferentes condições
        self.environment_settings = {
            'sunny_outdoor': {
                'brightness': 40,
                'contrast': 55,
                'exposure': -1,
                'gain': -2
            },
            'cloudy_outdoor': {
                'brightness': 55,
                'contrast': 50,
                'exposure': 0,
                'gain': 0
            },
            'indoor': {
                'brightness': 70,
                'contrast': 40,
                'exposure': 2,
                'gain': 1
            },
            'low_light': {
                'brightness': 80,
                'contrast': 35,
                'exposure': 3,
                'gain': 3
            }
        }
        
        self.current_environment = 'cloudy_outdoor'  # Padrão
    
    def start_camera(self) -> bool:
        """Inicia a câmera com configurações otimizadas"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"Não foi possível abrir a câmera {self.camera_id}")
                return False
            
            # Aplica configurações padrão
            self._apply_camera_settings(self.default_settings)
            
            # Aplica configurações específicas do ambiente
            self._apply_camera_settings(self.environment_settings[self.current_environment])
            
            self.logger.info(f"Câmera {self.camera_id} iniciada com sucesso")
            self.logger.info(f"Resolução: {self.default_settings['resolution']}")
            self.logger.info(f"FPS: {self.default_settings['fps']}")
            self.logger.info(f"Ambiente: {self.current_environment}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar câmera: {e}")
            return False
    
    def _apply_camera_settings(self, settings: dict) -> None:
        """Aplica configurações de câmera"""
        try:
            # Configurações básicas
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['resolution'][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['resolution'][1])
            self.cap.set(cv2.CAP_PROP_FPS, settings['fps'])
            
            # Configurações de qualidade de imagem
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'])
            self.cap.set(cv2.CAP_PROP_CONTRAST, settings['contrast'])
            self.cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
            self.cap.set(cv2.CAP_PROP_GAIN, settings['gain'])
            self.cap.set(cv2.CAP_PROP_SATURATION, settings['saturation'])
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, settings['autofocus'])
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, settings['auto_exposure'])
            
        except Exception as e:
            self.logger.warning(f"Algumas configurações de câmera não puderam ser aplicadas: {e}")
    
    def set_environment(self, environment: str) -> bool:
        """Define o ambiente e aplica configurações específicas"""
        if environment not in self.environment_settings:
            self.logger.error(f"Ambiente não suportado: {environment}")
            return False
        
        self.current_environment = environment
        if self.cap and self.cap.isOpened():
            self._apply_camera_settings(self.environment_settings[environment])
            self.logger.info(f"Ambiente alterado para: {environment}")
        
        return True
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Captura um frame da câmera"""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Erro ao capturar frame")
            return None
        
        return frame
    
    def get_camera_info(self) -> dict:
        """Retorna informações da câmera"""
        if not self.cap or not self.cap.isOpened():
            return {}
        
        info = {}
        try:
            info['width'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info['height'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info['fps'] = self.cap.get(cv2.CAP_PROP_FPS)
            info['brightness'] = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            info['contrast'] = self.cap.get(cv2.CAP_PROP_CONTRAST)
            info['exposure'] = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            info['gain'] = self.cap.get(cv2.CAP_PROP_GAIN)
            info['saturation'] = self.cap.get(cv2.CAP_PROP_SATURATION)
        except Exception as e:
            self.logger.warning(f"Erro ao obter informações da câmera: {e}")
        
        return info
    
    def adjust_settings_interactive(self) -> None:
        """Ajusta configurações de forma interativa"""
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Câmera não está ativa")
            return
        
        print("\n🔧 AJUSTE INTERATIVO DE CONFIGURAÇÕES DE CÂMERA")
        print("=" * 60)
        print("Controles:")
        print("  B/N: Aumentar/Diminuir brilho")
        print("  C/V: Aumentar/Diminuir contraste")
        print("  E/D: Aumentar/Diminuir exposição")
        print("  G/F: Aumentar/Diminuir ganho")
        print("  S: Aumentar/Diminuir saturação")
        print("  A: Alternar ambiente")
        print("  Q: Sair")
        print("=" * 60)
        
        brightness = self.default_settings['brightness']
        contrast = self.default_settings['contrast']
        exposure = self.default_settings['exposure']
        gain = self.default_settings['gain']
        saturation = self.default_settings['saturation']
        
        while True:
            frame = self.get_frame()
            if frame is None:
                continue
            
            # Aplica ajustes em tempo real
            adjusted_frame = cv2.convertScaleAbs(
                frame, 
                alpha=contrast/50, 
                beta=brightness-50
            )
            
            # Adiciona informações na tela
            info_text = [
                f"B:{brightness} C:{contrast} E:{exposure}",
                f"G:{gain} S:{saturation}",
                f"Ambiente: {self.current_environment}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(adjusted_frame, text, (10, 30 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Ajuste de Configurações de Câmera', adjusted_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):  # Aumenta brilho
                brightness = min(100, brightness + 5)
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            elif key == ord('n'):  # Diminui brilho
                brightness = max(0, brightness - 5)
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            elif key == ord('c'):  # Aumenta contraste
                contrast = min(100, contrast + 5)
                self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            elif key == ord('v'):  # Diminui contraste
                contrast = max(0, contrast - 5)
                self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
            elif key == ord('e'):  # Aumenta exposição
                exposure = min(10, exposure + 1)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            elif key == ord('d'):  # Diminui exposição
                exposure = max(-10, exposure - 1)
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            elif key == ord('g'):  # Aumenta ganho
                gain = min(10, gain + 1)
                self.cap.set(cv2.CAP_PROP_GAIN, gain)
            elif key == ord('f'):  # Diminui ganho
                gain = max(-10, gain - 1)
                self.cap.set(cv2.CAP_PROP_GAIN, gain)
            elif key == ord('s'):  # Aumenta saturação
                saturation = min(100, saturation + 5)
                self.cap.set(cv2.CAP_PROP_SATURATION, saturation)
            elif key == ord('a'):  # Alterna ambiente
                environments = list(self.environment_settings.keys())
                current_idx = environments.index(self.current_environment)
                next_idx = (current_idx + 1) % len(environments)
                self.set_environment(environments[next_idx])
        
        cv2.destroyAllWindows()
        print("✅ Ajuste de configurações finalizado")
    
    def optimize_for_epi_detection(self) -> None:
        """Otimiza configurações especificamente para detecção de EPIs"""
        if not self.cap or not self.cap.isOpened():
            self.logger.error("Câmera não está ativa")
            return
        
        # Configurações otimizadas para EPIs
        epi_optimized = {
            'brightness': 65,      # Brilho aumentado para detectar capacetes escuros
            'contrast': 40,        # Contraste reduzido para evitar sombras
            'exposure': 2,         # Exposição positiva para ambientes externos
            'gain': 1,             # Ganho ligeiramente positivo
            'saturation': 60       # Saturação aumentada para cores dos EPIs
        }
        
        self._apply_camera_settings(epi_optimized)
        self.logger.info("Configurações otimizadas para detecção de EPIs aplicadas")
    
    def release(self) -> None:
        """Libera recursos da câmera"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Recursos da câmera liberados")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_camera()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()
