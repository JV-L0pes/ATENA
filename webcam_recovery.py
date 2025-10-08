#!/usr/bin/env python3
"""
ATHENA - Sistema de Recuperação de Webcam para Windows
====================================================
Sistema para resolver problemas de travamento da webcam no Windows
"""

import cv2
import time
import logging
import threading
from queue import Queue, Empty
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class WebcamRecoverySystem:
    """Sistema de recuperação automática da webcam"""
    
    def __init__(self, camera_source: int = 0):
        self.camera_source = camera_source
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=5)
        self.capture_thread = None
        self.recovery_count = 0
        self.max_recovery_attempts = 10
        self.last_frame_time = 0
        self.frame_timeout = 2.0  # 2 segundos sem frame = problema
        
        # Configurações otimizadas para Windows
        self.backend_preferences = [
            cv2.CAP_DSHOW,      # DirectShow (mais estável no Windows)
            cv2.CAP_MSMF,       # Media Foundation (fallback)
            cv2.CAP_ANY         # Qualquer backend disponível
        ]
        
        logger.info("🔄 Sistema de recuperação de webcam inicializado")
    
    def initialize_camera(self) -> bool:
        """Inicializa a câmera com diferentes backends"""
        for backend in self.backend_preferences:
            try:
                logger.info(f"🎥 Tentando backend: {backend}")
                self.cap = cv2.VideoCapture(self.camera_source, backend)
                
                if self.cap.isOpened():
                    # Configurar propriedades para estabilidade
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer pequeno
                    
                    # Testar captura
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logger.info(f"✅ Câmera inicializada com backend {backend}")
                        return True
                    else:
                        logger.warning(f"⚠️ Backend {backend} não consegue capturar frames")
                        self.cap.release()
                        self.cap = None
                else:
                    logger.warning(f"⚠️ Backend {backend} não consegue abrir câmera")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                        
            except Exception as e:
                logger.error(f"❌ Erro com backend {backend}: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        logger.error("❌ Nenhum backend funcionou")
        return False
    
    def start_capture(self) -> bool:
        """Inicia captura de frames"""
        if not self.initialize_camera():
            return False
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        logger.info("🚀 Captura de frames iniciada")
        return True
    
    def _capture_loop(self):
        """Loop principal de captura com recuperação automática"""
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    logger.warning("🔄 Câmera não está aberta, tentando recuperar...")
                    if not self._recover_camera():
                        time.sleep(1)
                        continue
                
                # Tentar capturar frame
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    self.last_frame_time = time.time()
                    
                    # Adicionar frame à fila (não bloquear)
                    try:
                        if not self.frame_queue.full():
                            self.frame_queue.put_nowait(frame.copy())
                        else:
                            # Remover frame antigo e adicionar novo
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(frame.copy())
                            except Empty:
                                pass
                    except:
                        pass
                    
                else:
                    consecutive_failures += 1
                    logger.warning(f"⚠️ Falha na captura ({consecutive_failures}/{max_consecutive_failures})")
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error("🔄 Muitas falhas consecutivas, tentando recuperar câmera...")
                        self._recover_camera()
                        consecutive_failures = 0
                
                # Verificar timeout de frames
                if time.time() - self.last_frame_time > self.frame_timeout:
                    logger.warning("⏰ Timeout de frames detectado")
                    self._recover_camera()
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de captura: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    self._recover_camera()
                    consecutive_failures = 0
                
                time.sleep(0.1)
    
    def _recover_camera(self) -> bool:
        """Tenta recuperar a câmera"""
        if self.recovery_count >= self.max_recovery_attempts:
            logger.error("❌ Máximo de tentativas de recuperação atingido")
            return False
        
        self.recovery_count += 1
        logger.info(f"🔄 Tentativa de recuperação {self.recovery_count}/{self.max_recovery_attempts}")
        
        # Fechar câmera atual
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        # Aguardar um pouco
        time.sleep(1)
        
        # Tentar reinicializar
        return self.initialize_camera()
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Obtém frame da fila"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_frame_non_blocking(self) -> Optional[np.ndarray]:
        """Obtém frame sem bloquear"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None
    
    def stop(self):
        """Para o sistema"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        
        logger.info("🛑 Sistema de captura parado")
    
    def get_status(self) -> dict:
        """Retorna status do sistema"""
        return {
            "is_running": self.is_running,
            "camera_opened": self.cap is not None and self.cap.isOpened() if self.cap else False,
            "queue_size": self.frame_queue.qsize(),
            "recovery_count": self.recovery_count,
            "last_frame_age": time.time() - self.last_frame_time if self.last_frame_time > 0 else None
        }

class StableWebcamCapture:
    """Classe simplificada para captura estável"""
    
    def __init__(self, camera_source: int = 0):
        self.recovery_system = WebcamRecoverySystem(camera_source)
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Inicializa o sistema"""
        if self.recovery_system.start_capture():
            self.is_initialized = True
            logger.info("✅ Sistema de webcam estável inicializado")
            return True
        else:
            logger.error("❌ Falha ao inicializar sistema de webcam")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Lê frame da câmera"""
        if not self.is_initialized:
            return False, None
        
        frame = self.recovery_system.get_frame(timeout=0.1)
        
        if frame is not None:
            return True, frame
        else:
            return False, None
    
    def isOpened(self) -> bool:
        """Verifica se a câmera está aberta"""
        if not self.is_initialized:
            return False
        
        status = self.recovery_system.get_status()
        return status["camera_opened"]
    
    def release(self):
        """Libera recursos"""
        self.recovery_system.stop()
        self.is_initialized = False
        logger.info("🔄 Recursos da webcam liberados")

# Função de teste
def test_webcam_recovery():
    """Testa o sistema de recuperação"""
    logger.info("🧪 Testando sistema de recuperação de webcam...")
    
    capture = StableWebcamCapture(0)
    
    if not capture.initialize():
        logger.error("❌ Falha ao inicializar")
        return False
    
    logger.info("✅ Sistema inicializado, testando captura...")
    
    for i in range(100):
        ret, frame = capture.read()
        
        if ret and frame is not None:
            logger.info(f"✅ Frame {i+1}: {frame.shape}")
        else:
            logger.warning(f"⚠️ Frame {i+1}: Falha na captura")
        
        time.sleep(0.1)
    
    capture.release()
    logger.info("✅ Teste concluído")
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_webcam_recovery()
