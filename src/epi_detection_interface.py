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
from datetime import datetime

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

@dataclass
class DetectionContext:
    confidence: float
    bbox: List[int]
    geometry: ObjectGeometry
    frame_number: int
    timestamp: float

class HumanVehicleClassifier:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
        self.config = {
            'human_constraints': {
                'min_height_px': 90,            # Altura mínima para pessoa (mais permissivo)
                'max_height_px': 1500,          # Altura máxima
                'min_aspect_ratio': 1.3,        # height/width mínimo (mais flexível)
                'max_aspect_ratio': 5.0,        # height/width máximo
                'min_width_px': 20,             # Largura mínima
                'max_width_px': 350             # Largura máxima
            },
            
            'vehicle_signatures': {
                'motorcycle': {
                    'width_height_ratio_min': 0.6,  # Motos tendem a ser mais largas
                    'width_height_ratio_max': 1.4,  # Proporção quase quadrada
                    'min_width': 70,                 # Largura mínima de moto
                    'min_area': 4000,               # Área mínima
                    'max_area': 50000               # Área máxima típica
                },
                'wide_vehicle': {
                    'width_height_ratio_min': 1.8,  # Veículos muito largos
                    'min_width': 150
                },
                'rectangular_object': {
                    'width_height_ratio_min': 0.7,  # Objetos retangulares
                    'width_height_ratio_max': 1.3,
                    'min_area': 8000                 # Área mínima
                }
            },
            
            'confidence_thresholds': {
                'high_confidence': 0.6,      # Confiança alta - validação relaxada
                'medium_confidence': 0.30,   # Confiança média - validação rigorosa
                'low_confidence': 0.12       # Confiança baixa - validação extrema
            }
        }
    
    def classify_object(self, detection_data: Dict) -> ObjectType:
        """Método principal para classificar objeto"""
        context = self._create_detection_context(detection_data)
        confidence = context.confidence
        
        # Estratégia baseada na confiança
        if confidence >= self.config['confidence_thresholds']['high_confidence']:
            return self._classify_high_confidence(context)
        elif confidence >= self.config['confidence_thresholds']['medium_confidence']:
            return self._classify_medium_confidence(context)
        else:
            return self._classify_low_confidence(context)
    
    def _create_detection_context(self, detection_data: Dict) -> DetectionContext:
        """Cria contexto completo da detecção"""
        bbox = detection_data['bbox']
        confidence = detection_data['confidence']
        geometry = ObjectGeometry.from_bbox(bbox)
        
        return DetectionContext(
            confidence=confidence,
            bbox=bbox,
            geometry=geometry,
            frame_number=detection_data.get('frame_number', 0),
            timestamp=detection_data.get('timestamp', time.time())
        )
    
    def _classify_high_confidence(self, context: DetectionContext) -> ObjectType:
        """Classificação para alta confiança - validação básica"""
        if self._has_extreme_proportions(context.geometry):
            return ObjectType.VEHICLE
        
        if self._matches_motorcycle_signature(context.geometry):
            return ObjectType.MOTORCYCLE
        
        return ObjectType.HUMAN
    
    def _classify_medium_confidence(self, context: DetectionContext) -> ObjectType:
        """Classificação para confiança média - validação rigorosa"""
        geometry = context.geometry
        
        # Validações dimensionais rigorosas
        if not self._meets_human_size_constraints(geometry):
            return ObjectType.VEHICLE
        
        if not self._meets_human_proportions(geometry):
            return ObjectType.VEHICLE
        
        # Detecção específica de veículos
        vehicle_type = self._detect_vehicle_signature(geometry)
        if vehicle_type != ObjectType.UNKNOWN:
            return vehicle_type
        
        return ObjectType.HUMAN
    
    def _classify_low_confidence(self, context: DetectionContext) -> ObjectType:
        """Classificação para baixa confiança - validação extrema"""
        geometry = context.geometry
        
        if not self._meets_strict_human_constraints(geometry):
            return ObjectType.VEHICLE
        
        if self._shows_vehicle_characteristics(geometry):
            return ObjectType.VEHICLE
        
        return ObjectType.HUMAN
    
    def _meets_human_size_constraints(self, geometry: ObjectGeometry) -> bool:
        """Valida dimensões compatíveis com pessoa"""
        constraints = self.config['human_constraints']
        
        return (constraints['min_height_px'] <= geometry.height <= constraints['max_height_px'] and
                constraints['min_width_px'] <= geometry.width <= constraints['max_width_px'])
    
    def _meets_human_proportions(self, geometry: ObjectGeometry) -> bool:
        """Valida proporções compatíveis com pessoa"""
        constraints = self.config['human_constraints']
        
        return (constraints['min_aspect_ratio'] <= geometry.aspect_ratio <= constraints['max_aspect_ratio'])
    
    def _meets_strict_human_constraints(self, geometry: ObjectGeometry) -> bool:
        """Validações extremamente rigorosas"""
        if not self._meets_human_size_constraints(geometry):
            return False
        
        if not self._meets_human_proportions(geometry):
            return False
        
        # Validações adicionais para baixa confiança
        if geometry.height < 120:  # Mais rigoroso
            return False
        
        if geometry.aspect_ratio < 1.6:  # Mais rigoroso
            return False
        
        return True
    
    def _detect_vehicle_signature(self, geometry: ObjectGeometry) -> ObjectType:
        """Detecta assinaturas específicas de veículos"""
        if self._matches_motorcycle_signature(geometry):
            return ObjectType.MOTORCYCLE
        
        if self._matches_wide_vehicle_signature(geometry):
            return ObjectType.VEHICLE
        
        if self._matches_rectangular_signature(geometry):
            return ObjectType.VEHICLE
        
        return ObjectType.UNKNOWN
    
    def _matches_motorcycle_signature(self, geometry: ObjectGeometry) -> bool:
        """Detecta assinatura de motocicleta - REGRA ESPECÍFICA PARA SUA MOTO"""
        moto_config = self.config['vehicle_signatures']['motorcycle']
        ratio = geometry.width_height_ratio
        
        # Critérios específicos para motos
        moto_indicators = 0
        
        # Indicador 1: Proporção típica de moto (quase quadrada)
        if moto_config['width_height_ratio_min'] <= ratio <= moto_config['width_height_ratio_max']:
            moto_indicators += 1
        
        # Indicador 2: Largura mínima característica
        if geometry.width >= moto_config['min_width']:
            moto_indicators += 1
        
        # Indicador 3: Área dentro do range típico
        if moto_config['min_area'] <= geometry.area <= moto_config['max_area']:
            moto_indicators += 1
        
        # Indicador 4: Proporção suspeita (entre 0.7 e 1.3 é muito suspeito)
        if 0.7 <= ratio <= 1.3:
            moto_indicators += 1
        
        # Se 3+ indicadores, é moto
        is_motorcycle = moto_indicators >= 3
        
        if is_motorcycle:
            self.logger.info(f"🏍️ MOTO DETECTADA: ratio={ratio:.2f}, area={geometry.area}, indicators={moto_indicators}")
        
        return is_motorcycle
    
    def _matches_wide_vehicle_signature(self, geometry: ObjectGeometry) -> bool:
        """Detecta veículos largos"""
        vehicle_config = self.config['vehicle_signatures']['wide_vehicle']
        
        return (geometry.width_height_ratio >= vehicle_config['width_height_ratio_min'] and
                geometry.width >= vehicle_config['min_width'])
    
    def _matches_rectangular_signature(self, geometry: ObjectGeometry) -> bool:
        """Detecta objetos retangulares típicos de veículos"""
        rect_config = self.config['vehicle_signatures']['rectangular_object']
        ratio = geometry.width_height_ratio
        
        return (rect_config['width_height_ratio_min'] <= ratio <= rect_config['width_height_ratio_max'] and
                geometry.area >= rect_config['min_area'])
    
    def _has_extreme_proportions(self, geometry: ObjectGeometry) -> bool:
        """Detecta proporções extremas"""
        return (geometry.width_height_ratio > 2.5 or 
                geometry.aspect_ratio > 8.0 or 
                geometry.height < 80)
    
    def _shows_vehicle_characteristics(self, geometry: ObjectGeometry) -> bool:
        """Análise avançada de características de veículos"""
        suspicious_factors = 0
        
        # Múltiplos fatores suspeitos
        if 0.6 <= geometry.width_height_ratio <= 1.6:
            suspicious_factors += 1
        
        if geometry.area > 35000:
            suspicious_factors += 1
        
        if geometry.width > 180:
            suspicious_factors += 1
        
        if geometry.height < geometry.width * 1.4:  # Muito largo para a altura
            suspicious_factors += 1
        
        return suspicious_factors >= 2

# ============================================================================
# SISTEMA DE SNAPSHOT AUTOMÁTICO INTEGRADO
# ============================================================================

class EPISnapshotManager:
    """Gerenciador de snapshot integrado na interface gráfica"""
    
    def __init__(self, interface):
        """Inicializa gerenciador de snapshot"""
        self.interface = interface
        self.logger = logging.getLogger(__name__)
        
        # Estado do sistema
        self.is_monitoring = False
        self.violation_timers = {}  # {person_id: {'start_time': timestamp, 'epi_type': 'helmet/vest'}}
        self.patience_period = 3.0  # segundos configurável
        
        # Diretório de snapshots
        self.snapshot_dir = Path("snapshots")
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # Histórico de snapshots
        self.snapshot_history = []
        
        # Thread para processamento de snapshots
        self.snapshot_thread = None
        self.stop_event = threading.Event()
        
        # Lock para proteger acesso ao dicionário de violações
        self.violation_lock = threading.Lock()
        
        # Configurações
        self.config = {
            'patience_period': 3.0,
            'snapshot_dir': 'snapshots',
            'image_quality': 95,
            'include_timestamp': True,
            'include_violation_details': True
        }
        
        self.logger.info("📸 Sistema de Snapshot integrado na interface")
    
    def start_monitoring(self) -> bool:
        """Inicia o monitoramento de violações"""
        if self.is_monitoring:
            return True
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Inicia thread de processamento
        self.snapshot_thread = threading.Thread(target=self._snapshot_processor)
        self.snapshot_thread.daemon = True
        self.snapshot_thread.start()
        
        self.logger.info("🔄 Monitoramento de violações iniciado")
        return True
    
    def stop_monitoring(self) -> None:
        """Para o monitoramento"""
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.snapshot_thread and self.snapshot_thread.is_alive():
            self.snapshot_thread.join(timeout=5)
        
        self.logger.info("⏹️ Monitoramento de violações parado")
    
    def process_violations(self, people: List, frame: np.ndarray) -> None:
        """Processa violações detectadas e inicia timers"""
        if not self.is_monitoring:
            return
        
        current_time = time.time()
        
        for person in people:
            person_id = id(person)
            
            # Verifica violações de capacete
            if person.get('helmet_status') == "AUSENTE":
                self._handle_violation(person_id, 'helmet', current_time, frame)
            else:
                self._clear_violation(person_id, 'helmet')
            
            # Verifica violações de colete
            if person.get('vest_status') == "AUSENTE":
                self._handle_violation(person_id, 'vest', current_time, frame)
            else:
                self._clear_violation(person_id, 'vest')
    
    def _handle_violation(self, person_id: int, epi_type: str, current_time: float, frame: np.ndarray) -> None:
        """Gerencia uma violação de EPI"""
        violation_key = f"{person_id}_{epi_type}"
        
        with self.violation_lock:
            if violation_key not in self.violation_timers:
                # Nova violação - inicia timer
                self.violation_timers[violation_key] = {
                    'start_time': current_time,
                    'epi_type': epi_type,
                    'person_id': person_id,
                    'frame': frame.copy()
                }
                self.logger.info(f"🚨 Nova violação de {epi_type} detectada para pessoa {person_id}")
            else:
                # Verifica se já passou do tempo de paciência
                elapsed_time = current_time - self.violation_timers[violation_key]['start_time']
                
                if elapsed_time >= self.patience_period:
                    # Tempo de paciência esgotado - tira snapshot
                    self._take_snapshot(violation_key, frame)
                    # Remove do timer após snapshot
                    if violation_key in self.violation_timers:
                        del self.violation_timers[violation_key]
    
    def _clear_violation(self, person_id: int, epi_type: str) -> None:
        """Limpa uma violação quando EPI é recolocado"""
        violation_key = f"{person_id}_{epi_type}"
        
        with self.violation_lock:
            if violation_key in self.violation_timers:
                try:
                    elapsed_time = time.time() - self.violation_timers[violation_key]['start_time']
                    self.logger.info(f"✅ EPI {epi_type} recolocado para pessoa {person_id} após {elapsed_time:.1f}s")
                    del self.violation_timers[violation_key]
                except KeyError:
                    # Violação já foi removida por outro processo
                    pass
                except Exception as e:
                    self.logger.error(f"Erro ao limpar violação {violation_key}: {e}")
    
    def _take_snapshot(self, violation_key: str, frame: np.ndarray) -> None:
        """Tira snapshot da violação"""
        try:
            # Gera nome do arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            person_id, epi_type = violation_key.split('_')
            
            filename = f"epi_violation_{epi_type}_{person_id}_{timestamp}.jpg"
            filepath = self.snapshot_dir / filename
            
            # Adiciona informações na imagem
            annotated_frame = self._annotate_violation_image(frame, epi_type, person_id)
            
            # Salva imagem
            cv2.imwrite(str(filepath), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, self.config['image_quality']])
            
            # Registra no histórico
            snapshot_info = {
                'timestamp': datetime.now().isoformat(),
                'filename': filename,
                'filepath': str(filepath),
                'epi_type': epi_type,
                'person_id': person_id,
                'violation_duration': self.patience_period
            }
            
            self.snapshot_history.append(snapshot_info)
            
            # Atualiza interface
            self.interface.update_snapshot_status()
            
            self.logger.info(f"📸 Snapshot salvo: {filename}")
            
        except Exception as e:
            self.logger.error(f"Erro ao tirar snapshot: {e}")
    
    def _annotate_violation_image(self, frame: np.ndarray, epi_type: str, person_id: int) -> np.ndarray:
        """Adiciona anotações na imagem de violação"""
        annotated = frame.copy()
        
        # Adiciona texto de violação
        text = f"VIOLACAO DE EPI: {epi_type.upper()}"
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Configurações de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 0, 255)  # Vermelho
        
        # Adiciona texto principal
        cv2.putText(annotated, text, (50, 50), font, font_scale, color, thickness)
        
        # Adiciona timestamp
        cv2.putText(annotated, timestamp, (50, 100), font, 0.7, (255, 255, 255), 2)
        
        # Adiciona borda vermelha
        cv2.rectangle(annotated, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, 5)
        
        return annotated
    
    def _snapshot_processor(self) -> None:
        """Thread para processamento de snapshots em background"""
        while not self.stop_event.is_set():
            try:
                # Processa snapshots pendentes
                current_time = time.time()
                
                # Verifica timers expirados
                expired_violations = []
                with self.violation_lock:
                    for violation_key, timer_info in self.violation_timers.items():
                        elapsed_time = current_time - timer_info['start_time']
                        if elapsed_time >= self.patience_period:
                            expired_violations.append(violation_key)
                
                # Processa violações expiradas
                for violation_key in expired_violations:
                    with self.violation_lock:
                        if violation_key in self.violation_timers:
                            try:
                                frame = self.violation_timers[violation_key]['frame']
                                self._take_snapshot(violation_key, frame)
                                # Remove do timer após snapshot
                                if violation_key in self.violation_timers:
                                    del self.violation_timers[violation_key]
                            except KeyError:
                                # Violação já foi removida por outro processo
                                continue
                            except Exception as e:
                                self.logger.error(f"Erro ao processar violação {violation_key}: {e}")
                
                time.sleep(0.1)  # Verifica a cada 100ms
                
            except Exception as e:
                self.logger.error(f"Erro no processador de snapshots: {e}")
                time.sleep(1)
    
    def get_status(self) -> Dict:
        """Retorna status atual do sistema"""
        current_time = time.time()
        active_violations = []
        
        with self.violation_lock:
            for violation_key, timer_info in self.violation_timers.items():
                elapsed_time = current_time - timer_info['start_time']
                remaining_time = max(0, self.patience_period - elapsed_time)
                
                active_violations.append({
                    'person_id': timer_info['person_id'],
                    'epi_type': timer_info['epi_type'],
                    'elapsed_time': elapsed_time,
                    'remaining_time': remaining_time
                })
        
        return {
            'is_monitoring': self.is_monitoring,
            'active_violations': active_violations,
            'patience_period': self.patience_period,
            'total_snapshots': len(self.snapshot_history),
            'snapshot_directory': str(self.snapshot_dir)
        }
    
    def get_snapshot_history(self, limit: int = 50) -> List[Dict]:
        """Retorna histórico de snapshots"""
        return self.snapshot_history[-limit:] if self.snapshot_history else []
    
    def set_patience_period(self, period: float) -> None:
        """Define o período de paciência"""
        self.patience_period = max(1.0, min(60.0, period))
        self.logger.info(f"⏰ Período de paciência alterado para {self.patience_period}s")

# ============================================================================
# INTERFACE PRINCIPAL DE DETECÇÃO DE EPIs
# ============================================================================

class EPIDetectionInterface:
    """Interface completa para detecção de EPIs com sistema anti-veículo e fullscreen"""
    
    def __init__(self):
        """Inicializa interface"""
        # Configura logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Configura OpenCV para evitar problemas de thread (resolve erro libavcodec)
        self._configure_opencv_threading()
        
        # Inicializa componentes
        self.model = None
        self.cap = None
        self.is_camera_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Sistema anti-veículo integrado
        self.human_validator = HumanVehicleClassifier()
        
        # Sistema de snapshot integrado
        self.snapshot_manager = EPISnapshotManager(self)
        
        # Classes do modelo
        self.classes = ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']
        
        # Cores para visualização
        self.colors = {
            'person': (255, 255, 0),      # Amarelo
            'helmet': (0, 255, 0),        # Verde
            'vest': (0, 255, 0),          # Verde
            'no-helmet': (0, 0, 255),     # Vermelho
            'no-vest': (0, 0, 255)        # Vermelho
        }
        
        # Sistema de tracking temporal otimizado
        self.person_tracks = {}
        self.next_track_id = 0
        
        # Controle de fullscreen
        self.is_fullscreen = False
        self.fullscreen_window = None
        
        # Lock para sincronizar acesso ao decoder de vídeo (resolve erro libavcodec)
        self.video_lock = threading.Lock()
        
        # Carrega modelo
        self.load_model()
        
        # Cria interface
        self.create_interface()
        
    def load_model(self):
        """Carrega modelo YOLOv5 otimizado"""
        try:
            # Tenta modelo superior primeiro
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
                self.logger.error("❌ Nenhum modelo encontrado")
                return False
            
            self.logger.info(f"🚀 Carregando modelo: {weights_path}")
            
            # Usa torch.hub diretamente
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
            
            # Configurações otimizadas para canteiro de obra
            self.model.conf = 0.15  # Baixo para capturar pessoas distantes
            self.model.iou = 0.33   # Leve aumento para reduzir supressões indevidas
            
            self.logger.info("✅ Modelo carregado com sucesso!")
            self.logger.info(f"⚙️ Configurações: conf={self.model.conf}, iou={self.model.iou}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def create_interface(self):
        """Cria interface gráfica otimizada"""
        self.root = tk.Tk()
        self.root.title("EPI Detection Interface - Sistema Anti-Veículo + Fullscreen")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Bind para ESC sair do fullscreen
        self.root.bind('<Escape>', self.exit_fullscreen)
        self.root.bind('<F11>', self.toggle_fullscreen)
        # Bind para barra de espaço (apenas quando não estiver em campos de texto)
        self.root.bind('<KeyPress-space>', self.toggle_video_pause)
        
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(main_frame, text="🛡️ DETECÇÃO DE EPIs - SISTEMA ANTI-VEÍCULO + FULLSCREEN", 
                               font=('Arial', 20, 'bold'), fg='white', bg='#2b2b2b')
        title_label.pack(pady=(0, 10))
        
        # Subtítulo
        subtitle_label = tk.Label(main_frame, text="Diferenciação Inteligente: Pessoas vs Motos/Veículos | F11: Fullscreen | ESC: Sair", 
                                 font=('Arial', 12), fg='#4CAF50', bg='#2b2b2b')
        subtitle_label.pack(pady=(0, 15))
        
        # Frame de controles
        controls_frame = tk.Frame(main_frame, bg='#3b3b3b', relief='raised', bd=2)
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Botões de controle
        btn_frame = tk.Frame(controls_frame, bg='#3b3b3b')
        btn_frame.pack(pady=10)
        
        # Botão Imagem
        self.img_btn = tk.Button(btn_frame, text="📸 DETECTAR IMAGEM", 
                                command=self.detect_image, 
                                font=('Arial', 11, 'bold'),
                                bg='#4CAF50', fg='white', 
                                relief='raised', bd=3, padx=15, pady=8)
        self.img_btn.pack(side=tk.LEFT, padx=8)
        
        # Botão Vídeo
        self.video_btn = tk.Button(btn_frame, text="🎥 DETECTAR VÍDEO", 
                                  command=self.detect_video, 
                                  font=('Arial', 11, 'bold'),
                                  bg='#2196F3', fg='white', 
                                  relief='raised', bd=3, padx=15, pady=8)
        self.video_btn.pack(side=tk.LEFT, padx=8)
        
        # Botão Câmera
        self.camera_btn = tk.Button(btn_frame, text="📹 CÂMERA AO VIVO", 
                                   command=self.toggle_camera, 
                                   font=('Arial', 11, 'bold'),
                                   bg='#FF9800', fg='white', 
                                   relief='raised', bd=3, padx=15, pady=8)
        self.camera_btn.pack(side=tk.LEFT, padx=8)
        
        # Botão Fullscreen
        self.fullscreen_btn = tk.Button(btn_frame, text="🖥️ FULLSCREEN", 
                                       command=self.toggle_fullscreen, 
                                       font=('Arial', 11, 'bold'),
                                       bg='#9C27B0', fg='white', 
                                       relief='raised', bd=3, padx=15, pady=8)
        self.fullscreen_btn.pack(side=tk.LEFT, padx=8)
        
        # Botão Parar
        self.stop_btn = tk.Button(btn_frame, text="⏹️ PARAR", 
                                 command=self.stop_camera, 
                                 font=('Arial', 11, 'bold'),
                                 bg='#f44336', fg='white', 
                                 relief='raised', bd=3, padx=15, pady=8)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Frame para controles do sistema de snapshot
        snapshot_frame = tk.Frame(controls_frame, bg='#3b3b3b')
        snapshot_frame.pack(pady=(5, 10))
        
        # Título do sistema de snapshot
        snapshot_title = tk.Label(snapshot_frame, text="📸 SISTEMA DE SNAPSHOT AUTOMÁTICO", 
                                 font=('Arial', 12, 'bold'), fg='#FFD700', bg='#3b3b3b')
        snapshot_title.pack(pady=(0, 5))
        
        # Botões do sistema de snapshot
        snapshot_btn_frame = tk.Frame(snapshot_frame, bg='#3b3b3b')
        snapshot_btn_frame.pack()
        
        # Botão Iniciar/Parar Snapshot
        self.snapshot_toggle_btn = tk.Button(snapshot_btn_frame, text="🔄 INICIAR SNAPSHOT", 
                                            command=self.toggle_snapshot_system, 
                                            font=('Arial', 10, 'bold'),
                                            bg='#4CAF50', fg='white', 
                                            relief='raised', bd=2, padx=12, pady=6)
        self.snapshot_toggle_btn.pack(side=tk.LEFT, padx=5)
        
        # Botão Status do Snapshot
        self.snapshot_status_btn = tk.Button(snapshot_btn_frame, text="📊 STATUS", 
                                            command=self.show_snapshot_status, 
                                            font=('Arial', 10, 'bold'),
                                            bg='#2196F3', fg='white', 
                                            relief='raised', bd=2, padx=12, pady=6)
        self.snapshot_status_btn.pack(side=tk.LEFT, padx=5)
        
        # Botão Histórico de Snapshots
        self.snapshot_history_btn = tk.Button(snapshot_btn_frame, text="📚 HISTÓRICO", 
                                             command=self.show_snapshot_history, 
                                             font=('Arial', 10, 'bold'),
                                             bg='#FF9800', fg='white', 
                                             relief='raised', bd=2, padx=12, pady=6)
        self.snapshot_history_btn.pack(side=tk.LEFT, padx=5)
        
        # Botão Configurar Tempo de Paciência
        self.snapshot_config_btn = tk.Button(snapshot_btn_frame, text="⏰ CONFIGURAR", 
                                            command=self.configure_snapshot_patience, 
                                            font=('Arial', 10, 'bold'),
                                            bg='#9C27B0', fg='white', 
                                            relief='raised', bd=2, padx=12, pady=6)
        self.snapshot_config_btn.pack(side=tk.LEFT, padx=5)
        
        # Label de status do snapshot
        self.snapshot_status_label = tk.Label(snapshot_frame, text="📸 Sistema de Snapshot: INATIVO", 
                                             font=('Arial', 10), fg='#FF6B6B', bg='#3b3b3b')
        self.snapshot_status_label.pack(pady=(5, 0))
        
        # Frame de visualização
        view_frame = tk.Frame(main_frame, bg='#1b1b1b', relief='sunken', bd=2)
        view_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas para imagem/vídeo
        self.canvas = tk.Canvas(view_frame, bg='#1b1b1b', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de informações
        info_frame = tk.Frame(main_frame, bg='#3b3b3b', relief='raised', bd=2)
        info_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Labels de informação
        self.status_label = tk.Label(info_frame, text="Status: Sistema anti-veículo ativo", 
                                    font=('Arial', 11), fg='white', bg='#3b3b3b')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        self.fps_label = tk.Label(info_frame, text="FPS: 0", 
                                 font=('Arial', 11), fg='white', bg='#3b3b3b')
        self.fps_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        self.detection_label = tk.Label(info_frame, text="Humanos: 0 | Rejeitados: 0", 
                                       font=('Arial', 11), fg='white', bg='#3b3b3b')
        self.detection_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        # Label de snapshots tirados
        self.snapshots_label = tk.Label(info_frame, text="📸 Snapshots: 0", 
                                       font=('Arial', 11), fg='#FFD700', bg='#3b3b3b')
        self.snapshots_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        # Label de violações ativas
        self.violations_label = tk.Label(info_frame, text="🚨 Violações Ativas: 0", 
                                        font=('Arial', 11), fg='#FF6B6B', bg='#3b3b3b')
        self.violations_label.pack(side=tk.LEFT, padx=10, pady=8)
        
        # Configurações
        config_frame = tk.Frame(main_frame, bg='#3b3b3b', relief='raised', bd=2)
        config_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Thresholds
        tk.Label(config_frame, text="Confidence:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(10, 5), pady=8)
        
        self.conf_var = tk.DoubleVar(value=0.15)
        self.conf_scale = tk.Scale(config_frame, from_=0.05, to=0.8, resolution=0.05,
                                  orient=tk.HORIZONTAL, variable=self.conf_var,
                                  bg='#3b3b3b', fg='white', highlightthickness=0)
        self.conf_scale.pack(side=tk.LEFT, padx=(0, 15), pady=8)
        
        tk.Label(config_frame, text="IoU:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(0, 5), pady=8)
        
        self.iou_var = tk.DoubleVar(value=0.25)
        self.iou_scale = tk.Scale(config_frame, from_=0.1, to=0.8, resolution=0.05,
                                 orient=tk.HORIZONTAL, variable=self.iou_var,
                                 bg='#3b3b3b', fg='white', highlightthickness=0)
        self.iou_scale.pack(side=tk.LEFT, padx=(0, 15), pady=8)
        
        # Configuração do tempo de paciência do snapshot
        tk.Label(config_frame, text="⏰ Paciência (s):", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(0, 5), pady=8)
        
        self.patience_var = tk.DoubleVar(value=3.0)
        self.patience_scale = tk.Scale(config_frame, from_=1.0, to=10.0, resolution=0.5,
                                      orient=tk.HORIZONTAL, variable=self.patience_var,
                                      bg='#3b3b3b', fg='white', highlightthickness=0)
        self.patience_scale.pack(side=tk.LEFT, padx=(0, 15), pady=8)
        
        # Botão aplicar configurações
        self.apply_btn = tk.Button(config_frame, text="✅ APLICAR", 
                                  command=self.apply_config, 
                                  font=('Arial', 10, 'bold'),
                                  bg='#4CAF50', fg='white', 
                                  relief='raised', bd=2, padx=12, pady=5)
        self.apply_btn.pack(side=tk.LEFT, padx=(0, 10), pady=8)
        
        # Label de ajuda
        help_label = tk.Label(config_frame, text="F11: Fullscreen | ESC: Sair Fullscreen", 
                             font=('Arial', 9), fg='#FFB74D', bg='#3b3b3b')
        help_label.pack(side=tk.RIGHT, padx=10, pady=8)
        
        # Bind eventos
        self.conf_scale.bind("<ButtonRelease-1>", self.on_scale_change)
        self.iou_scale.bind("<ButtonRelease-1>", self.on_scale_change)
        self.patience_scale.bind("<ButtonRelease-1>", self.on_patience_change)
        
        # Inicializa variáveis
        self.current_image = None
        self.current_detections = []
        self.frame_count = 0
        self.start_time = time.time()
        self.rejected_objects = 0
        
        # Variáveis de vídeo
        self.is_video_playing = False
        self.video_paused = False
        self.current_frame = 0
        
        # Frame atual para fullscreen
        self.current_display_frame = None
        
        # Atualiza interface
        self.update_interface()
    
    def _configure_opencv_threading(self):
        """Configura OpenCV para evitar problemas de thread (resolve erro libavcodec)"""
        try:
            # Configura número de threads para OpenCV
            cv2.setNumThreads(1)  # Força uso de apenas 1 thread
            
            # Configura flags para evitar problemas de thread
            cv2.setUseOptimized(True)
            
            self.logger.info("✅ OpenCV configurado para threading seguro")
        except Exception as e:
            self.logger.warning(f"⚠️ Não foi possível configurar OpenCV threading: {e}")
    
    def toggle_fullscreen(self, event=None):
        """Alterna modo fullscreen"""
        if not self.is_fullscreen:
            self.enter_fullscreen()
        else:
            self.exit_fullscreen()
    
    def enter_fullscreen(self):
        """Entra no modo fullscreen apenas para o vídeo"""
        if self.is_fullscreen:
            return
        
        self.is_fullscreen = True
        
        # Cria janela fullscreen APENAS para o vídeo
        self.fullscreen_window = tk.Toplevel(self.root)
        self.fullscreen_window.title("EPI Detection - Vídeo Fullscreen")
        self.fullscreen_window.configure(bg='black')
        self.fullscreen_window.attributes('-fullscreen', True)
        self.fullscreen_window.attributes('-topmost', True)
        
        # Bind teclas para fullscreen
        self.fullscreen_window.bind('<Escape>', self.exit_fullscreen)
        self.fullscreen_window.bind('<F11>', self.exit_fullscreen)
        self.fullscreen_window.bind('<space>', self.toggle_video_pause_fullscreen)
        self.fullscreen_window.focus_set()
        
        # Canvas fullscreen APENAS para vídeo
        self.fullscreen_canvas = tk.Canvas(self.fullscreen_window, bg='black', highlightthickness=0)
        self.fullscreen_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Atualiza botão
        self.fullscreen_btn.config(text="🔙 JANELA", bg='#E91E63')
        
        # Label de instruções
        instructions = tk.Label(self.fullscreen_window, 
                               text="ESC ou F11: Sair Fullscreen | ESPAÇO: Play/Pause | Apenas Vídeo", 
                               font=('Arial', 16), fg='white', bg='black')
        instructions.place(relx=0.5, y=50, anchor='center')
        
        self.logger.info("🖥️ Modo fullscreen apenas para vídeo ativado")
    
    def exit_fullscreen(self, event=None):
        """Sai do modo fullscreen"""
        if not self.is_fullscreen:
            return
        
        self.is_fullscreen = False
        
        # Destrói janela fullscreen
        if self.fullscreen_window:
            self.fullscreen_window.destroy()
            self.fullscreen_window = None
            self.fullscreen_canvas = None
        
        # Atualiza botão
        self.fullscreen_btn.config(text="🖥️ FULLSCREEN", bg='#9C27B0')
        
        # Retorna foco para janela principal
        self.root.focus_set()
        
        self.logger.info("🔙 Modo fullscreen desativado")
    
    def on_scale_change(self, event):
        """Atualiza configurações quando sliders mudam"""
        if self.model:
            self.model.conf = self.conf_var.get()
            self.model.iou = self.iou_var.get()
    
    def apply_config(self):
        """Aplica configurações ao modelo"""
        if self.model:
            self.model.conf = self.conf_var.get()
            self.model.iou = self.iou_var.get()
            messagebox.showinfo("Configuração", 
                              f"Configurações aplicadas!\nConf: {self.model.conf}\nIoU: {self.model.iou}")
    
    def on_patience_change(self, event):
        """Atualiza tempo de paciência quando slider muda"""
        patience = self.patience_var.get()
        self.snapshot_manager.set_patience_period(patience)
    
    def toggle_snapshot_system(self):
        """Inicia/para o sistema de snapshot"""
        if not self.snapshot_manager.is_monitoring:
            # Inicia sistema
            if self.snapshot_manager.start_monitoring():
                self.snapshot_toggle_btn.config(text="⏹️ PARAR SNAPSHOT", bg='#f44336')
                self.snapshot_status_label.config(text="📸 Sistema de Snapshot: ATIVO", fg='#4CAF50')
                messagebox.showinfo("Sistema de Snapshot", "✅ Sistema de snapshot iniciado com sucesso!")
            else:
                messagebox.showerror("Erro", "❌ Falha ao iniciar sistema de snapshot!")
        else:
            # Para sistema
            self.snapshot_manager.stop_monitoring()
            self.snapshot_toggle_btn.config(text="🔄 INICIAR SNAPSHOT", bg='#4CAF50')
            self.snapshot_status_label.config(text="📸 Sistema de Snapshot: INATIVO", fg='#FF6B6B')
            messagebox.showinfo("Sistema de Snapshot", "⏹️ Sistema de snapshot parado!")
    
    def show_snapshot_status(self):
        """Mostra status detalhado do sistema de snapshot"""
        status = self.snapshot_manager.get_status()
        
        status_text = f"""📊 STATUS DO SISTEMA DE SNAPSHOT
{'=' * 40}

🔄 Monitoramento: {'ATIVO' if status['is_monitoring'] else 'INATIVO'}
⏰ Período de paciência: {status['patience_period']} segundos
📊 Total de snapshots: {status['total_snapshots']}
📁 Diretório: {status['snapshot_directory']}

🚨 VIOLAÇÕES ATIVAS: {len(status['active_violations'])}"""

        if status['active_violations']:
            for violation in status['active_violations']:
                status_text += f"\n  • Pessoa {violation['person_id']} - {violation['epi_type']}"
                status_text += f"\n    Tempo decorrido: {violation['elapsed_time']:.1f}s"
                status_text += f"\n    Tempo restante: {violation['remaining_time']:.1f}s"
        else:
            status_text += "\n  ✅ Nenhuma violação ativa no momento"
        
        messagebox.showinfo("Status do Sistema de Snapshot", status_text)
    
    def show_snapshot_history(self):
        """Mostra histórico de snapshots"""
        history = self.snapshot_manager.get_snapshot_history(limit=20)
        
        if not history:
            messagebox.showinfo("Histórico de Snapshots", "📭 Nenhum snapshot encontrado")
            return
        
        history_text = f"📚 HISTÓRICO DE SNAPSHOTS\n{'=' * 40}\n\n"
        
        for i, snapshot in enumerate(reversed(history), 1):
            timestamp = snapshot['timestamp'].split('T')[1][:8]  # Apenas hora
            date = snapshot['timestamp'].split('T')[0]  # Apenas data
            
            history_text += f"{i:2d}. {date} {timestamp} - {snapshot['epi_type'].upper()}\n"
            history_text += f"     Pessoa: {snapshot['person_id']} | Arquivo: {snapshot['filename']}\n\n"
        
        # Cria janela de histórico com scroll
        history_window = tk.Toplevel(self.root)
        history_window.title("Histórico de Snapshots")
        history_window.geometry("600x400")
        history_window.configure(bg='#2b2b2b')
        
        # Texto com scroll
        text_widget = tk.Text(history_window, bg='#1b1b1b', fg='white', 
                             font=('Courier', 10), wrap=tk.WORD)
        scrollbar = tk.Scrollbar(history_window, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        text_widget.insert(tk.END, history_text)
        text_widget.config(state=tk.DISABLED)
        
        # Botão fechar
        close_btn = tk.Button(history_window, text="✅ FECHAR", 
                             command=history_window.destroy,
                             font=('Arial', 11, 'bold'),
                             bg='#4CAF50', fg='white')
        close_btn.pack(pady=10)
    
    def configure_snapshot_patience(self):
        """Configura o período de paciência do sistema de snapshot"""
        current_period = self.snapshot_manager.patience_period
        
        # Cria janela de configuração
        config_window = tk.Toplevel(self.root)
        config_window.title("Configurar Tempo de Paciência")
        config_window.geometry("400x200")
        config_window.configure(bg='#2b2b2b')
        config_window.transient(self.root)
        config_window.grab_set()
        
        # Frame principal
        main_frame = tk.Frame(config_window, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_label = tk.Label(main_frame, text="⏰ CONFIGURAÇÃO DO TEMPO DE PACIÊNCIA", 
                               font=('Arial', 14, 'bold'), fg='white', bg='#2b2b2b')
        title_label.pack(pady=(0, 20))
        
        # Informação atual
        current_label = tk.Label(main_frame, text=f"Período atual: {current_period} segundos", 
                                font=('Arial', 11), fg='#4CAF50', bg='#2b2b2b')
        current_label.pack(pady=(0, 15))
        
        # Frame para controles
        controls_frame = tk.Frame(main_frame, bg='#2b2b2b')
        controls_frame.pack(pady=(0, 20))
        
        # Label e entrada
        tk.Label(controls_frame, text="Novo período (segundos):", 
                font=('Arial', 10), fg='white', bg='#2b2b2b').pack(side=tk.LEFT, padx=(0, 10))
        
        period_var = tk.DoubleVar(value=current_period)
        period_entry = tk.Entry(controls_frame, textvariable=period_var, 
                               font=('Arial', 11), width=10)
        period_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botões
        button_frame = tk.Frame(main_frame, bg='#2b2b2b')
        button_frame.pack()
        
        # Botão aplicar
        apply_btn = tk.Button(button_frame, text="✅ APLICAR", 
                             command=lambda: self._apply_patience_config(period_var.get(), config_window), 
                             font=('Arial', 11, 'bold'),
                             bg='#4CAF50', fg='white', padx=15, pady=5)
        apply_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Botão cancelar
        cancel_btn = tk.Button(button_frame, text="❌ CANCELAR", 
                              command=config_window.destroy, 
                              font=('Arial', 11, 'bold'),
                              bg='#f44336', fg='white', padx=15, pady=5)
        cancel_btn.pack(side=tk.LEFT)
        
        # Foca na entrada
        period_entry.focus()
        period_entry.select_range(0, tk.END)
    
    def _apply_patience_config(self, new_period: float, window):
        """Aplica nova configuração de paciência"""
        try:
            if new_period < 1.0 or new_period > 60.0:
                messagebox.showerror("Erro", "Período deve estar entre 1 e 60 segundos!")
                return
            
            # Aplica configuração
            self.snapshot_manager.set_patience_period(new_period)
            self.patience_var.set(new_period)
            
            messagebox.showinfo("Sucesso", f"✅ Período de paciência alterado para {new_period} segundos!")
            window.destroy()
            
        except ValueError:
            messagebox.showerror("Erro", "Valor inválido! Digite um número.")
    
    def update_snapshot_status(self):
        """Atualiza status do sistema de snapshot na interface"""
        status = self.snapshot_manager.get_status()
        
        # Atualiza labels
        self.snapshots_label.config(text=f"📸 Snapshots: {status['total_snapshots']}")
        self.violations_label.config(text=f"🚨 Violações Ativas: {len(status['active_violations'])}")
        
        # Atualiza label de status principal
        if status['is_monitoring']:
            self.snapshot_status_label.config(text=f"📸 Sistema de Snapshot: ATIVO ({status['total_snapshots']} snapshots)", fg='#4CAF50')
        else:
            self.snapshot_status_label.config(text="📸 Sistema de Snapshot: INATIVO", fg='#FF6B6B')
    
    def detect_image(self):
        """Detecta EPIs em imagem com sistema anti-veículo"""
        if not self.model:
            messagebox.showerror("Erro", "Modelo não carregado!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Carrega imagem
            image = cv2.imread(file_path)
            if image is None:
                messagebox.showerror("Erro", "Não foi possível carregar a imagem!")
                return
            
            # Reset contadores
            self.rejected_objects = 0
            
            # Detecta objetos com filtro anti-veículo
            detections = self.detect_objects_with_vehicle_filter(image)
            
            # Valida EPIs
            validated_people = self.validate_epis(detections)
            
            # Salva detecções para uso posterior
            self._last_detections = detections
            
            # Processa violações para o sistema de snapshot
            if self.snapshot_manager.is_monitoring:
                self.snapshot_manager.process_violations(validated_people, image)
            
            # Desenha resultados
            result_image = self.draw_results(image, detections, validated_people)
            
            # Exibe resultado
            self.display_image(result_image)
            self.current_display_frame = result_image  # Para fullscreen
            
            # Atualiza status
            human_count = len([d for d in detections if d['class_name'] == 'person'])
            self.status_label.config(text=f"Status: Imagem processada - {len(detections)} detecções")
            self.detection_label.config(text=f"Humanos: {human_count} | Rejeitados: {self.rejected_objects}")
            
            # Atualiza status do snapshot
            self.update_snapshot_status()
            
            # Salva resultado
            self.save_result(result_image, "image")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagem: {str(e)}")
    
    def detect_video(self):
        """Detecta EPIs em vídeo com fullscreen"""
        if not self.model:
            messagebox.showerror("Erro", "Modelo não carregado!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Selecione um vídeo",
            filetypes=[
                ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Todos os arquivos", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Abre vídeo
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                messagebox.showerror("Erro", "Não foi possível abrir o vídeo!")
                return
            
            # Reset contadores
            self.rejected_objects = 0
            
            # Processa vídeo
            self.process_video(cap, file_path)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar vídeo: {str(e)}")
    
    def process_video(self, cap, file_path):
        """Processa vídeo com sistema anti-veículo e fullscreen"""
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configurações de reprodução
        self.is_video_playing = True
        self.video_paused = False
        self.current_frame = 0
        self.video_cap = cap
        self.video_fps = fps
        
        # Cria controles de vídeo
        self.create_video_window(cap, file_path)
        
        # Inicia reprodução em thread separada
        self.video_thread = threading.Thread(target=self.video_loop_with_vehicle_filter, daemon=True)
        self.video_thread.start()
    
    def video_loop_with_vehicle_filter(self):
        """Loop de vídeo com filtro anti-veículo e suporte a fullscreen"""
        try:
            target_fps = 30
            frame_delay = 1.0 / target_fps
            
            while self.is_video_playing and self.video_cap and self.video_cap.isOpened():
                if not self.video_paused:
                    # Usa lock para sincronizar acesso ao decoder (resolve erro libavcodec)
                    with self.video_lock:
                        ret, frame = self.video_cap.read()
                        if not ret:
                            break
                    
                    # Detecta com filtro anti-veículo
                    detections = self.detect_objects_with_vehicle_filter(frame)
                    validated_people = self.validate_epis(detections)
                    
                    # Processa violações para o sistema de snapshot
                    if self.snapshot_manager.is_monitoring:
                        self.snapshot_manager.process_violations(validated_people, frame)
                    
                    result_frame = self.draw_results(frame, detections, validated_people)
                    self.current_display_frame = result_frame  # Para fullscreen
                    
                    # Atualiza interface na thread principal
                    self.root.after(0, self.update_video_frame, result_frame, detections, validated_people)
                    
                    time.sleep(frame_delay)
                    self.current_frame += 1
                    
                    if not self.root.winfo_exists():
                        break
                else:
                    time.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"❌ Erro no loop de vídeo: {e}")
        finally:
            # Usa lock para liberar recursos de vídeo
            with self.video_lock:
                if self.video_cap:
                    self.video_cap.release()
            self.is_video_playing = False
            # Remove controles de vídeo
            self.root.after(0, self.cleanup_video_controls)
    
    def create_video_window(self, cap, file_path):
        """Cria controles de vídeo com opção fullscreen"""
        self.video_controls = tk.Frame(self.root, bg='#3b3b3b', relief='raised', bd=2)
        self.video_controls.pack(fill=tk.X, pady=(10, 0))
        
        # Frame para botões
        buttons_frame = tk.Frame(self.video_controls, bg='#3b3b3b')
        buttons_frame.pack(side=tk.LEFT, padx=10, pady=8)
        
        # Botão Play/Pause
        self.play_pause_btn = tk.Button(buttons_frame, text="⏸️ PAUSAR", 
                                       command=self.toggle_video_pause, 
                                       font=('Arial', 11, 'bold'),
                                       bg='#FF9800', fg='white', 
                                       relief='raised', bd=2, padx=15, pady=6)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        # Botão Fullscreen para vídeo
        self.video_fullscreen_btn = tk.Button(buttons_frame, text="🖥️ FULLSCREEN", 
                                             command=self.toggle_fullscreen, 
                                             font=('Arial', 11, 'bold'),
                                             bg='#9C27B0', fg='white', 
                                             relief='raised', bd=2, padx=15, pady=6)
        self.video_fullscreen_btn.pack(side=tk.LEFT, padx=5)
        
        # Botão parar vídeo
        self.stop_video_btn = tk.Button(buttons_frame, text="⏹️ PARAR VÍDEO", 
                                       command=self.stop_video, 
                                       font=('Arial', 11, 'bold'),
                                       bg='#f44336', fg='white', 
                                       relief='raised', bd=2, padx=15, pady=6)
        self.stop_video_btn.pack(side=tk.LEFT, padx=5)
        
        # Labels de status
        status_frame = tk.Frame(self.video_controls, bg='#3b3b3b')
        status_frame.pack(side=tk.RIGHT, padx=10, pady=8)
        
        self.video_status_label = tk.Label(status_frame, text="🎥 Vídeo com filtro anti-veículo", 
                                          font=('Arial', 11), fg='white', bg='#3b3b3b')
        self.video_status_label.pack(side=tk.RIGHT, padx=10)
        
        self.video_progress_label = tk.Label(status_frame, text="Frame: 0", 
                                            font=('Arial', 11), fg='#FFB74D', bg='#3b3b3b')
        self.video_progress_label.pack(side=tk.RIGHT, padx=10)
    
    def toggle_video_pause(self, event=None):
        """Alterna pausa do vídeo"""
        # Só funciona se estiver reproduzindo vídeo
        if not hasattr(self, 'is_video_playing') or not self.is_video_playing:
            return
        
        self.video_paused = not self.video_paused
        if self.video_paused:
            self.play_pause_btn.config(text="▶️ CONTINUAR", bg='#4CAF50')
        else:
            self.play_pause_btn.config(text="⏸️ PAUSAR", bg='#FF9800')
    
    def toggle_video_pause_fullscreen(self, event=None):
        """Alterna pausa do vídeo no modo fullscreen (barra de espaço)"""
        # Só funciona se estiver reproduzindo vídeo
        if not hasattr(self, 'is_video_playing') or not self.is_video_playing:
            return
        
        self.video_paused = not self.video_paused
        
        # Atualiza botão na interface principal se existir
        if hasattr(self, 'play_pause_btn') and self.play_pause_btn.winfo_exists():
            if self.video_paused:
                self.play_pause_btn.config(text="▶️ CONTINUAR", bg='#4CAF50')
            else:
                self.play_pause_btn.config(text="⏸️ PAUSAR", bg='#FF9800')
        
        # Mostra feedback visual no fullscreen
        if self.is_fullscreen and self.fullscreen_canvas:
            self.show_fullscreen_feedback()
        
        self.logger.info(f"🎬 Vídeo {'pausado' if self.video_paused else 'continuado'} via fullscreen")
    
    def show_fullscreen_feedback(self):
        """Mostra feedback visual no fullscreen para play/pause"""
        if not self.fullscreen_canvas:
            return
        
        try:
            # Cria texto de feedback
            feedback_text = "⏸️ PAUSADO" if self.video_paused else "▶️ REPRODUZINDO"
            feedback_color = "#FF9800" if self.video_paused else "#4CAF50"
            
            # Remove feedback anterior se existir
            self.fullscreen_canvas.delete("feedback")
            
            # Cria retângulo de fundo
            text_width = len(feedback_text) * 12  # Aproximação da largura do texto
            x = self.fullscreen_window.winfo_screenwidth() // 2
            y = 100
            
            # Desenha retângulo de fundo
            self.fullscreen_canvas.create_rectangle(
                x - text_width//2 - 20, y - 15,
                x + text_width//2 + 20, y + 15,
                fill='black', outline=feedback_color, width=3,
                tags="feedback"
            )
            
            # Desenha texto
            self.fullscreen_canvas.create_text(
                x, y, text=feedback_text,
                font=('Arial', 18, 'bold'),
                fill=feedback_color,
                tags="feedback"
            )
            
            # Remove feedback após 2 segundos
            self.fullscreen_window.after(2000, self.remove_fullscreen_feedback)
            
        except Exception as e:
            self.logger.error(f"Erro ao mostrar feedback fullscreen: {e}")
    
    def remove_fullscreen_feedback(self):
        """Remove feedback visual do fullscreen"""
        if self.fullscreen_canvas:
            self.fullscreen_canvas.delete("feedback")
    
    def stop_video(self):
        """Para reprodução do vídeo"""
        self.is_video_playing = False
        self.video_paused = False
        
        # Usa lock para liberar recursos de vídeo de forma segura
        with self.video_lock:
            if hasattr(self, 'video_cap') and self.video_cap:
                self.video_cap.release()
                self.video_cap = None
        
        # Sai do fullscreen se estiver ativo
        if self.is_fullscreen:
            self.exit_fullscreen()
    
    def cleanup_video_controls(self):
        """Remove controles de vídeo da interface"""
        try:
            # Libera recursos de vídeo de forma segura
            with self.video_lock:
                if hasattr(self, 'video_cap') and self.video_cap:
                    self.video_cap.release()
                    self.video_cap = None
            
            # Remove controles da interface
            if hasattr(self, 'video_controls') and self.video_controls.winfo_exists():
                self.video_controls.destroy()
        except Exception as e:
            self.logger.error(f"Erro ao limpar controles de vídeo: {e}")
    
    def update_video_frame(self, frame, detections, validated_people):
        """Atualiza frame do vídeo com suporte a fullscreen"""
        try:
            # Atualiza canvas principal
            self.display_image(frame)
            
            # Atualiza canvas fullscreen se ativo
            if self.is_fullscreen and self.fullscreen_canvas:
                self.display_image_fullscreen(frame)
            
            # Atualiza status
            human_count = len([d for d in detections if d['class_name'] == 'person'])
            self.status_label.config(text=f"Status: Frame {self.current_frame} - Filtro ativo")
            self.detection_label.config(text=f"Humanos: {human_count} | Rejeitados: {self.rejected_objects}")
            
            # Atualiza progresso do vídeo
            if hasattr(self, 'video_progress_label'):
                self.video_progress_label.config(text=f"Frame: {self.current_frame}")
                
        except Exception as e:
            self.logger.error(f"❌ Erro ao atualizar frame: {e}")
    
    def display_image_fullscreen(self, image):
        """Exibe imagem no canvas fullscreen separado"""
        if not self.fullscreen_canvas:
            return
        
        try:
            # Pega dimensões da tela
            screen_width = self.fullscreen_window.winfo_screenwidth()
            screen_height = self.fullscreen_window.winfo_screenheight()
            
            # Converte BGR para RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensiona mantendo proporção para tela cheia
            h, w = image_rgb.shape[:2]
            scale = min(screen_width / w, screen_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
            
            # Converte para PIL
            pil_image = Image.fromarray(image_resized)
            self.fullscreen_image = ImageTk.PhotoImage(pil_image)
            
            # Limpa e exibe nova imagem
            self.fullscreen_canvas.delete("all")
            self.fullscreen_canvas.create_image(screen_width//2, screen_height//2, 
                                               image=self.fullscreen_image, anchor=tk.CENTER)
        except Exception as e:
            self.logger.error(f"Erro ao exibir imagem fullscreen: {e}")
    
    def toggle_camera(self):
        """Alterna câmera ao vivo"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Inicia câmera com sistema anti-veículo e fullscreen"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Não foi possível abrir a câmera!")
                return
            
            # Configurações otimizadas para canteiro de obra
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Removido: ajustes de brilho/contraste/saturação para usar padrão da câmera
            
            self.is_camera_running = True
            self.camera_btn.config(text="📹 PARAR CÂMERA", bg='#f44336')
            self.stop_btn.config(state=tk.NORMAL)
            
            # Reset contadores
            self.frame_count = 0
            self.rejected_objects = 0
            self.start_time = time.time()
            
            # Inicia thread de câmera
            self.camera_thread = threading.Thread(target=self.camera_loop_with_vehicle_filter, daemon=True)
            self.camera_thread.start()
            
            self.status_label.config(text="Status: Câmera ativa - Sistema anti-veículo operacional")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")
    
    def stop_camera(self):
        """Para câmera ao vivo"""
        self.is_camera_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Para vídeo se estiver rodando
        if hasattr(self, 'is_video_playing'):
            self.is_video_playing = False
        
        # Sai do fullscreen
        if self.is_fullscreen:
            self.exit_fullscreen()
        
        try:
            if self.root.winfo_exists():
                self.camera_btn.config(text="📹 CÂMERA AO VIVO", bg='#FF9800')
                self.stop_btn.config(state=tk.DISABLED)
                self.status_label.config(text="Status: Sistema parado")
                
                # Remove controles de vídeo se existirem
                self.cleanup_video_controls()
        except:
            pass
    
    def camera_loop_with_vehicle_filter(self):
        """Loop principal da câmera com filtro anti-veículo e fullscreen"""
        while self.is_camera_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Detecta com filtro anti-veículo
                    detections = self.detect_objects_with_vehicle_filter(frame)
                    
                    # Valida EPIs apenas em humanos válidos
                    validated_people = self.validate_epis(detections)
                    
                    # Processa violações para o sistema de snapshot
                    if self.snapshot_manager.is_monitoring:
                        self.snapshot_manager.process_violations(validated_people, frame)
                    
                    # Desenha resultados
                    result_frame = self.draw_results(frame, detections, validated_people)
                    self.current_display_frame = result_frame  # Para fullscreen
                    
                    # Adiciona à fila para exibição
                    if not self.frame_queue.full():
                        self.frame_queue.put(result_frame)
                    
                    # Atualiza estatísticas
                    self.frame_count += 1
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    human_count = len([d for d in detections if d['class_name'] == 'person'])
                    
                    # Atualiza interface na thread principal
                    self.root.after(0, self.update_camera_stats_with_filter, fps, human_count, result_frame)
            
            time.sleep(0.033)  # ~30 FPS
    
    def update_camera_stats_with_filter(self, fps, human_count, frame):
        """Atualiza estatísticas com informações do filtro anti-veículo e fullscreen"""
        try:
            if self.root.winfo_exists():
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.detection_label.config(text=f"Humanos: {human_count} | Rejeitados: {self.rejected_objects}")
                
                # Atualiza imagem se houver frame na fila
                try:
                    if not self.frame_queue.empty():
                        display_frame = self.frame_queue.get_nowait()
                        self.display_image(display_frame)
                        
                        # Atualiza fullscreen se ativo
                        if self.is_fullscreen and self.fullscreen_canvas:
                            self.display_image_fullscreen(display_frame)
                        
                except queue.Empty:
                    pass
        except:
            pass
    
    def detect_objects_with_vehicle_filter(self, frame):
        """
        Detecção de objetos com filtro anti-veículo integrado
        MÉTODO PRINCIPAL OTIMIZADO - RESOLVE PROBLEMA DA MOTO
        """
        if self.model is None:
            return []
        
        try:
            # Reset contador de rejeitados para este frame
            frame_rejected = 0
            
            # Redimensiona para YOLO (otimizado)
            frame_resized = cv2.resize(frame, (640, 640))
            
            # Detecção YOLO
            results = self.model(frame_resized)
            
            # Converte coordenadas
            h, w = frame.shape[:2]
            detections = []
            
            if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                detections_array = results.xyxy[0].cpu().numpy()
                
                for detection in detections_array:
                    x1, y1, x2, y2, conf, cls = detection
                    
                    # Converte coordenadas para frame original
                    x1 = int(x1 * w / 640)
                    y1 = int(y1 * h / 640)
                    x2 = int(x2 * w / 640)
                    y2 = int(y2 * h / 640)
                    
                    if 0 <= int(cls) < len(self.classes):
                        class_name = self.classes[int(cls)]
                        
                        # 🛡️ APLICA FILTRO ANTI-VEÍCULO APENAS PARA 'PERSON'
                        if class_name == 'person':
                            detection_data = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'class_id': int(cls),
                                'class_name': class_name,
                                'frame_number': self.frame_count,
                                'timestamp': time.time()
                            }
                            
                            # 🔍 CLASSIFICAÇÃO ANTI-VEÍCULO
                            object_type = self.human_validator.classify_object(detection_data)
                            
                            if object_type == ObjectType.HUMAN:
                                # ✅ Validado como humano
                                detections.append(detection_data)
                                self.logger.debug(f"✅ Humano validado: {x2-x1}x{y2-y1}px, conf: {conf:.2f}")
                            else:
                                # ❌ Rejeitado como veículo
                                frame_rejected += 1
                                geometry = ObjectGeometry.from_bbox([x1, y1, x2, y2])
                                self.logger.info(f"🚫 {object_type.value.upper()} rejeitado: "
                                               f"{geometry.width}x{geometry.height}px, "
                                               f"ratio: {geometry.aspect_ratio:.2f}, "
                                               f"conf: {conf:.2f}")
                        else:
                            # Outras classes (helmet, vest, etc.) passam direto
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'class_id': int(cls),
                                'class_name': class_name
                            })
            
            # Atualiza contador global de rejeitados
            self.rejected_objects += frame_rejected
            
            return detections
            
        except Exception as e:
            self.logger.error(f"❌ Erro na detecção com filtro: {e}")
            return []
    
    def validate_epis(self, detections):
        """Validação inteligente de EPIs baseada nas detecções da IA"""
        people = [d for d in detections if d['class_name'] == 'person']
        helmets = [d for d in detections if d['class_name'] == 'helmet']
        vests = [d for d in detections if d['class_name'] == 'vest']
        no_helmets = [d for d in detections if d['class_name'] == 'no-helmet']
        no_vests = [d for d in detections if d['class_name'] == 'no-vest']
        
        validated_people = []
        
        for person in people:
            px1, py1, px2, py2 = person['bbox']
            person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            person_height = py2 - py1
            
            # Procura capacete próximo da cabeça (zona expandida)
            helmet_found = False
            best_helmet_distance = float('inf')
            
            for helmet in helmets:
                hx1, hy1, hx2, hy2 = helmet['bbox']
                helmet_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)
                
                # Distância euclidiana
                distance = np.sqrt((person_center[0] - helmet_center[0])**2 + 
                                 (person_center[1] - helmet_center[1])**2)
                
                # Capacete na região da cabeça (top 35% da pessoa)
                head_zone_bottom = py1 + (person_height * 0.35)
                
                if distance < 120 and hy2 <= head_zone_bottom and distance < best_helmet_distance:
                    helmet_found = True
                    best_helmet_distance = distance
            
            # Procura colete próximo do torso (zona expandida)
            vest_found = False
            best_vest_distance = float('inf')
            
            for vest in vests:
                vx1, vy1, vx2, vy2 = vest['bbox']
                vest_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
                
                distance = np.sqrt((person_center[0] - vest_center[0])**2 + 
                                 (person_center[1] - vest_center[1])**2)
                
                # Colete na região do torso (30-75% da altura da pessoa)
                torso_start = py1 + (person_height * 0.30)
                torso_end = py1 + (person_height * 0.75)
                
                if (distance < 160 and vy1 >= torso_start and vy2 <= torso_end and 
                    distance < best_vest_distance):
                    vest_found = True
                    best_vest_distance = distance
            
            # Verifica se há detecções de violação da IA próximas à pessoa
            helmet_violation_detected = False
            vest_violation_detected = False
            
            # Procura violações de capacete próximas à pessoa
            for no_helmet in no_helmets:
                nhx1, nhy1, nhx2, nhy2 = no_helmet['bbox']
                no_helmet_center = ((nhx1 + nhx2) // 2, (nhy1 + nhy2) // 2)
                
                # Distância para violação de capacete
                distance = np.sqrt((person_center[0] - no_helmet_center[0])**2 + 
                                 (person_center[1] - no_helmet_center[1])**2)
                
                # Se a violação está próxima à pessoa, considera como violação
                if distance < 150:  # Distância maior para capturar violações
                    helmet_violation_detected = True
                    break
            
            # Procura violações de colete próximas à pessoa
            for no_vest in no_vests:
                nvx1, nvy1, nvx2, nvy2 = no_vest['bbox']
                no_vest_center = ((nvx1 + nvx2) // 2, (nvy1 + nvy2) // 2)
                
                # Distância para violação de colete
                distance = np.sqrt((person_center[0] - no_vest_center[0])**2 + 
                                 (person_center[1] - no_vest_center[1])**2)
                
                # Se a violação está próxima à pessoa, considera como violação
                if distance < 200:  # Distância maior para capturar violações
                    vest_violation_detected = True
                    break
            
            # Determina status final dos EPIs (IA tem prioridade sobre validação geométrica)
            if helmet_violation_detected:
                helmet_status = "AUSENTE"
            elif helmet_found:
                helmet_status = "CORRETO"
            else:
                helmet_status = "AUSENTE"
                
            if vest_violation_detected:
                vest_status = "AUSENTE"
            elif vest_found:
                vest_status = "CORRETO"
            else:
                vest_status = "AUSENTE"
            
            # Cria objeto pessoa para o sistema de snapshot
            person_obj = {
                'person': person,
                'helmet_found': helmet_found and not helmet_violation_detected,
                'vest_found': vest_found and not vest_violation_detected,
                'helmet_status': helmet_status,
                'vest_status': vest_status,
                'helmet_violation_detected': helmet_violation_detected,
                'vest_violation_detected': vest_violation_detected,
                'bbox': person['bbox'],  # Bounding box da pessoa para snapshot
                'helmet_violation_bbox': None,  # Bounding box específico da violação de capacete
                'vest_violation_bbox': None     # Bounding box específico da violação de colete
            }
            
            # Armazena bounding boxes específicos das violações detectadas
            if helmet_violation_detected:
                for no_helmet in no_helmets:
                    nhx1, nhy1, nhx2, nhy2 = no_helmet['bbox']
                    no_helmet_center = ((nhx1 + nhx2) // 2, (nhy1 + nhy2) // 2)
                    distance = np.sqrt((person_center[0] - no_helmet_center[0])**2 + 
                                     (person_center[1] - no_helmet_center[1])**2)
                    if distance < 150:
                        person_obj['helmet_violation_bbox'] = [nhx1, nhy1, nhx2, nhy2]
                        break
            
            if vest_violation_detected:
                for no_vest in no_vests:
                    nvx1, nvy1, nvx2, nvy2 = no_vest['bbox']
                    no_vest_center = ((nvx1 + nvx2) // 2, (nvy1 + nvy2) // 2)
                    distance = np.sqrt((person_center[0] - no_vest_center[0])**2 + 
                                     (person_center[1] - no_vest_center[1])**2)
                    if distance < 200:
                        person_obj['vest_violation_bbox'] = [nvx1, nvy1, nvx2, nvy2]
                        break
            
            validated_people.append(person_obj)
        
        return validated_people
    
    def draw_results(self, frame, detections, validated_people):
        """Desenha resultados limpos com apenas detecções essenciais"""
        frame_copy = frame.copy()
        
        # Desenha todas as detecções com bounding boxes simples
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            conf = detection['confidence']
            
            color = self.colors.get(class_name, (128, 128, 128))
            
            # Desenha bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Label simples apenas com classe e confiança
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Destaca violações detectadas pela IA com bounding boxes específicos
        for person_data in validated_people:
            person = person_data['person']
            px1, py1, px2, py2 = person['bbox']
            
            # Destaca violações com borda vermelha
            if person_data['helmet_violation_detected'] or person_data['vest_violation_detected']:
                # Borda vermelha grossa para violações
                cv2.rectangle(frame_copy, (px1, py1), (px2, py2), (0, 0, 255), 4)
                
                # Desenha bounding boxes específicos das violações
                if person_data['helmet_violation_detected'] and person_data.get('helmet_violation_bbox'):
                    hx1, hy1, hx2, hy2 = person_data['helmet_violation_bbox']
                    cv2.rectangle(frame_copy, (hx1, hy1), (hx2, hy2), (0, 0, 255), 3)
                
                if person_data['vest_violation_detected'] and person_data.get('vest_violation_bbox'):
                    vx1, vy1, vx2, vy2 = person_data['vest_violation_bbox']
                    cv2.rectangle(frame_copy, (vx1, vy1), (vx2, vy2), (0, 0, 255), 3)
        
        return frame_copy
    

    
    def display_image(self, image):
        """Exibe imagem no canvas principal"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.height()
        
        if canvas_width > 1 and canvas_height > 1:
            # Converte BGR para RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Redimensiona mantendo proporção
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            image_resized = cv2.resize(image_rgb, (new_w, new_h))
            
            # Converte para PIL
            pil_image = Image.fromarray(image_resized)
            self.current_image = ImageTk.PhotoImage(pil_image)
            
            # Exibe no canvas
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                   image=self.current_image, anchor=tk.CENTER)
            
            # Atualiza detecções atuais para sistema de info
            self.current_detections = getattr(self, '_last_detections', [])
            
            # Salva detecções para uso posterior
            self._last_detections = self.current_detections
    
    def save_result(self, image, source_type):
        """Salva resultado da detecção"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"epi_detection_{source_type}_{timestamp}.jpg"
        
        # Cria diretório se não existir
        Path("output").mkdir(exist_ok=True)
        output_path = Path("output") / filename
        
        cv2.imwrite(str(output_path), image)
        self.logger.info(f"📸 Resultado salvo: {output_path}")
        
        # Mensagem na interface
        self.status_label.config(text=f"Status: Resultado salvo - {filename}")
    
    def update_interface(self):
        """Atualiza interface periodicamente"""
        try:
            if self.root.winfo_exists():
                # Processa teclas globais
                self.root.after(100, self.update_interface)
                
                # Atualiza fullscreen se necessário
                if self.is_fullscreen and self.fullscreen_canvas and self.current_display_frame is not None:
                    self.display_image_fullscreen(self.current_display_frame)
        except:
            pass
    
    def handle_keypress(self, event):
        """Manipula teclas pressionadas"""
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
                self.stop_camera()
    
    def run(self):
        """Executa interface com suporte a teclas globais"""
        try:
            # Bind teclas globais
            self.root.bind('<Key>', self.handle_keypress)
            self.root.focus_set()

            # Torna janela focalizável
            self.root.attributes('-topmost', False)

            self.root.mainloop()
        except KeyboardInterrupt:
            self.logger.info("🛑 Interface interrompida pelo usuário")
        finally:
            self.stop_camera()
            if self.is_fullscreen:
                self.exit_fullscreen()

    # ========================================================================
    # MÉTODOS AUXILIARES E COMPLEMENTARES
    # ========================================================================
    
    def is_valid_detection_confidence(self, class_id, confidence):
        """Valida se a confiança é adequada para cada classe"""
        # Thresholds otimizados para canteiro de obra
        confidence_thresholds = {
            0: 0.25,  # helmet - sensível
            1: 0.30,  # no-helmet - rigoroso para evitar falsos alarmes
            2: 0.27,  # no-vest - levemente mais permissivo para recuperar recall
            3: 0.15,  # person - MUITO sensível (detecta pessoas distantes)
            4: 0.18   # vest - um pouco mais sensível
        }
        threshold = confidence_thresholds.get(int(class_id), 0.20)
        return confidence >= threshold
    
    def track_person_temporally(self, person_detection):
        """Sistema de tracking temporal para melhorar detecção de pessoas"""
        bbox = person_detection['bbox']
        confidence = person_detection['confidence']
        person_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        
        # Procura por track existente próximo
        best_track_id = None
        best_distance = float('inf')
        
        for track_id, track_data in self.person_tracks.items():
            track_bbox = track_data['bbox']
            track_center = ((track_bbox[0] + track_bbox[2]) // 2, (track_bbox[1] + track_bbox[3]) // 2)
            
            distance = np.sqrt((person_center[0] - track_center[0])**2 + 
                             (person_center[1] - track_center[1])**2)
            
            if distance < 100 and distance < best_distance:
                best_distance = distance
                best_track_id = track_id
        
        if best_track_id is not None:
            # Atualiza track existente
            track_data = self.person_tracks[best_track_id]
            track_data.update({
                'bbox': bbox,
                'confidence': confidence,
                'frames_seen': track_data['frames_seen'] + 1,
                'last_seen': time.time()
            })
            
            # Adiciona ao histórico de movimento
            if 'movement_history' not in track_data:
                track_data['movement_history'] = []
            
            track_data['movement_history'].append({
                'center': person_center,
                'frame': self.frame_count,
                'bbox': bbox,
                'timestamp': time.time()
            })
            
            # Mantém apenas os últimos 10 frames
            if len(track_data['movement_history']) > 10:
                track_data['movement_history'] = track_data['movement_history'][-10:]
            
            return best_track_id
        else:
            # Cria novo track
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.person_tracks[track_id] = {
                'bbox': bbox,
                'confidence': confidence,
                'frames_seen': 1,
                'last_seen': time.time(),
                'movement_history': [{
                    'center': person_center,
                    'frame': self.frame_count,
                    'bbox': bbox,
                    'timestamp': time.time()
                }]
            }
            return track_id
    
    def cleanup_old_tracks(self):
        """Remove tracks antigos para otimizar memória"""
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, track_data in self.person_tracks.items():
            # Remove tracks não vistos há mais de 5 segundos
            if current_time - track_data['last_seen'] > 5.0:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.person_tracks[track_id]
    
    def save_screenshot_with_timestamp(self):
        """Salva screenshot com timestamp detalhado"""
        if self.current_display_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_epi_{timestamp}.jpg"
            
            Path("screenshots").mkdir(exist_ok=True)
            output_path = Path("screenshots") / filename
            
            cv2.imwrite(str(output_path), self.current_display_frame)
            
            # Adiciona informação na imagem
            info_text = f"Screenshot: {timestamp} | Frame: {self.frame_count}"
            self.status_label.config(text=f"Status: {info_text}")
            
            self.logger.info(f"📸 Screenshot salvo: {output_path}")
            return str(output_path)
        
        return None
    
    def get_system_statistics(self):
        """Retorna estatísticas do sistema"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'elapsed_time': elapsed_time,
            'fps': fps,
            'rejected_objects': self.rejected_objects,
            'active_tracks': len(self.person_tracks),
            'model_confidence': self.model.conf if self.model else 0,
            'model_iou': self.model.iou if self.model else 0,
            'is_fullscreen': self.is_fullscreen,
            'is_camera_running': self.is_camera_running,
            'is_video_playing': getattr(self, 'is_video_playing', False)
        }
    
    def export_detection_log(self):
        """Exporta log de detecções para análise"""
        stats = self.get_system_statistics()
        
        log_data = {
            'session_info': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'duration': stats['elapsed_time'],
                'total_frames': stats['frames_processed']
            },
            'performance': {
                'avg_fps': stats['fps'],
                'rejected_objects': stats['rejected_objects'],
                'active_tracks': stats['active_tracks']
            },
            'configuration': {
                'model_confidence': stats['model_confidence'],
                'model_iou': stats['model_iou']
            },
            'tracks': self.person_tracks
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = Path("logs") / f"epi_detection_session_{timestamp}.json"
        log_file.parent.mkdir(exist_ok=True)
        
        import json
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        self.logger.info(f"📊 Log exportado: {log_file}")
        return str(log_file)
    
    def show_system_info_dialog(self):
        """Mostra diálogo com informações detalhadas do sistema"""
        stats = self.get_system_statistics()
        
        info_window = tk.Toplevel(self.root)
        info_window.title("Informações do Sistema")
        info_window.geometry("500x400")
        info_window.configure(bg='#2b2b2b')
        
        # Texto de informações
        info_text = f"""
🛡️ SISTEMA DE DETECÇÃO DE EPIs
{'=' * 40}

📊 PERFORMANCE:
   Frames Processados: {stats['frames_processed']}
   FPS Médio: {stats['fps']:.1f}
   Tempo Decorrido: {stats['elapsed_time']:.1f}s

🚫 FILTRO ANTI-VEÍCULO:
   Objetos Rejeitados: {stats['rejected_objects']}
   Tracks Ativos: {stats['active_tracks']}

⚙️ CONFIGURAÇÕES:
   Model Confidence: {stats['model_confidence']}
   Model IoU: {stats['model_iou']}

📺 STATUS:
   Fullscreen: {'Ativo' if stats['is_fullscreen'] else 'Inativo'}
   Câmera: {'Ativa' if stats['is_camera_running'] else 'Inativa'}
   Vídeo: {'Reproduzindo' if stats['is_video_playing'] else 'Parado'}

🎯 REGRAS ANTI-MOTO:
   Proporção Width/Height: 0.6-1.4 = REJEITADO
   Área Típica: 4.000-50.000 pixels
   Largura Mínima: 70px
   Validação Multi-Camadas: ATIVA
        """
        
        text_widget = tk.Text(info_window, bg='#1b1b1b', fg='white', 
                             font=('Courier', 10), wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
        
        # Botão fechar
        close_btn = tk.Button(info_window, text="✅ FECHAR", 
                             command=info_window.destroy,
                             font=('Arial', 11, 'bold'),
                             bg='#4CAF50', fg='white')
        close_btn.pack(pady=10)
    
    # ========================================================================
    # COMPATIBILIDADE COM CÓDIGO LEGADO (se necessário)
    # ========================================================================
    
    def detect_objects(self, frame):
        """Método de compatibilidade - usa o novo sistema com filtro"""
        return self.detect_objects_with_vehicle_filter(frame)
    
    def is_likely_human(self, width, height, person_detection):
        """Método de compatibilidade - usa novo classificador"""
        detection_data = {
            'bbox': person_detection['bbox'],
            'confidence': person_detection['confidence'],
            'class_name': person_detection['class_name'],
            'frame_number': self.frame_count,
            'timestamp': time.time()
        }
        
        object_type = self.human_validator.classify_object(detection_data)
        return object_type == ObjectType.HUMAN

def main():
    """Função principal"""
    print("🛡️ INTERFACE COMPLETA - DETECÇÃO DE EPIs COM SISTEMA ANTI-VEÍCULO + FULLSCREEN")
    print("=" * 90)
    print("🎯 FUNCIONALIDADES:")
    print("   ✅ Sistema anti-veículo integrado (resolve problema da moto)")
    print("   ✅ Diferenciação inteligente: Pessoas vs Motos/Veículos")
    print("   ✅ Modo Fullscreen para vídeos e câmera (F11)")
    print("   ✅ Validação em camadas baseada na confiança")
    print("   ✅ Configurações otimizadas para canteiro de obra")
    print("   ✅ Logging detalhado das decisões")
    print("=" * 90)
    print("📋 CONTROLES:")
    print("   📸 Imagens: Processamento com filtro anti-veículo")
    print("   🎥 Vídeos: Reprodução em tempo real com validação")
    print("   📹 Câmera: Monitoramento ao vivo otimizado")
    print("   🖥️ Fullscreen: F11 para tela cheia, ESC para sair")
    print("   📸 Screenshot: S para salvar imagem atual")
    print("=" * 90)
    print("⚙️ CONFIGURAÇÕES RECOMENDADAS:")
    print("   🎯 Confidence: 0.15 (baixo para capturar pessoas distantes)")
    print("   🔗 IoU: 0.25 (baixo para permitir sobreposições)")
    print("   🏍️ Anti-Moto: Ativo (ratio 0.6-1.4 = rejeitado)")
    print("=" * 90)
    
    # Cria e executa interface
    interface = EPIDetectionInterface()
    interface.run()

if __name__ == "__main__":
    main()