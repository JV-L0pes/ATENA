"""
Utilitários do Backend Athena
Funções auxiliares e helpers
"""

import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np

def setup_logging(level: str = "INFO", format_str: str = None) -> logging.Logger:
    """Configura logging padrão"""
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("athena_backend.log")
        ]
    )
    
    return logging.getLogger(__name__)

def format_timestamp(timestamp: float = None) -> str:
    """Formata timestamp para string legível"""
    if timestamp is None:
        timestamp = time.time()
    
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

def format_uptime(seconds: int) -> str:
    """Formata uptime em segundos para string legível"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes}m {seconds % 60}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"

def format_percentage(value: float) -> str:
    """Formata valor decimal como porcentagem"""
    return f"{value * 100:.1f}%"

def format_number(value: float, decimals: int = 2) -> str:
    """Formata número com casas decimais"""
    return f"{value:.{decimals}f}"

def calculate_iou(box1: Dict[str, int], box2: Dict[str, int]) -> float:
    """Calcula Intersection over Union entre duas caixas"""
    try:
        # Coordenadas das caixas
        x1_1, y1_1 = box1["x"], box1["y"]
        x2_1, y2_1 = box1["x"] + box1["w"], box1["y"] + box1["h"]
        
        x1_2, y1_2 = box2["x"], box2["y"]
        x2_2, y2_2 = box2["x"] + box2["w"], box2["y"] + box2["h"]
        
        # Calcular interseção
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calcular união
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    except Exception:
        return 0.0

def validate_detection_data(data: Dict[str, Any]) -> bool:
    """Valida dados de detecção"""
    required_fields = ["frame_id", "boxes"]
    
    if not isinstance(data, dict):
        return False
    
    for field in required_fields:
        if field not in data:
            return False
    
    if not isinstance(data["frame_id"], int):
        return False
    
    if not isinstance(data["boxes"], list):
        return False
    
    # Validar cada box
    for box in data["boxes"]:
        if not isinstance(box, dict):
            return False
        
        box_fields = ["x", "y", "w", "h", "label", "conf"]
        for field in box_fields:
            if field not in box:
                return False
        
        # Validar tipos
        if not all(isinstance(box[field], (int, float)) for field in ["x", "y", "w", "h"]):
            return False
        
        if not isinstance(box["label"], str):
            return False
        
        if not isinstance(box["conf"], (int, float)) or not 0 <= box["conf"] <= 1:
            return False
    
    return True

def resize_frame(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Redimensiona frame mantendo proporção"""
    if frame is None:
        return None
    
    h, w = frame.shape[:2]
    
    # Calcular proporção
    aspect_ratio = w / h
    target_ratio = target_width / target_height
    
    if aspect_ratio > target_ratio:
        # Frame é mais largo
        new_w = target_width
        new_h = int(target_width / aspect_ratio)
    else:
        # Frame é mais alto
        new_h = target_height
        new_w = int(target_height * aspect_ratio)
    
    # Redimensionar
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Criar frame com padding se necessário
    if new_w != target_width or new_h != target_height:
        padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Centralizar frame redimensionado
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return padded
    
    return resized

def encode_frame_jpeg(frame: np.ndarray, quality: int = 95) -> bytes:
    """Codifica frame para JPEG"""
    if frame is None:
        return b""
    
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    return buffer.tobytes()

def save_frame_as_image(frame: np.ndarray, filepath: Path, quality: int = 95) -> bool:
    """Salva frame como imagem"""
    try:
        if frame is None:
            return False
        
        # Criar diretório se não existir
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Salvar imagem
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success = cv2.imwrite(str(filepath), frame, encode_params)
        
        return success
        
    except Exception as e:
        logging.error(f"Erro ao salvar frame: {e}")
        return False

def load_image_as_frame(filepath: Path) -> Optional[np.ndarray]:
    """Carrega imagem como frame"""
    try:
        if not filepath.exists():
            return None
        
        frame = cv2.imread(str(filepath))
        return frame
        
    except Exception as e:
        logging.error(f"Erro ao carregar imagem: {e}")
        return None

def debounce(func, delay: float):
    """Decorator para debounce de função"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(wrapper, '_timer'):
            wrapper._timer.cancel()
        
        import threading
        wrapper._timer = threading.Timer(delay, func, args, kwargs)
        wrapper._timer.start()
    
    return wrapper

def throttle(func, delay: float):
    """Decorator para throttle de função"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, '_last_call'):
            wrapper._last_call = 0
        
        current_time = time.time()
        if current_time - wrapper._last_call >= delay:
            wrapper._last_call = current_time
            return func(*args, **kwargs)
    
    return wrapper

def safe_json_dumps(obj: Any) -> str:
    """Serializa objeto para JSON de forma segura"""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return "{}"

def safe_json_loads(json_str: str) -> Any:
    """Deserializa JSON de forma segura"""
    try:
        return json.loads(json_str)
    except Exception:
        return {}

def get_system_info() -> Dict[str, Any]:
    """Obtém informações do sistema"""
    import platform
    import psutil
    
    try:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent
        }
    except ImportError:
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": "N/A",
            "memory_total": "N/A",
            "memory_available": "N/A",
            "disk_usage": "N/A"
        }
