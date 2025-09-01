#!/usr/bin/env python3
"""
Sistema de Snapshot Automático para Violações de EPI
Monitora quando funcionários removem EPIs e tira snapshots após 3 segundos
"""

import cv2
import numpy as np
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import requests
import os

class EPISnapshotSystem:
    """
    Sistema de snapshot automático para violações de EPI
    Monitora violações e tira fotos após período de paciência
    """
    
    def __init__(self, config_path: str = "config/epi_snapshot_config.yaml"):
        """Inicializa o sistema de snapshot"""
        # Configura logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('epi_snapshot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Carrega configurações
        self.config = self._load_config(config_path)
        
        # Estado do sistema
        self.is_monitoring = False
        self.violation_timers = {}  # {person_id: {'start_time': timestamp, 'epi_type': 'helmet/vest'}}
        self.patience_period = self.config.get('patience_period', 3.0)  # segundos
        
        # Diretórios
        self.snapshot_dir = Path(self.config.get('snapshot_dir', 'snapshots'))
        self.snapshot_dir.mkdir(exist_ok=True)
        
        # Histórico de snapshots
        self.snapshot_history = []
        
        # Thread para processamento de snapshots
        self.snapshot_thread = None
        self.stop_event = threading.Event()
        
        # Lock para proteger acesso ao dicionário de violações
        self.violation_lock = threading.Lock()
        
        self.logger.info("Sistema de Snapshot de EPI inicializado")
    
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações do arquivo YAML"""
        try:
            with open(config_path, 'r') as f:
                import yaml
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Configuração padrão se arquivo não existir
            return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Erro ao carregar configurações: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Retorna configuração padrão"""
        return {
            'patience_period': 3.0,
            'snapshot_dir': 'snapshots',
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'recipients': []
            },
            'whatsapp': {
                'enabled': False,
                'api_key': '',
                'phone_numbers': []
            },
            'notification': {
                'include_timestamp': True,
                'include_violation_details': True,
                'image_quality': 95
            }
        }
    
    def start_monitoring(self) -> bool:
        """Inicia o monitoramento de violações"""
        if self.is_monitoring:
            self.logger.warning("Sistema já está monitorando")
            return True
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Inicia thread de processamento
        self.snapshot_thread = threading.Thread(target=self._snapshot_processor)
        self.snapshot_thread.daemon = True
        self.snapshot_thread.start()
        
        self.logger.info("Monitoramento de violações iniciado")
        return True
    
    def stop_monitoring(self) -> None:
        """Para o monitoramento"""
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.snapshot_thread and self.snapshot_thread.is_alive():
            self.snapshot_thread.join(timeout=5)
        
        self.logger.info("Monitoramento de violações parado")
    
    def process_violations(self, validated_people: List, frame: np.ndarray) -> None:
        """
        Processa violações detectadas pela IA e inicia timers
        Deve ser chamado a cada frame de detecção
        """
        if not self.is_monitoring:
            return
        
        current_time = time.time()
        
        for person_data in validated_people:
            person_id = id(person_data['person'])
            person_bbox = person_data['bbox']
            
            # Verifica violações de capacete detectadas pela IA
            if person_data['helmet_violation_detected']:
                violation_bbox = person_data.get('helmet_violation_bbox', person_bbox)
                self._handle_violation(person_id, 'helmet', current_time, frame, violation_bbox)
            else:
                self._clear_violation(person_id, 'helmet')
            
            # Verifica violações de colete detectadas pela IA
            if person_data['vest_violation_detected']:
                violation_bbox = person_data.get('vest_violation_bbox', person_bbox)
                self._handle_violation(person_id, 'vest', current_time, frame, violation_bbox)
            else:
                self._clear_violation(person_id, 'vest')
    
    def _handle_violation(self, person_id: int, epi_type: str, current_time: float, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Gerencia uma violação de EPI"""
        violation_key = f"{person_id}_{epi_type}"
        
        with self.violation_lock:
            if violation_key not in self.violation_timers:
                # Nova violação - inicia timer
                self.violation_timers[violation_key] = {
                    'start_time': current_time,
                    'epi_type': epi_type,
                    'person_id': person_id,
                    'frame': frame.copy(),
                    'bbox': bbox
                }
                self.logger.info(f"Nova violação de {epi_type} detectada para pessoa {person_id}")
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
                    self.logger.info(f"EPI {epi_type} recolocado para pessoa {person_id} após {elapsed_time:.1f}s")
                    del self.violation_timers[violation_key]
                except KeyError:
                    # Violação já foi removida por outro processo
                    self.logger.debug(f"Violação {violation_key} já foi removida")
                except Exception as e:
                    self.logger.error(f"Erro ao limpar violação {violation_key}: {e}")
                    # Tenta remover mesmo com erro
                    if violation_key in self.violation_timers:
                        del self.violation_timers[violation_key]
    
    def _take_snapshot(self, violation_key: str, frame: np.ndarray) -> None:
        """Tira snapshot da violação com bounding box vermelho"""
        try:
            # Gera nome do arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            person_id, epi_type = violation_key.split('_')
            
            filename = f"epi_violation_{epi_type}_{person_id}_{timestamp}.jpg"
            filepath = self.snapshot_dir / filename
            
            # Obtém informações da violação
            violation_info = self.violation_timers.get(violation_key, {})
            person_bbox = violation_info.get('bbox', None)
            
            # Cria imagem anotada com bounding box vermelho
            annotated_frame = self._annotate_violation_image(frame, epi_type, person_id, person_bbox)
            
            # Salva imagem anotada
            cv2.imwrite(str(filepath), annotated_frame)
            
            # Registra no histórico
            snapshot_info = {
                'timestamp': datetime.now().isoformat(),
                'filename': filename,
                'filepath': str(filepath),
                'epi_type': epi_type,
                'person_id': person_id,
                'violation_duration': self.patience_period,
                'bbox': person_bbox
            }
            
            self.snapshot_history.append(snapshot_info)
            
            # Envia notificações
            self._send_notifications(snapshot_info, filepath)
            
            self.logger.info(f"Snapshot salvo: {filename}")
            
        except Exception as e:
            self.logger.error(f"Erro ao tirar snapshot: {e}")
    
    def _annotate_violation_image(self, frame: np.ndarray, epi_type: str, person_id: int, bbox: Tuple[int, int, int, int] = None) -> np.ndarray:
        """Adiciona anotações na imagem de violação com bounding box vermelho específico"""
        annotated = frame.copy()
        
        # Adiciona apenas timestamp (sem texto de violação)
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        # Configurações de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 0, 255)  # Vermelho
        
        # Adiciona timestamp no canto superior esquerdo
        cv2.putText(annotated, timestamp, (20, 30), font, font_scale, (255, 255, 255), 2)
        
        # Adiciona bounding box vermelho na área específica da violação se disponível
        if bbox:
            x1, y1, x2, y2 = bbox
            # Bounding box vermelho grosso para destacar a violação específica
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
            
            # Adiciona label simples acima do bounding box
            label_text = f"FALTANDO: {epi_type.upper()}"
            label_size = cv2.getTextSize(label_text, font, 0.7, 2)[0]
            label_x = x1
            label_y = y1 - 10 if y1 > 30 else y1 + 30
            
            # Fundo preto para o label
            cv2.rectangle(annotated, 
                         (label_x, label_y - label_size[1] - 5),
                         (label_x + label_size[0] + 10, label_y + 5),
                         (0, 0, 0), -1)
            
            # Texto do label em branco
            cv2.putText(annotated, label_text, (label_x + 5, label_y), 
                        font, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def _send_notifications(self, snapshot_info: Dict, image_path: Path) -> None:
        """Envia notificações por email e WhatsApp"""
        try:
            # Email
            if self.config['email']['enabled']:
                self._send_email_notification(snapshot_info, image_path)
            
            # WhatsApp
            if self.config['whatsapp']['enabled']:
                self._send_whatsapp_notification(snapshot_info, image_path)
                
        except Exception as e:
            self.logger.error(f"Erro ao enviar notificações: {e}")
    
    def _send_email_notification(self, snapshot_info: Dict, image_path: Path) -> None:
        """Envia notificação por email"""
        try:
            email_config = self.config['email']
            
            # Cria mensagem
            msg = MIMEMultipart()
            msg['Subject'] = f"ALERTA: Violação de EPI - {snapshot_info['epi_type'].upper()}"
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            
            # Corpo da mensagem
            body = f"""
            ALERTA DE VIOLACAO DE EPI
            
            Tipo de EPI: {snapshot_info['epi_type'].upper()}
            Pessoa ID: {snapshot_info['person_id']}
            Timestamp: {snapshot_info['timestamp']}
            Duração da violação: {snapshot_info['violation_duration']} segundos
            
            Este snapshot foi tirado automaticamente após o período de paciência.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Anexa imagem
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=image_path.name)
                msg.attach(img)
            
            # Envia email
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls()
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)
            
            self.logger.info(f"Email de notificação enviado para {email_config['recipients']}")
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar email: {e}")
    
    def _send_whatsapp_notification(self, snapshot_info: Dict, image_path: Path) -> None:
        """Envia notificação por WhatsApp (usando API externa)"""
        try:
            whatsapp_config = self.config['whatsapp']
            
            # Mensagem
            message = f"""
🚨 ALERTA DE VIOLACAO DE EPI 🚨

Tipo: {snapshot_info['epi_type'].upper()}
Pessoa: {snapshot_info['person_id']}
Horário: {snapshot_info['timestamp']}
Duração: {snapshot_info['violation_duration']}s

Snapshot tirado automaticamente após período de paciência.
            """.strip()
            
            # Para cada número de telefone
            for phone in whatsapp_config['phone_numbers']:
                # Aqui você pode integrar com APIs como:
                # - WhatsApp Business API
                # - Twilio
                # - Outras APIs de mensageria
                
                # Exemplo básico (requer implementação específica)
                self._send_whatsapp_message(phone, message, image_path)
            
            self.logger.info(f"Notificações WhatsApp enviadas para {len(whatsapp_config['phone_numbers'])} números")
            
        except Exception as e:
            self.logger.error(f"Erro ao enviar WhatsApp: {e}")
    
    def _send_whatsapp_message(self, phone: str, message: str, image_path: Path) -> None:
        """Implementação básica para envio de WhatsApp (requer API específica)"""
        # Esta é uma implementação de exemplo
        # Para produção, você precisará integrar com uma API real de WhatsApp
        
        self.logger.info(f"Simulando envio de WhatsApp para {phone}")
        self.logger.info(f"Mensagem: {message}")
        self.logger.info(f"Imagem: {image_path}")
        
        # Aqui você implementaria a integração real com a API escolhida
    
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
                                # Remove violação problemática
                                if violation_key in self.violation_timers:
                                    del self.violation_timers[violation_key]
                
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
    
    def clear_snapshot_history(self) -> None:
        """Limpa histórico de snapshots"""
        self.snapshot_history.clear()
        self.logger.info("Histórico de snapshots limpo")
