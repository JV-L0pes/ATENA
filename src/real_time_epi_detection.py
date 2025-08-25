#!/usr/bin/env python3
"""
Sistema de Detecção de EPIs em Tempo Real para Canteiros de Obra
Integra câmera otimizada com detector de EPIs para monitoramento contínuo
"""

import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

from epi_detector_v2 import EPIDetectorV2
from construction_camera_system import ConstructionCameraSystem

class RealTimeEPIDetection:
    """
    Sistema principal de detecção de EPIs em tempo real
    Monitora continuamente o canteiro de obra
    """
    
    def __init__(self, config_path: str = "config/epi_config_v2.yaml"):
        """Inicializa o sistema de detecção em tempo real"""
        # Configura logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('real_time_epi.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Inicializa componentes
        self.epi_detector = EPIDetectorV2(config_path)
        self.camera_system = ConstructionCameraSystem()
        
        # Configurações do sistema
        self.is_running = False
        self.frame_count = 0
        self.start_time = time.time()
        
        # Estatísticas de detecção
        self.detection_stats = {
            'total_frames': 0,
            'people_detected': 0,
            'helmet_violations': 0,
            'vest_violations': 0,
            'compliant_people': 0,
            'partial_compliance': 0
        }
        
        # Configurações de alerta
        self.alert_settings = {
            'enable_alerts': True,
            'violation_threshold': 3,  # Alertas após N violações consecutivas
            'alert_cooldown': 30,      # Segundos entre alertas
            'last_alert_time': 0
        }
        
        # Histórico de detecções
        self.detection_history = []
        
        self.logger.info("Sistema de Detecção de EPIs em Tempo Real inicializado")
    
    def start_monitoring(self) -> bool:
        """Inicia o monitoramento em tempo real"""
        try:
            # Inicia câmera
            if not self.camera_system.start_camera():
                self.logger.error("Falha ao iniciar câmera")
                return False
            
            # Otimiza câmera para detecção de EPIs
            self.camera_system.optimize_for_epi_detection()
            
            # Mostra informações da câmera
            camera_info = self.camera_system.get_camera_info()
            self.logger.info(f"Informações da câmera: {camera_info}")
            
            self.is_running = True
            self.logger.info("Monitoramento iniciado com sucesso")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao iniciar monitoramento: {e}")
            return False
    
    def run_detection_loop(self) -> None:
        """Executa o loop principal de detecção"""
        if not self.is_running:
            self.logger.error("Sistema não está rodando")
            return
        
        print("\n🚀 SISTEMA DE DETECÇÃO DE EPIs EM TEMPO REAL")
        print("=" * 60)
        print("Controles:")
        print("  Q: Sair")
        print("  S: Salvar screenshot")
        print("  C: Ajustar configurações de câmera")
        print("  A: Alternar ambiente")
        print("  O: Otimizar para EPIs")
        print("  I: Mostrar informações")
        print("=" * 60)
        
        try:
            while self.is_running:
                # Captura frame
                frame = self.camera_system.get_frame()
                if frame is None:
                    continue
                
                # Processa frame para detecção de EPIs
                processed_frame, people = self.epi_detector.process_image(frame)
                
                # Atualiza estatísticas
                self._update_statistics(people)
                
                # Verifica violações e gera alertas
                self._check_violations(people)
                
                # Adiciona informações na tela
                info_frame = self._add_system_info(processed_frame)
                
                # Exibe frame
                cv2.imshow('Detecção de EPIs em Tempo Real - Canteiro de Obra', info_frame)
                
                # Processa teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_screenshot(info_frame)
                elif key == ord('c'):
                    self.camera_system.adjust_settings_interactive()
                elif key == ord('a'):
                    self._cycle_environment()
                elif key == ord('o'):
                    self.camera_system.optimize_for_epi_detection()
                elif key == ord('i'):
                    self._show_system_info()
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            self.logger.info("Monitoramento interrompido pelo usuário")
        except Exception as e:
            self.logger.error(f"Erro no loop de detecção: {e}")
        finally:
            self.stop_monitoring()
    
    def _update_statistics(self, people: list) -> None:
        """Atualiza estatísticas de detecção"""
        self.detection_stats['total_frames'] += 1
        self.detection_stats['people_detected'] = len(people)
        
        helmet_violations = 0
        vest_violations = 0
        compliant = 0
        partial = 0
        
        for person in people:
            if person.helmet_status.value == "AUSENTE":
                helmet_violations += 1
            if person.vest_status.value == "AUSENTE":
                vest_violations += 1
            
            if person.helmet_status.value == "CORRETO" and person.vest_status.value == "CORRETO":
                compliant += 1
            elif person.helmet_status.value == "CORRETO" or person.vest_status.value == "CORRETO":
                partial += 1
        
        self.detection_stats['helmet_violations'] = helmet_violations
        self.detection_stats['vest_violations'] = vest_violations
        self.detection_stats['compliant_people'] = compliant
        self.detection_stats['partial_compliance'] = partial
        
        # Adiciona ao histórico
        if people:
            self.detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'frame': self.frame_count,
                'people_count': len(people),
                'violations': helmet_violations + vest_violations,
                'compliant': compliant
            })
            
            # Mantém apenas os últimos 1000 registros
            if len(self.detection_history) > 1000:
                self.detection_history.pop(0)
    
    def _check_violations(self, people: list) -> None:
        """Verifica violações e gera alertas"""
        if not self.alert_settings['enable_alerts']:
            return
        
        current_time = time.time()
        total_violations = self.detection_stats['helmet_violations'] + self.detection_stats['vest_violations']
        
        # Verifica se deve gerar alerta
        if (total_violations >= self.alert_settings['violation_threshold'] and 
            current_time - self.alert_settings['last_alert_time'] > self.alert_settings['alert_cooldown']):
            
            self._generate_violation_alert(people)
            self.alert_settings['last_alert_time'] = current_time
    
    def _generate_violation_alert(self, people: list) -> None:
        """Gera alerta de violação"""
        alert_msg = f"🚨 ALERTA DE VIOLAÇÃO DE EPIs - {datetime.now().strftime('%H:%M:%S')}"
        print(f"\n{alert_msg}")
        print("=" * len(alert_msg))
        
        for i, person in enumerate(people):
            status = []
            if person.helmet_status.value == "AUSENTE":
                status.append("SEM CAPACETE")
            if person.vest_status.value == "AUSENTE":
                status.append("SEM COLETE")
            
            if status:
                print(f"Pessoa {i+1}: {' | '.join(status)}")
        
        print("=" * len(alert_msg))
        
        # Salva alerta em arquivo
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'frame': self.frame_count,
            'people_count': len(people),
            'violations': self.detection_stats['helmet_violations'] + self.detection_stats['vest_violations'],
            'details': [
                {
                    'person_id': i,
                    'helmet_status': person.helmet_status.value,
                    'vest_status': person.vest_status.value
                }
                for i, person in enumerate(people)
            ]
        }
        
        alert_file = f"alerts/epi_violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(alert_file).parent.mkdir(exist_ok=True)
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        self.logger.warning(f"Alerta de violação salvo: {alert_file}")
    
    def _add_system_info(self, frame: np.ndarray) -> np.ndarray:
        """Adiciona informações do sistema na tela"""
        info_frame = frame.copy()
        
        # Calcula FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Informações básicas
        info_text = [
            f"EPI Detection System - Canteiro de Obra",
            f"FPS: {fps:.1f} | Frame: {self.frame_count}",
            f"Pessoas: {self.detection_stats['people_detected']}",
            f"Compliance: {self.detection_stats['compliant_people']}",
            f"Parcial: {self.detection_stats['partial_compliance']}",
            f"Violacoes: {self.detection_stats['helmet_violations'] + self.detection_stats['vest_violations']}",
            f"Ambiente: {self.camera_system.current_environment}",
            "",
            "Q: Sair | S: Screenshot | C: Camera | A: Ambiente | O: Otimizar"
        ]
        
        # Desenha informações
        y_offset = 30
        for i, text in enumerate(info_text):
            if i == 0:  # Título
                color = (255, 255, 255)
                thickness = 2
            elif i < 7:  # Métricas
                color = (0, 255, 255)
                thickness = 1
            elif i == 7:  # Linha vazia
                continue
            else:  # Controles
                color = (0, 255, 0)
                thickness = 1
            
            cv2.putText(info_frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        
        return info_frame
    
    def _save_screenshot(self, frame: np.ndarray) -> None:
        """Salva screenshot da detecção"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshots/epi_detection_{timestamp}.jpg"
        
        Path(filename).parent.mkdir(exist_ok=True)
        cv2.imwrite(filename, frame)
        
        print(f"📸 Screenshot salvo: {filename}")
        self.logger.info(f"Screenshot salvo: {filename}")
    
    def _cycle_environment(self) -> None:
        """Alterna entre ambientes disponíveis"""
        environments = list(self.camera_system.environment_settings.keys())
        current_idx = environments.index(self.camera_system.current_environment)
        next_idx = (current_idx + 1) % len(environments)
        next_env = environments[next_idx]
        
        if self.camera_system.set_environment(next_env):
            print(f"🌍 Ambiente alterado para: {next_env}")
    
    def _show_system_info(self) -> None:
        """Mostra informações detalhadas do sistema"""
        print("\n📊 INFORMAÇÕES DO SISTEMA")
        print("=" * 50)
        
        # Informações da câmera
        camera_info = self.camera_system.get_camera_info()
        print("📹 CÂMERA:")
        for key, value in camera_info.items():
            print(f"  {key}: {value}")
        
        # Estatísticas de detecção
        print(f"\n📈 ESTATÍSTICAS:")
        print(f"  Frames processados: {self.detection_stats['total_frames']}")
        print(f"  Pessoas detectadas: {self.detection_stats['people_detected']}")
        print(f"  Compliance total: {self.detection_stats['compliant_people']}")
        print(f"  Compliance parcial: {self.detection_stats['partial_compliance']}")
        print(f"  Violações de capacete: {self.detection_stats['helmet_violations']}")
        print(f"  Violações de colete: {self.detection_stats['vest_violations']}")
        
        # Performance
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        print(f"\n⚡ PERFORMANCE:")
        print(f"  FPS médio: {fps:.1f}")
        print(f"  Tempo de execução: {elapsed_time:.1f}s")
        
        print("=" * 50)
    
    def stop_monitoring(self) -> None:
        """Para o monitoramento"""
        self.is_running = False
        self.camera_system.release()
        cv2.destroyAllWindows()
        
        # Salva estatísticas finais
        self._save_final_statistics()
        
        self.logger.info("Monitoramento parado")
    
    def _save_final_statistics(self) -> None:
        """Salva estatísticas finais da sessão"""
        final_stats = {
            'session_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.now().isoformat(),
                'total_frames': self.frame_count,
                'total_time': time.time() - self.start_time
            },
            'detection_summary': self.detection_stats,
            'detection_history': self.detection_history
        }
        
        stats_file = f"statistics/epi_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        Path(stats_file).parent.mkdir(exist_ok=True)
        
        with open(stats_file, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        self.logger.info(f"Estatísticas finais salvas: {stats_file}")
    
    def run(self) -> None:
        """Executa o sistema completo"""
        try:
            if self.start_monitoring():
                self.run_detection_loop()
            else:
                self.logger.error("Falha ao iniciar monitoramento")
        except Exception as e:
            self.logger.error(f"Erro no sistema: {e}")
        finally:
            self.stop_monitoring()

def main():
    """Função principal"""
    print("🚧 SISTEMA DE DETECÇÃO AUTOMÁTICA DE EPIs EM CANTEIROS DE OBRA")
    print("=" * 70)
    print("🎯 OBJETIVO: Detectar e validar EPIs fisicamente na pessoa")
    print("📋 REGRAS:")
    print("   ✅ Capacete deve estar na cabeça (não na mão, no chão)")
    print("   ✅ Colete deve estar no torso (não pendurado, dobrado)")
    print("   ❌ Infração: EPI presente mas não na pessoa")
    print("=" * 70)
    
    # Cria diretórios necessários
    Path("screenshots").mkdir(exist_ok=True)
    Path("alerts").mkdir(exist_ok=True)
    Path("statistics").mkdir(exist_ok=True)
    
    # Inicia sistema
    system = RealTimeEPIDetection()
    system.run()

if __name__ == "__main__":
    main()
