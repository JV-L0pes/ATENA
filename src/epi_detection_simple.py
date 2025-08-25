#!/usr/bin/env python3
"""
MVP SIMPLES - Detecção de EPIs sem conflitos
Versão funcional para POC comercial
"""

import cv2
import numpy as np
import time
from pathlib import Path

class SimpleEPIDetector:
    """Detector simples de EPIs para MVP"""
    
    def __init__(self, weights_path):
        """Inicializa detector simples"""
        self.weights_path = weights_path
        self.model = None
        self.cap = None
        
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
        
        # Carrega modelo
        self.load_model()
        
        # Estatísticas
        self.frame_count = 0
        self.start_time = time.time()
        
    def load_model(self):
        """Carrega modelo YOLOv5 de forma simples"""
        try:
            print(f"🚀 Carregando modelo: {self.weights_path}")
            
            # Usa torch.hub diretamente para evitar conflitos
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights_path)
            
            # Configurações básicas
            self.model.conf = 0.25
            self.model.iou = 0.45
            
            print("✅ Modelo carregado com sucesso!")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            print("💡 Tentando carregar modelo alternativo...")
            
            # Tenta modelo padrão se o custom falhar
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
                print("✅ Modelo padrão YOLOv5n carregado!")
                return True
            except:
                print("❌ Falha ao carregar qualquer modelo")
                return False
    
    def start_camera(self, camera_id=0):
        """Inicia câmera de forma simples"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"❌ Não foi possível abrir a câmera {camera_id}")
                return False
            
            # Configurações básicas (sem forçar resolução)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ Câmera {camera_id} iniciada!")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao iniciar câmera: {e}")
            return False
    
    def detect_objects(self, frame):
        """Detecção simples de objetos"""
        if self.model is None:
            return []
        
        try:
            # Redimensiona para 640x640 (padrão YOLO)
            frame_resized = cv2.resize(frame, (640, 640))
            
            # Detecção YOLO
            results = self.model(frame_resized)
            
            # Converte coordenadas para frame original
            h, w = frame.shape[:2]
            detections = []
            
            # Acessa resultados corretamente para YOLOv5
            if hasattr(results, 'xyxy') and len(results.xyxy) > 0:
                # Formato YOLOv5: [x1, y1, x2, y2, conf, cls]
                detections_array = results.xyxy[0].cpu().numpy()
                
                for detection in detections_array:
                    x1, y1, x2, y2, conf, cls = detection
                    
                    # Converte para coordenadas do frame original
                    x1 = int(x1 * w / 640)
                    y1 = int(y1 * h / 640)
                    x2 = int(x2 * w / 640)
                    y2 = int(y2 * h / 640)
                    
                    # Verifica se a classe está no range válido
                    if 0 <= int(cls) < len(self.classes):
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(cls),
                            'class_name': self.classes[int(cls)]
                        })
            
            return detections
            
        except Exception as e:
            print(f"❌ Erro na detecção: {e}")
            return []
    
    def validate_epis(self, detections):
        """Validação simples de EPIs"""
        people = [d for d in detections if d['class_name'] == 'person']
        helmets = [d for d in detections if d['class_name'] == 'helmet']
        vests = [d for d in detections if d['class_name'] == 'vest']
        
        validated_people = []
        
        for person in people:
            px1, py1, px2, py2 = person['bbox']
            person_center = ((px1 + px2) // 2, (py1 + py2) // 2)
            
            # Procura capacete próximo da cabeça
            helmet_found = False
            for helmet in helmets:
                hx1, hy1, hx2, hy2 = helmet['bbox']
                helmet_center = ((hx1 + hx2) // 2, (hy1 + hy2) // 2)
                
                # Distância simples
                distance = np.sqrt((person_center[0] - helmet_center[0])**2 + 
                                 (person_center[1] - helmet_center[1])**2)
                
                # Capacete no topo da pessoa
                if distance < 100 and hy2 < py2 * 0.4:
                    helmet_found = True
                    break
            
            # Procura colete próximo do torso
            vest_found = False
            for vest in vests:
                vx1, vy1, vx2, vy2 = vest['bbox']
                vest_center = ((vx1 + vx2) // 2, (vy1 + vy2) // 2)
                
                # Distância simples
                distance = np.sqrt((person_center[0] - vest_center[0])**2 + 
                                 (person_center[1] - vest_center[1])**2)
                
                # Colete no meio da pessoa
                if distance < 120 and vy1 > py1 * 0.3 and vy2 < py2 * 0.8:
                    vest_found = True
                    break
            
            # Status da pessoa
            if helmet_found and vest_found:
                status = "COMPLIANT"
                color = (0, 255, 0)  # Verde
            elif helmet_found or vest_found:
                status = "PARTIAL"
                color = (0, 255, 255)  # Amarelo
            else:
                status = "VIOLATION"
                color = (0, 0, 255)  # Vermelho
            
            validated_people.append({
                'person': person,
                'helmet_found': helmet_found,
                'vest_found': vest_found,
                'status': status,
                'color': color
            })
        
        return validated_people
    
    def draw_results(self, frame, detections, validated_people):
        """Desenha resultados na tela"""
        frame_copy = frame.copy()
        
        # Desenha todas as detecções
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            conf = detection['confidence']
            
            # Cor baseada na classe
            color = self.colors.get(class_name, (128, 128, 128))
            
            # Desenha bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Desenha status das pessoas validadas
        for person_data in validated_people:
            person = person_data['person']
            px1, py1, px2, py2 = person['bbox']
            status = person_data['status']
            color = person_data['color']
            
            # Status da pessoa
            cv2.putText(frame_copy, f"PESSOA: {status}", (px1, py2 + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Adiciona informações na tela
        self.add_info(frame_copy)
        
        return frame_copy
    
    def add_info(self, frame):
        """Adiciona informações na tela"""
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        info_text = [
            f"MVP SIMPLES - EPI Detection",
            f"FPS: {fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Tempo: {elapsed_time:.1f}s"
        ]
        
        # Desenha informações
        for i, text in enumerate(info_text):
            color = (255, 255, 255) if i == 0 else (0, 255, 255)
            thickness = 2 if i == 0 else 1
            cv2.putText(frame, text, (10, 30 + i * 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    
    def run_detection(self):
        """Executa detecção simples"""
        if not self.start_camera():
            return
        
        print("🚀 MVP SIMPLES - DETECÇÃO INICIADA!")
        print("📋 CONTROLES:")
        print("   - Pressione 'q' para sair")
        print("   - Pressione 's' para salvar screenshot")
        print("🎯 LÓGICA SIMPLES:")
        print("   - Detecta pessoa primeiro")
        print("   - Procura EPIs próximos")
        print("   - Status: COMPLIANT/PARTIAL/VIOLATION")
        print("=" * 60)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Erro ao ler frame da câmera")
                    break
                
                # Detecta objetos
                detections = self.detect_objects(frame)
                
                # Valida EPIs
                validated_people = self.validate_epis(detections)
                
                # Desenha resultados
                result_frame = self.draw_results(frame, detections, validated_people)
                
                # Mostra frame
                cv2.imshow('MVP Simples - EPI Detection', result_frame)
                
                # Processa teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("🛑 Detecção finalizada pelo usuário")
                    break
                elif key == ord('s'):
                    self.save_screenshot(result_frame)
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("🛑 Detecção interrompida pelo usuário")
        except Exception as e:
            print(f"❌ Erro na detecção: {e}")
        finally:
            self.cleanup()
    
    def save_screenshot(self, frame):
        """Salva screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"mvp_simple_screenshot_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot salvo: {filename}")
    
    def cleanup(self):
        """Limpa recursos"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("🧹 Recursos liberados!")

def main():
    print("🚀 MVP SIMPLES - DETECÇÃO DE EPIs FUNCIONAL")
    print("=" * 60)
    print("💡 Lógica simples: pessoa primeiro, depois EPIs")
    print("🎯 Sem conflitos de dependências")
    print("=" * 60)
    
    # Caminho para o modelo
    weights_path = "yolov5/runs/train/epi_detection_v24/weights/best.pt"
    
    if not Path(weights_path).exists():
        print(f"❌ Modelo não encontrado: {weights_path}")
        print("💡 Execute primeiro o fine-tuning!")
        return
    
    # Cria detector simples
    detector = SimpleEPIDetector(weights_path)
    
    # Executa detecção
    detector.run_detection()

if __name__ == "__main__":
    main()
