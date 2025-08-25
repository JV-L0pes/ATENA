#!/usr/bin/env python3
"""
INTERFACE COMPLETA - Detecção de EPIs em Imagens, Vídeos e Câmera
Versão funcional para POC comercial com múltiplas fontes
"""

import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
import queue
from PIL import Image, ImageTk

class EPIDetectionInterface:
    """Interface completa para detecção de EPIs"""
    
    def __init__(self):
        """Inicializa interface"""
        self.model = None
        self.cap = None
        self.is_camera_running = False
        self.frame_queue = queue.Queue(maxsize=10)
        
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
        
        # Cria interface
        self.create_interface()
        
    def load_model(self):
        """Carrega modelo YOLOv5 otimizado"""
        try:
            # Modelo superior: epi_safe_fine_tuned
            weights_path = "yolov5/runs/train/epi_safe_fine_tuned/weights/best.pt"
            
            if not Path(weights_path).exists():
                print(f"❌ Modelo não encontrado: {weights_path}")
                return False
            
            print(f"🚀 Carregando modelo SUPERIOR: {weights_path}")
            print("🎯 Fine-tuned a partir de epi_detection_balanced_long_distance4")
            print("📏 Otimizado para longa distância (30m)")
            print("⚡ Treinamento eficiente: 16 epochs")
            
            # Usa torch.hub diretamente
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
            
            # Configurações ULTRA sensíveis para detectar pessoas
            self.model.conf = 0.20
            self.model.iou = 0.35
            
            print("✅ Modelo SUPERIOR carregado com sucesso!")
            print("🏆 mAP@0.5: 0.91 | Precision: 0.91 | Recall: 0.82")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return False
    
    def create_interface(self):
        """Cria interface gráfica"""
        self.root = tk.Tk()
        self.root.title("EPI Detection Interface - MVP")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Estilo
        style = ttk.Style()
        style.theme_use('clam')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
                # Título
        title_label = tk.Label(main_frame, text="🏆 DETECÇÃO DE EPIs - MODELO SUPERIOR", 
                               font=('Arial', 20, 'bold'), fg='white', bg='#2b2b2b')
        title_label.pack(pady=(0, 20))
        
        # Subtítulo com informações do modelo
        subtitle_label = tk.Label(main_frame, text="Fine-tuned para Longa Distância (30m) | mAP@0.5: 0.91", 
                                 font=('Arial', 12), fg='#4CAF50', bg='#2b2b2b')
        subtitle_label.pack(pady=(0, 20))
        
        # Frame de controles
        controls_frame = tk.Frame(main_frame, bg='#3b3b3b', relief='raised', bd=2)
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Botões de controle
        btn_frame = tk.Frame(controls_frame, bg='#3b3b3b')
        btn_frame.pack(pady=10)
        
        # Botão Imagem
        self.img_btn = tk.Button(btn_frame, text="📸 DETECTAR IMAGEM", 
                                command=self.detect_image, 
                                font=('Arial', 12, 'bold'),
                                bg='#4CAF50', fg='white', 
                                relief='raised', bd=3, padx=20, pady=10)
        self.img_btn.pack(side=tk.LEFT, padx=10)
        
        # Botão Vídeo
        self.video_btn = tk.Button(btn_frame, text="🎥 DETECTAR VÍDEO", 
                                  command=self.detect_video, 
                                  font=('Arial', 12, 'bold'),
                                  bg='#2196F3', fg='white', 
                                  relief='raised', bd=3, padx=20, pady=10)
        self.video_btn.pack(side=tk.LEFT, padx=10)
        
        # Botão Câmera
        self.camera_btn = tk.Button(btn_frame, text="📹 CÂMERA AO VIVO", 
                                   command=self.toggle_camera, 
                                   font=('Arial', 12, 'bold'),
                                   bg='#FF9800', fg='white', 
                                   relief='raised', bd=3, padx=20, pady=10)
        self.camera_btn.pack(side=tk.LEFT, padx=10)
        
        # Botão Parar
        self.stop_btn = tk.Button(btn_frame, text="⏹️ PARAR", 
                                 command=self.stop_camera, 
                                 font=('Arial', 12, 'bold'),
                                 bg='#f44336', fg='white', 
                                 relief='raised', bd=3, padx=20, pady=10)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Frame de visualização
        view_frame = tk.Frame(main_frame, bg='#1b1b1b', relief='sunken', bd=2)
        view_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas para imagem/vídeo
        self.canvas = tk.Canvas(view_frame, bg='#1b1b1b', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de informações
        info_frame = tk.Frame(main_frame, bg='#3b3b3b', relief='raised', bd=2)
        info_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Labels de informação
        self.status_label = tk.Label(info_frame, text="Status: Aguardando...", 
                                    font=('Arial', 12), fg='white', bg='#3b3b3b')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.fps_label = tk.Label(info_frame, text="FPS: 0", 
                                 font=('Arial', 12), fg='white', bg='#3b3b3b')
        self.fps_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.detection_label = tk.Label(info_frame, text="Detecções: 0", 
                                       font=('Arial', 12), fg='white', bg='#3b3b3b')
        self.detection_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Configurações
        config_frame = tk.Frame(main_frame, bg='#3b3b3b', relief='raised', bd=2)
        config_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Thresholds
        tk.Label(config_frame, text="Confidence:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(10, 5), pady=10)
        
        self.conf_var = tk.DoubleVar(value=0.20)
        self.conf_scale = tk.Scale(config_frame, from_=0.1, to=0.9, resolution=0.05,
                                  orient=tk.HORIZONTAL, variable=self.conf_var,
                                  bg='#3b3b3b', fg='white', highlightthickness=0)
        self.conf_scale.pack(side=tk.LEFT, padx=(0, 20), pady=10)
        
        tk.Label(config_frame, text="IoU:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(0, 5), pady=10)
        
        self.iou_var = tk.DoubleVar(value=0.35)
        self.iou_scale = tk.Scale(config_frame, from_=0.1, to=0.9, resolution=0.05,
                                 orient=tk.HORIZONTAL, variable=self.iou_var,
                                 bg='#3b3b3b', fg='white', highlightthickness=0)
        self.iou_scale.pack(side=tk.LEFT, padx=(0, 20), pady=10)
        
        # Botão aplicar configurações
        self.apply_btn = tk.Button(config_frame, text="✅ APLICAR", 
                                  command=self.apply_config, 
                                  font=('Arial', 10, 'bold'),
                                  bg='#4CAF50', fg='white', 
                                  relief='raised', bd=2, padx=15, pady=5)
        self.apply_btn.pack(side=tk.LEFT, padx=(0, 10), pady=10)
        
        # Controle de otimização de vídeo
        tk.Label(config_frame, text="Vídeo:", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(20, 5), pady=10)
        
        self.video_opt_var = tk.IntVar(value=5)
        self.video_opt_scale = tk.Scale(config_frame, from_=1, to=15, resolution=1,
                                       orient=tk.HORIZONTAL, variable=self.video_opt_var,
                                       bg='#3b3b3b', fg='white', highlightthickness=0)
        self.video_opt_scale.pack(side=tk.LEFT, padx=(0, 20), pady=10)
        
        tk.Label(config_frame, text="frames", font=('Arial', 10), 
                fg='white', bg='#3b3b3b').pack(side=tk.LEFT, padx=(0, 10), pady=10)
        
        # Bind eventos
        self.conf_scale.bind("<ButtonRelease-1>", self.on_scale_change)
        self.iou_scale.bind("<ButtonRelease-1>", self.on_scale_change)
        
        # Inicializa variáveis
        self.current_image = None
        self.current_detections = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # Variáveis de vídeo
        self.is_video_playing = False
        self.video_paused = False
        self.current_frame = 0
        
        # Sistema de tracking temporal para pessoas
        self.person_tracks = {}  # {track_id: {bbox, confidence, frames_seen}}
        self.next_track_id = 0
        
        # Atualiza interface
        self.update_interface()
    
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
            messagebox.showinfo("Configuração", "Configurações aplicadas com sucesso!")
    
    def detect_image(self):
        """Detecta EPIs em imagem"""
        if not self.model:
            messagebox.showerror("Erro", "Modelo não carregado!")
            return
        
        # Abre diálogo de arquivo
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
            
            # Detecta objetos
            detections = self.detect_objects(image)
            
            # Valida EPIs
            validated_people = self.validate_epis(detections)
            
            # Desenha resultados
            result_image = self.draw_results(image, detections, validated_people)
            
            # Converte para Tkinter
            self.display_image(result_image)
            
            # Atualiza status
            self.status_label.config(text=f"Status: Imagem processada - {len(detections)} detecções")
            self.detection_label.config(text=f"Detecções: {len(detections)}")
            
            # Salva resultado
            self.save_result(result_image, "image")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar imagem: {str(e)}")
    
    def detect_video(self):
        """Detecta EPIs em vídeo"""
        if not self.model:
            messagebox.showerror("Erro", "Modelo não carregado!")
            return
        
        # Abre diálogo de arquivo
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
            
            # Processa vídeo
            self.process_video(cap, file_path)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao processar vídeo: {str(e)}")
    
    def process_video(self, cap, file_path):
        """Reproduz vídeo na velocidade nativa com detecção em tempo real"""
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Configurações de reprodução
        self.is_video_playing = True
        self.video_paused = False
        self.current_frame = 0
        self.video_cap = cap  # Guarda referência para controles
        self.video_fps = fps  # Guarda FPS para usar no loop
        
        # Cria janela de vídeo com controles
        self.create_video_window(cap, file_path)
        
        # Inicia reprodução em thread separada
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def video_loop(self):
        """Loop de vídeo em thread separada otimizado para 30 FPS"""
        try:
            # Força 30 FPS para reprodução suave
            target_fps = 30
            frame_delay = 1.0 / target_fps
            
            while self.is_video_playing and self.video_cap and self.video_cap.isOpened():
                if not self.video_paused:
                    ret, frame = self.video_cap.read()
                    if not ret:
                        break
                    
                    # Detecta objetos em tempo real
                    detections = self.detect_objects(frame)
                    validated_people = self.validate_epis(detections)
                    
                    # Desenha resultados com overlay
                    result_frame = self.draw_results(frame, detections, validated_people)
                    
                    # Atualiza interface na thread principal
                    self.root.after(0, self.update_video_frame, result_frame, detections, validated_people)
                    
                    # Controle de velocidade fixa em 30 FPS
                    time.sleep(frame_delay)
                    
                    self.current_frame += 1
                    
                    # Permite cancelar
                    if not self.root.winfo_exists():
                        break
                else:
                    # Pausado - aguarda
                    time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Erro no loop de vídeo: {e}")
            # Corrige erro de escopo na lambda
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao reproduzir vídeo: {error_msg}"))
        finally:
            if self.video_cap:
                self.video_cap.release()
            self.is_video_playing = False
    
    def update_video_frame(self, frame, detections, validated_people):
        """Atualiza frame do vídeo na interface principal"""
        try:
            # Exibe frame com detecções
            self.display_video_frame(frame)
            
            # Atualiza status
            self.update_video_status(frame, detections, validated_people)
        except Exception as e:
            print(f"❌ Erro ao atualizar frame: {e}")
    
    def create_video_window(self, cap, file_path):
        """Cria janela de vídeo com controles simplificados"""
        # Frame de controles de vídeo
        self.video_controls = tk.Frame(self.root, bg='#3b3b3b', relief='raised', bd=2)
        self.video_controls.pack(fill=tk.X, pady=(10, 0))
        
        # Botões de controle simplificados
        self.play_pause_btn = tk.Button(self.video_controls, text="⏸️ PAUSAR", 
                                       command=self.toggle_video_pause, 
                                       font=('Arial', 12, 'bold'),
                                       bg='#FF9800', fg='white', 
                                       relief='raised', bd=2, padx=20, pady=8)
        self.play_pause_btn.pack(side=tk.LEFT, padx=20, pady=8)
        
        # Label de status simples
        self.video_status_label = tk.Label(self.video_controls, text="🎥 Vídeo em reprodução", 
                                          font=('Arial', 12), fg='white', bg='#3b3b3b')
        self.video_status_label.pack(side=tk.LEFT, padx=20, pady=8)
    
    def toggle_video_pause(self):
        """Alterna pausa do vídeo"""
        self.video_paused = not self.video_paused
        if self.video_paused:
            self.play_pause_btn.config(text="▶️ CONTINUAR", bg='#4CAF50')
        else:
            self.play_pause_btn.config(text="⏸️ PAUSAR", bg='#FF9800')
    
    def stop_video(self):
        """Para reprodução do vídeo"""
        self.is_video_playing = False
        self.video_paused = False
        
        # Remove controles de vídeo da interface
        try:
            if hasattr(self, 'video_controls'):
                self.video_controls.destroy()
        except:
            pass
    
    def seek_video(self, frame_num):
        """Pula para frame específico (mantido para compatibilidade)"""
        pass  # Removido - controles simplificados
    
    def display_video_frame(self, frame):
        """Exibe frame do vídeo com detecções"""
        self.display_image(frame)
    
    def update_video_status(self, frame, detections, validated_people):
        """Atualiza status do vídeo simplificado"""
        self.status_label.config(text=f"Status: Reproduzindo vídeo - Frame {self.current_frame}")
        self.detection_label.config(text=f"Detecções: {len(detections)}")
    
    def toggle_camera(self):
        """Alterna câmera ao vivo"""
        if not self.is_camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Inicia câmera ao vivo"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Não foi possível abrir a câmera!")
                return
            
            # Configurações básicas
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_camera_running = True
            self.camera_btn.config(text="📹 PARAR CÂMERA", bg='#f44336')
            self.stop_btn.config(state=tk.NORMAL)
            
            # Inicia thread de câmera
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.status_label.config(text="Status: Câmera ativa")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao iniciar câmera: {str(e)}")
    
    def stop_camera(self):
        """Para câmera ao vivo"""
        self.is_camera_running = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        try:
            if self.root.winfo_exists():
                self.camera_btn.config(text="📹 CÂMERA AO VIVO", bg='#FF9800')
                self.stop_btn.config(state=tk.DISABLED)
                self.status_label.config(text="Status: Câmera parada")
        except:
            pass
    
    def camera_loop(self):
        """Loop principal da câmera"""
        while self.is_camera_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Detecta objetos
                    detections = self.detect_objects(frame)
                    
                    # Valida EPIs
                    validated_people = self.validate_epis(detections)
                    
                    # Desenha resultados
                    result_frame = self.draw_results(frame, detections, validated_people)
                    
                    # Adiciona à fila para exibição
                    if not self.frame_queue.full():
                        self.frame_queue.put(result_frame)
                    
                    # Atualiza estatísticas
                    self.frame_count += 1
                    elapsed_time = time.time() - self.start_time
                    fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Atualiza interface na thread principal
                    self.root.after(0, self.update_camera_stats, fps, len(detections))
            
            time.sleep(0.033)  # ~30 FPS
    
    def update_camera_stats(self, fps, detections):
        """Atualiza estatísticas da câmera na interface"""
        try:
            if self.root.winfo_exists():
                self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.detection_label.config(text=f"Detecções: {detections}")
                
                # Atualiza imagem se houver frame na fila
                try:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get_nowait()
                        self.display_image(frame)
                except queue.Empty:
                    pass
        except:
            pass
    
    def detect_objects(self, frame):
        """Detecção de objetos YOLOv5 com filtros anti-falso positivo"""
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
                        # 🔍 FILTRO ADICIONAL: Validação de confiança por classe
                        if not self.is_valid_detection_confidence(cls, conf):
                            continue
                        
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
        """Validação inteligente de EPIs com tracking temporal"""
        people = [d for d in detections if d['class_name'] == 'person']
        helmets = [d for d in detections if d['class_name'] == 'helmet']
        vests = [d for d in detections if d['class_name'] == 'vest']
        
        validated_people = []
        
        for person in people:
            px1, py1, px2, py2 = person['bbox']
            person_width = px2 - px1
            person_height = py2 - py1
            
            # 🔍 VALIDAÇÃO INTELIGENTE: Combina análise de forma + tracking temporal
            is_human_by_form = self.is_likely_human(person_width, person_height, person)
            
            # Adiciona tracking temporal
            track_id = self.track_person_temporally(person)
            is_human_by_tracking = self.is_tracked_person_likely_human(track_id)
            
            # 🚗 VALIDAÇÃO DE MOVIMENTO: Detecta veículos vs pessoas
            is_human_by_movement = self._validate_human_movement(track_id)
            
            # ✅ DECISÃO FINAL: Todas as validações devem passar para ser humano
            if not (is_human_by_form and is_human_by_tracking and is_human_by_movement):
                print(f"❌ Objeto descartado: não é humano (w:{person_width}, h:{person_height}, track:{track_id})")
                print(f"   - Forma: {is_human_by_form}, Tracking: {is_human_by_tracking}, Movimento: {is_human_by_movement}")
                continue
            
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
            
            validated_people.append({
                'person': person,
                'helmet_found': helmet_found,
                'vest_found': vest_found,
                'track_id': track_id
            })
        
        return validated_people
    

    
    def is_valid_detection_confidence(self, class_id, confidence):
        """Valida se a confiança é adequada para cada classe"""
        # Thresholds ULTRA sensíveis para detectar pessoas em qualquer situação
        confidence_thresholds = {
            0: 0.20,  # helmet - ultra sensível
            1: 0.25,  # no-helmet - ultra sensível
            2: 0.25,  # no-vest - ultra sensível
            3: 0.20,  # person - ultra sensível (detecta TODAS as pessoas)
            4: 0.20   # vest - ultra sensível
        }
        threshold = confidence_thresholds.get(int(class_id), 0.15)
        return confidence >= threshold
    
    def is_likely_human(self, width, height, person_detection):
        """Validação inteligente: diferencia pessoas de objetos usando heurísticas avançadas"""
        if height <= 0 or width <= 0:
            return False
        
        # 🎯 ESTRATÉGIA PRINCIPAL: CONFIANÇA ALTA = MAIS PROVÁVEL SER HUMANO
        confidence = person_detection['confidence']
        
        print(f"🔍 VALIDAÇÃO: {width}x{height}px, conf: {confidence:.2f}")
        
        # ✅ REGRA 1: Se confiança é alta (> 0.5), valida com filtros rigorosos
        if confidence > 0.5:
            print(f"🎯 REGRA 1: Confiança alta ({confidence:.2f}) - validação rigorosa")
            return self._strict_human_validation(width, height, person_detection)
        
        # ✅ REGRA 2: Se confiança é média-alta (> 0.4), valida com filtros médios
        if confidence > 0.4:
            print(f"🎯 REGRA 2: Confiança média-alta ({confidence:.2f}) - validação média")
            return self._medium_human_validation(width, height, person_detection)
        
        # ✅ REGRA 3: Se confiança é média (> 0.25), usa filtros rigorosos
        if confidence > 0.25:  # Ajustado para 0.25
            print(f"🎯 REGRA 3: Confiança média ({confidence:.2f}) - validação rigorosa")
            return self._strict_human_validation(width, height, person_detection)
        
        # ❌ REGRA 4: Confiança muito baixa (< 0.25) = descarta
        print(f"❌ REGRA 4: Confiança muito baixa ({confidence:.2f}) - REJEITADO")
        return False
    
    def _strict_human_validation(self, width, height, person_detection):
        """Validação rigorosa para diferenciar pessoas de motos/veículos"""
        
        print(f"🔍 VALIDAÇÃO RIGOROSA: {width}x{height}px")
        
        # 🚫 FILTROS ANTI-MOTO/VEÍCULO RIGOROSOS:
        
        # 1. Altura mínima para pessoa (1.40m relativo)
        min_height_px = 140
        if height < min_height_px:
            print(f"❌ Objeto muito baixo para ser pessoa: {height}px < {min_height_px}px")
            return False
        
        # 2. Proporções anatômicas humanas RIGOROSAS
        aspect_ratio = height / width
        print(f"📐 Aspect ratio: {aspect_ratio:.2f} (deve ser >= 1.8)")
        
        # Pessoas devem ser MUITO mais altas que largas
        if aspect_ratio < 1.8:  # Mais rigoroso que antes
            print(f"❌ Objeto muito largo para ser pessoa: ratio {aspect_ratio:.2f} < 1.8")
            return False
        
        if aspect_ratio > 6.0:  # Menos rigoroso para postes
            print(f"❌ Objeto muito estreito para ser pessoa: ratio {aspect_ratio:.2f}")
            return False
        
        # 3. Validação de forma corporal RIGOROSA
        print(f"🔍 Verificando forma de veículo...")
        if self._is_vehicle_like_shape(width, height, person_detection):
            print(f"❌ Forma muito similar a veículo")
            return False
        
        # 4. Validação de simetria (motos são mais simétricas)
        print(f"🔍 Verificando simetria...")
        if self._is_too_symmetric(width, height, person_detection):
            print(f"❌ Objeto muito simétrico (possível veículo)")
            return False
        
        # 5. Tamanhos extremos
        if width < 30 or height < 50:
            print(f"❌ Objeto muito pequeno: {width}x{height}")
            return False
        
        if width > 1500 or height > 3000:
            print(f"❌ Objeto muito grande: {width}x{height}")
            return False
        
        print(f"✅ VALIDAÇÃO RIGOROSA PASSOU - provavelmente pessoa")
        return True
    
    def _medium_human_validation(self, width, height, person_detection):
        """Validação média para confiança média-alta"""
        aspect_ratio = height / width
        
        # Mais flexível, mas ainda rigoroso
        if aspect_ratio < 1.6:  # Menos rigoroso que strict
            print(f"❌ Objeto muito largo para ser pessoa: ratio {aspect_ratio:.2f} < 1.6")
            return False
        
        if aspect_ratio > 7.0:
            print(f"❌ Objeto muito estreito para ser pessoa: ratio {aspect_ratio:.2f}")
            return False
        
        # Validação básica de forma
        if self._is_obviously_vehicle(width, height, person_detection):
            print(f"❌ Obviamente veículo")
            return False
        
        return True
    
    def _is_vehicle_like_shape(self, width, height, person_detection):
        """Detecção avançada de formas de veículos"""
        width_height_ratio = width / height
        print(f"🔍 Verificando forma: width/height = {width_height_ratio:.2f}")
        
        # 🏍️ DETECÇÃO ESPECÍFICA DE MOTOS:
        # Motos têm proporções características: largura similar à altura
        if 0.7 <= width_height_ratio <= 1.3:  # Quase quadrado
            print(f"🏍️ FORMA QUASE QUADRADA (ratio {width_height_ratio:.2f}) - POSSÍVEL MOTO!")
            return True
        
        # 🚗 VEÍCULOS LARGOS:
        if width_height_ratio > 1.5:  # Muito mais largo que alto
            print(f"🚗 MUITO LARGO (ratio {width_height_ratio:.2f}) - POSSÍVEL VEÍCULO!")
            return True
        
        # 🔲 FORMAS RETANGULARES PERFEITAS:
        if 0.8 <= width_height_ratio <= 1.2:  # Muito retangular
            print(f"🔲 FORMA MUITO RETANGULAR (ratio {width_height_ratio:.2f}) - POSSÍVEL VEÍCULO!")
            return True
        
        print(f"✅ Forma não parece veículo (ratio {width_height_ratio:.2f})")
        return False
    
    def _is_obviously_vehicle(self, width, height, person_detection):
        """Detecção de veículos óbvios"""
        width_height_ratio = width / height
        
        # Casos óbvios
        if width_height_ratio > 2.0:  # Extremamente largo
            return True
        
        if 0.6 <= width_height_ratio <= 1.4:  # Quase quadrado
            return True
        
        return False
    
    def _is_too_symmetric(self, width, height, person_detection):
        """Detecta se o objeto é muito simétrico (típico de veículos)"""
        # Motos e veículos são mais simétricos que pessoas
        # Pessoas têm variação natural de largura ao longo da altura
        
        # Se a largura é muito consistente, pode ser veículo
        # Esta é uma validação adicional para casos duvidosos
        
        # Para implementação futura: análise de densidade de pixels
        # Por enquanto, usa proporções como proxy
        width_height_ratio = width / height
        
        # Objetos muito simétricos tendem a ter proporções próximas de 1:1
        if 0.7 <= width_height_ratio <= 1.3:
            return True
        
        return False
    

    
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
            
            # Calcula distância entre centroides
            track_center = ((track_bbox[0] + track_bbox[2]) // 2, (track_bbox[1] + track_bbox[3]) // 2)
            
            distance = np.sqrt((person_center[0] - track_center[0])**2 + 
                             (person_center[1] - track_center[1])**2)
            
            if distance < 100 and distance < best_distance:  # Threshold de 100px
                best_distance = distance
                best_track_id = track_id
        
        if best_track_id is not None:
            # Atualiza track existente
            track_data = self.person_tracks[best_track_id]
            track_data.update({
                'bbox': bbox,
                'confidence': confidence,
                'frames_seen': track_data['frames_seen'] + 1
            })
            
            # Adiciona ao histórico de movimento
            if 'movement_history' not in track_data:
                track_data['movement_history'] = []
            
            track_data['movement_history'].append({
                'center': person_center,
                'frame': self.frame_count,
                'bbox': bbox
            })
            
            # Mantém apenas os últimos 10 frames para análise
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
                'movement_history': [{
                    'center': person_center,
                    'frame': self.frame_count,
                    'bbox': bbox
                }]
            }
            return track_id
    
    def is_tracked_person_likely_human(self, track_id):
        """Valida se uma pessoa trackeada é provavelmente humana"""
        if track_id not in self.person_tracks:
            return False
        
        track_data = self.person_tracks[track_id]
        frames_seen = track_data['frames_seen']
        avg_confidence = track_data['confidence']
        
        # ✅ REGRA 1: Se foi vista em muitos frames, provavelmente é humana
        if frames_seen >= 3:
            return True
        
        # ✅ REGRA 2: Se confiança média é alta, provavelmente é humana
        if avg_confidence > 0.4:
            return True
        
        # ✅ REGRA 3: Se foi vista em alguns frames com confiança média
        if frames_seen >= 2 and avg_confidence > 0.3:
            return True
        
        return False
    
    def _validate_human_movement(self, track_id):
        """Valida se o movimento é típico de pessoa vs veículo"""
        if track_id not in self.person_tracks:
            return False
        
        track_data = self.person_tracks[track_id]
        
        # Se não tem histórico suficiente, não pode validar
        if 'movement_history' not in track_data or len(track_data['movement_history']) < 3:
            return True  # Assume que é humano se não pode validar
        
        movement_history = track_data['movement_history']
        
        # Calcula variação de movimento
        # Pessoas têm movimento mais orgânico, veículos mais linear
        movements = []
        for i in range(1, len(movement_history)):
            prev_center = movement_history[i-1]['center']
            curr_center = movement_history[i]['center']
            
            dx = curr_center[0] - prev_center[0]
            dy = curr_center[1] - prev_center[1]
            movement = np.sqrt(dx*dx + dy*dy)
            movements.append(movement)
        
        if len(movements) < 2:
            return True
        
        # Calcula variação do movimento
        movement_std = np.std(movements)
        movement_mean = np.mean(movements)
        
        # Se movimento é muito consistente (linear), pode ser veículo
        if movement_mean > 0 and movement_std / movement_mean < 0.3:
            print(f"⚠️ Movimento muito linear - possível veículo (std/mean: {movement_std/movement_mean:.2f})")
            return False
        
        return True

    def draw_results(self, frame, detections, validated_people):
        """Desenha resultados na imagem"""
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
        
        # Desenha informações das pessoas
        for person_data in validated_people:
            person = person_data['person']
            px1, py1, px2, py2 = person['bbox']
            
            # Info da pessoa
            info_text = f"PESSOA"
            if person_data['helmet_found']:
                info_text += " + CAPACETE"
            if person_data['vest_found']:
                info_text += " + COLETE"
            
            cv2.putText(frame_copy, info_text, (px1, py2 + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy
    
    def display_image(self, image):
        """Exibe imagem no canvas"""
        # Redimensiona para caber no canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
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
            
            # Limpa canvas e exibe nova imagem
            self.canvas.delete("all")
            self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                   image=self.current_image, anchor=tk.CENTER)
    
    def save_result(self, image, source_type):
        """Salva resultado da detecção"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"epi_detection_{source_type}_{timestamp}.jpg"
        
        cv2.imwrite(filename, image)
        print(f"📸 Resultado salvo: {filename}")
    
    def update_interface(self):
        """Atualiza interface periodicamente"""
        try:
            if self.root.winfo_exists():
                self.root.after(100, self.update_interface)
        except:
            pass
    
    def run(self):
        """Executa interface"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("🛑 Interface interrompida pelo usuário")
        finally:
            self.stop_camera()

def main():
    print("🏆 INTERFACE COMPLETA - DETECÇÃO DE EPIs COM MODELO SUPERIOR")
    print("=" * 70)
    print("📸 Imagens, 🎥 Vídeos, 📹 Câmera ao vivo")
    print("🎯 Modelo: epi_safe_fine_tuned (Fine-tuned para Longa Distância)")
    print("📏 Otimizado para 30m | mAP@0.5: 0.91 | Precision: 0.91")
    print("=" * 70)
    
    # Cria e executa interface
    interface = EPIDetectionInterface()
    interface.run()

if __name__ == "__main__":
    main()
