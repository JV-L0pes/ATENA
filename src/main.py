"""
Script principal para execução do detector de EPIs
Permite processar imagens, vídeos ou webcam
"""

import argparse
import cv2
import os
import sys
from pathlib import Path
import logging
from epi_detector import EPIDetector

def setup_logging():
    """Configura sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('atena.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def process_image(detector: EPIDetector, image_path: str, output_path: str = None):
    """
    Processa uma imagem individual
    
    Args:
        detector: Instância do detector de EPIs
        image_path: Caminho para a imagem
        output_path: Caminho para salvar resultado (opcional)
    """
    # Carrega imagem
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Não foi possível carregar a imagem: {image_path}")
        return
    
    logging.info(f"Processando imagem: {image_path}")
    
    # Processa imagem
    processed_image, detections = detector.process_image(image)
    
    # Salva resultado se especificado
    if output_path:
        cv2.imwrite(output_path, processed_image)
        logging.info(f"Resultado salvo em: {output_path}")
    
    # Mostra estatísticas
    _print_detection_stats(detections)
    
    # Exibe imagem
    cv2.imshow('Atena - Detecção de EPIs', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(detector: EPIDetector, video_path: str, output_path: str = None):
    """
    Processa um vídeo
    
    Args:
        detector: Instância do detector de EPIs
        video_path: Caminho para o vídeo
        output_path: Caminho para salvar resultado (opcional)
    """
    # Abre vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Não foi possível abrir o vídeo: {video_path}")
        return
    
    # Configurações do vídeo
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Configura writer se output especificado
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    logging.info(f"Processando vídeo: {video_path}")
    logging.info(f"FPS: {fps}, Resolução: {width}x{height}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Log a cada 30 frames
            logging.info(f"Processando frame {frame_count}")
        
        # Processa frame
        processed_frame, detections = detector.process_image(frame)
        
        # Salva frame se writer configurado
        if writer:
            writer.write(processed_frame)
        
        # Exibe frame
        cv2.imshow('Atena - Detecção de EPIs', processed_frame)
        
        # Pressiona 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    logging.info(f"Vídeo processado: {frame_count} frames")

def process_webcam(detector: EPIDetector):
    """
    Processa stream da webcam em tempo real
    
    Args:
        detector: Instância do detector de EPIs
    """
    # Abre webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Não foi possível abrir a webcam")
        return
    
    logging.info("Iniciando detecção em tempo real via webcam")
    logging.info("Pressione 'q' para sair, 's' para salvar frame atual")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Erro ao capturar frame da webcam")
            break
        
        # Processa frame
        processed_frame, detections = detector.process_image(frame)
        
        # Exibe frame
        cv2.imshow('Atena - Detecção de EPIs (Webcam)', processed_frame)
        
        # Controles
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Salva frame atual
            output_path = f"webcam_frame_{len(os.listdir('output'))}.jpg"
            cv2.imwrite(output_path, processed_frame)
            logging.info(f"Frame salvo: {output_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam encerrada")

def _print_detection_stats(detections: list):
    """
    Imprime estatísticas das detecções
    
    Args:
        detections: Lista de detecções validadas
    """
    people_count = len([d for d in detections if d.get('status') == 'person'])
    helmet_correct = len([d for d in detections if d.get('status') == 'helmet_correct'])
    helmet_violations = len([d for d in detections if d.get('status') == 'helmet_violation'])
    vest_correct = len([d for d in detections if d.get('status') == 'vest_correct'])
    vest_violations = len([d for d in detections if d.get('status') == 'vest_violation'])
    
    print("\n" + "="*50)
    print("ESTATÍSTICAS DE DETECÇÃO")
    print("="*50)
    print(f"Pessoas detectadas: {people_count}")
    print(f"Capacetes corretos: {helmet_correct}")
    print(f"Violações de capacete: {helmet_violations}")
    print(f"Coletes corretos: {vest_correct}")
    print(f"Violações de colete: {vest_violations}")
    
    total_violations = helmet_violations + vest_violations
    if total_violations > 0:
        print(f"\n⚠️  TOTAL DE INFRAÇÕES: {total_violations}")
    else:
        print(f"\n✅ TODOS OS EPIs ESTÃO CORRETOS!")
    print("="*50)

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Atena - Sistema de Detecção de EPIs')
    parser.add_argument('--source', type=str, required=True,
                       help='Caminho para imagem, vídeo ou "webcam"')
    parser.add_argument('--output', type=str, default=None,
                       help='Caminho para salvar resultado (opcional)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Caminho para arquivo de configuração')
    
    args = parser.parse_args()
    
    # Configura logging
    setup_logging()
    
    try:
        # Inicializa detector
        logging.info("Inicializando detector de EPIs...")
        detector = EPIDetector(args.config)
        
        # Cria diretório de output se necessário
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Processa fonte baseado no tipo
        if args.source.lower() == 'webcam':
            process_webcam(detector)
        elif os.path.isfile(args.source):
            # Verifica extensão para determinar se é imagem ou vídeo
            ext = Path(args.source).suffix.lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                process_image(detector, args.source, args.output)
            elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
                process_video(detector, args.source, args.output)
            else:
                logging.error(f"Formato de arquivo não suportado: {ext}")
        else:
            logging.error(f"Fonte não encontrada: {args.source}")
    
    except Exception as e:
        logging.error(f"Erro durante execução: {str(e)}")
        raise

if __name__ == "__main__":
    main()
