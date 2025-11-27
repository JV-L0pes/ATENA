#!/usr/bin/env python3
"""
Script de inicialização da API Athena - SISTEMA OTIMIZADO INTEGRADO
Sistema de detecção funcional com modelo da Fase 1 (best.pt)
"""

import sys
import argparse
import logging
from pathlib import Path

# Adicionar diretório raiz ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Configura logging com suporte a Unicode"""
    import sys
    import io
    
    # Configurar stdout para UTF-8 no Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    # Criar handler para console com UTF-8
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Criar handler para arquivo com UTF-8
    file_handler = logging.FileHandler('athena_api_optimized.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Formato das mensagens
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )

def validate_environment():
    """Valida ambiente e dependências"""
    logger = logging.getLogger(__name__)
    
    try:
        # Verificar PyTorch
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} disponível")
        
        if torch.cuda.is_available():
            logger.info(f"🚀 GPU CUDA disponível: {torch.cuda.get_device_name()}")
        else:
            logger.info("💻 Usando CPU (CUDA não disponível)")
        
        # Verificar OpenCV
        import cv2
        logger.info(f"✅ OpenCV {cv2.__version__} disponível")
        
        # Verificar FastAPI
        import fastapi
        logger.info(f"✅ FastAPI {fastapi.__version__} disponível")
        
        # Verificar Ultralytics
        import ultralytics
        logger.info(f"✅ Ultralytics {ultralytics.__version__} disponível")
        
        # Verificar modelo da Fase 1
        model_path = "athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt"
        if Path(model_path).exists():
            logger.info(f"✅ Modelo da Fase 1 encontrado: {model_path}")
        else:
            logger.error(f"❌ Modelo da Fase 1 não encontrado: {model_path}")
            return False
        
        # Verificar sistema otimizado
        optimized_path = "athena_realtime_optimized.py"
        if Path(optimized_path).exists():
            logger.info(f"✅ Sistema otimizado encontrado: {optimized_path}")
        else:
            logger.error(f"❌ Sistema otimizado não encontrado: {optimized_path}")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Dependência faltando: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Erro na validação: {e}")
        return False

def check_webcam():
    """Verifica se webcam está disponível"""
    logger = logging.getLogger(__name__)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                logger.info(f"✅ Webcam funcionando - Resolução: {w}x{h}")
                cap.release()
                return True
            else:
                logger.warning("⚠️ Webcam detectada mas não consegue capturar frames")
                cap.release()
                return False
        else:
            logger.warning("⚠️ Webcam não detectada")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao verificar webcam: {e}")
        return False

def test_optimized_system():
    """Testa o sistema otimizado"""
    logger = logging.getLogger(__name__)
    
    try:
        # Importar sistema otimizado
        try:
            from athena_realtime_optimized import AthenaRealtimeDetector
            logger.info("✅ Usando sistema de tempo real otimizado")
        except ImportError as e:
            logger.error(f"❌ Erro ao importar sistema otimizado: {e}")
            return False
        
        # Criar detector de teste
        detector = AthenaRealtimeDetector()
        
        # Testar inicialização
        if detector.initialize_model():
            logger.info("✅ Sistema otimizado inicializado com sucesso")
            
            # Testar processamento de frame
            import numpy as np
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            results = detector.process_frame(test_frame)
            
            if results:
                logger.info("✅ Processamento de frame funcionando")
                detector.cleanup()
                return True
            else:
                logger.error("❌ Falha no processamento de frame")
                detector.cleanup()
                return False
        else:
            logger.error("❌ Falha na inicialização do sistema otimizado")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao testar sistema otimizado: {e}")
        return False

def main():
    """Função principal"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Parse argumentos
    parser = argparse.ArgumentParser(description="Athena EPI Detection API - SISTEMA OTIMIZADO")
    parser.add_argument("--host", default="0.0.0.0", help="Host do servidor")
    parser.add_argument("--port", type=int, default=8000, help="Porta do servidor")
    parser.add_argument("--model", help="Caminho para modelo (padrão: best.pt da Fase 1)")
    parser.add_argument("--video", type=int, default=0, help="Fonte de vídeo (webcam)")
    parser.add_argument("--reload", action="store_true", help="Habilitar auto-reload")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--skip-validation", action="store_true", help="Pular validação de ambiente")
    parser.add_argument("--test-only", action="store_true", help="Apenas testar sistema sem iniciar servidor")
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 80)
    print("🛡️  ATHENA EPI DETECTION API - SISTEMA OTIMIZADO INTEGRADO")
    print("=" * 80)
    print("🎯 Sistema de detecção com modelo da Fase 1 (best.pt)")
    print("🚀 Detector otimizado com validação avançada")
    print("📹 Webcam em tempo real com bounding boxes")
    print("🖥️  Dashboard web responsivo")
    print("⚡ Performance superior com Tesla T4")
    print("=" * 80)
    
    # Validar ambiente
    if not args.skip_validation:
        logger.info("🔍 Validando ambiente...")
        if not validate_environment():
            logger.error("❌ Validação falhou - use --skip-validation para ignorar")
            sys.exit(1)
        
        logger.info("📹 Verificando webcam...")
        if not check_webcam():
            logger.warning("⚠️ Webcam não funcionando - sistema pode não funcionar corretamente")
        
        logger.info("🧪 Testando sistema otimizado...")
        if not test_optimized_system():
            logger.error("❌ Sistema otimizado não funcionando - verifique configurações")
            sys.exit(1)
    
    # Se apenas teste, sair aqui
    if args.test_only:
        logger.info("✅ Teste concluído com sucesso!")
        return 0
    
    # Configurar variáveis de ambiente
    import os
    if args.model:
        os.environ["MODEL_PATH"] = args.model
    else:
        os.environ["MODEL_PATH"] = "athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt"
    
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level
    
    # Importar e iniciar API otimizada
    try:
        logger.info("🚀 Carregando sistema otimizado...")
        
        # Importar API otimizada
        from backend.api_optimized import start_server, app_state
        
        # Configurar argumentos
        os.environ["API_HOST"] = args.host
        os.environ["API_PORT"] = str(args.port)
        os.environ["API_RELOAD"] = str(args.reload).lower()
        os.environ["VIDEO_SOURCE"] = str(args.video)
        
        # Informações de inicialização
        logger.info(f"🌐 Servidor: http://{args.host}:{args.port}")
        logger.info(f"📱 Dashboard: http://{args.host}:{args.port}/athena")
        logger.info(f"📺 Stream: http://{args.host}:{args.port}/stream.mjpg")
        logger.info(f"📡 SSE: http://{args.host}:{args.port}/events/detections")
        logger.info(f"📹 Webcam: {args.video}")
        logger.info(f"🤖 Modelo: Fase 1 Otimizado (best.pt)")
        
        print("\n" + "=" * 80)
        print("✅ SISTEMA OTIMIZADO PRONTO PARA INICIAR")
        print("=" * 80)
        print(f"🌐 Acesse: http://{args.host}:{args.port}/athena")
        print("🤖 Modelo: Fase 1 Otimizado (best.pt)")
        print("⚡ Performance: Tesla T4 + Validação Avançada")
        print("🔧 Pressione Ctrl+C para parar")
        print("=" * 80)
        
        # Iniciar servidor
        start_server(host=args.host, port=args.port)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Parando servidor...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
