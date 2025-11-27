#!/usr/bin/env python3
"""
Script de inicialização para RTSP
Carrega variáveis do .env.rtsp e inicia o sistema
"""

import os
import sys
import subprocess
from pathlib import Path

def load_env_file(env_file):
    """Carrega variáveis de um arquivo .env"""
    env_vars = {}
    if Path(env_file).exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
                        os.environ[key] = value
    return env_vars

def main():
    """Função principal"""
    print("🚀 Iniciando Athena com RTSP...")
    
    # Carregar variáveis do .env.rtsp
    env_file = Path(__file__).parent / ".env.rtsp"
    env_vars = load_env_file(env_file)
    
    print(f"📹 RTSP_URL: {os.getenv('RTSP_URL')}")
    print(f"🎯 VIDEO_TYPE: {os.getenv('VIDEO_TYPE')}")
    print(f"🤖 MODEL_PATH: {os.getenv('MODEL_PATH')}")
    
    # Verificar se RTSP_URL está definida (mas não falhar se não estiver)
    if not os.getenv('RTSP_URL'):
        print("⚠️ RTSP_URL não definida - sistema iniciará com webcam padrão")
        # Definir valores padrão
        os.environ['RTSP_URL'] = '0'
        os.environ['VIDEO_TYPE'] = 'usb'
    
    # Testar conexão RTSP (opcional)
    rtsp_url = os.getenv('RTSP_URL')
    if rtsp_url and rtsp_url != '0':
        print("🔍 Testando conexão RTSP...")
        try:
            import cv2
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # Timeout de 5 segundos
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ RTSP funcionando: {frame.shape}")
                else:
                    print("⚠️ RTSP conectado mas sem frame - sistema continuará com webcam")
            else:
                print("⚠️ RTSP não conectado - sistema continuará com webcam")
            
            cap.release()
        except Exception as e:
            print(f"⚠️ Erro ao testar RTSP: {e} - sistema continuará com webcam")
    else:
        print("💡 Usando webcam padrão (RTSP não configurado)")
    
    # Iniciar sistema
    print("🚀 Iniciando sistema Athena...")
    try:
        # Importar e executar o sistema
        sys.path.insert(0, str(Path(__file__).parent))
        from start_api_optimized import main as start_main
        
        # Executar com argumentos da configuração
        api_port = os.getenv('API_PORT', '3000')
        # Passar --skip-validation para evitar encerramento precoce em ambientes sem webcam/RTSP
        sys.argv = ['start_api_optimized.py', '--host', '0.0.0.0', '--port', api_port, '--skip-validation']
        print(f"🌐 Iniciando na porta: {api_port}")
        start_main()
        
    except Exception as e:
        print(f"❌ Erro ao iniciar sistema: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
