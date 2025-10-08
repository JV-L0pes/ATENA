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
    
    # Verificar se RTSP_URL está definida
    if not os.getenv('RTSP_URL'):
        print("❌ RTSP_URL não definida!")
        return 1
    
    # Testar conexão RTSP
    print("🔍 Testando conexão RTSP...")
    try:
        import cv2
        rtsp_url = os.getenv('RTSP_URL')
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ RTSP funcionando: {frame.shape}")
            else:
                print("⚠️ RTSP conectado mas sem frame")
        else:
            print("❌ RTSP não conectado")
        
        cap.release()
    except Exception as e:
        print(f"❌ Erro ao testar RTSP: {e}")
    
    # Iniciar sistema
    print("🚀 Iniciando sistema Athena...")
    try:
        # Importar e executar o sistema
        sys.path.insert(0, str(Path(__file__).parent))
        from start_api_optimized import main as start_main
        
        # Executar com argumentos da configuração
        api_port = os.getenv('API_PORT', '3000')
        sys.argv = ['start_api_optimized.py', '--host', '0.0.0.0', '--port', api_port]
        print(f"🌐 Iniciando na porta: {api_port}")
        start_main()
        
    except Exception as e:
        print(f"❌ Erro ao iniciar sistema: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
