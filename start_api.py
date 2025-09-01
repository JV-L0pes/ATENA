#!/usr/bin/env python3
"""
Script simples para iniciar o sistema Athena
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def main():
    print("🚀 Iniciando Sistema Athena...")
    
    # Modelo hardcoded
    model_path = "yolov5/runs/train/epi_safe_fine_tuned/weights/best.pt"
    
    # Verificar se o modelo existe
    if not Path(model_path).exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print("📁 Procurando por modelos treinados...")
        
        # Procurar por qualquer modelo disponível
        train_dir = Path("yolov5/runs/train")
        if train_dir.exists():
            for train_folder in train_dir.iterdir():
                if train_folder.is_dir():
                    weights_dir = train_folder / "weights"
                    if weights_dir.exists():
                        best_model = weights_dir / "best.pt"
                        if best_model.exists():
                            model_path = str(best_model)
                            print(f"✅ Modelo encontrado: {model_path}")
                            break
        
        if not Path(model_path).exists():
            print("❌ Nenhum modelo treinado encontrado!")
            print("💡 Execute o treinamento primeiro ou verifique o diretório yolov5/runs/train")
            return
    
    print(f"🎯 Usando modelo: {model_path}")
    
    # Configurar variáveis de ambiente
    os.environ["MODEL_PATH"] = model_path
    os.environ["API_HOST"] = "0.0.0.0"
    os.environ["API_PORT"] = "8000"
    
    processes = []
    
    try:
        # 1. Iniciar Backend
        print("🔧 Iniciando Backend...")
        backend = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend.api:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])
        processes.append(backend)
        print(f"✅ Backend iniciado (PID: {backend.pid})")
        
        # Aguardar backend inicializar
        print("⏳ Aguardando backend...")
        time.sleep(15)  # Mais tempo para o modelo carregar
        
        # 2. Iniciar Frontend
        print("🌐 Iniciando Frontend...")
        frontend = subprocess.Popen([
            sys.executable, "-m", "http.server", "3000"
        ], cwd="frontend")
        processes.append(frontend)
        print(f"✅ Frontend iniciado (PID: {frontend.pid})")
        
        # Aguardar frontend inicializar
        print("⏳ Aguardando frontend...")
        time.sleep(3)
        
        # 3. Abrir Navegador
        print("🌐 Abrindo navegador...")
        webbrowser.open("http://localhost:3000")
        print("✅ Navegador aberto!")
        
        print("\n🎉 Sistema iniciado com sucesso!")
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend: http://localhost:8000")
        print("📊 API Docs: http://localhost:8000/docs")
        print("\n💡 Pressione Ctrl+C para encerrar")
        
        # Manter rodando
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Encerrando...")
    except Exception as e:
        print(f"❌ Erro: {e}")
    finally:
        # Encerrar processos
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        print("✅ Sistema encerrado")

if __name__ == "__main__":
    main()
