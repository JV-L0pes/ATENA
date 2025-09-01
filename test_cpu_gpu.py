#!/usr/bin/env python3
"""
Script de teste para verificar as configurações CPU/GPU
"""

import os
import sys
import torch
from pathlib import Path

# Adicionar src ao path
sys.path.append('src')

def test_cpu_gpu_config():
    """Testa as configurações de CPU/GPU"""
    
    print("🔍 Testando Configurações CPU/GPU...")
    print("=" * 50)
    
    # 1. Verificar disponibilidade de CUDA
    print(f"📊 CUDA disponível: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"📊 Número de GPUs: {torch.cuda.device_count()}")
        print(f"📊 GPU atual: {torch.cuda.get_device_name(0)}")
    
    # 2. Testar configurações de ambiente
    print("\n🔧 Configurações de Ambiente:")
    print(f"   FORCE_CPU_ONLY: {os.getenv('FORCE_CPU_ONLY', 'false')}")
    print(f"   DEVICE_PREFERENCE: {os.getenv('DEVICE_PREFERENCE', 'auto')}")
    
    # 3. Testar importação do detector
    try:
        from src.epi_detector import EPIDetector
        print("\n✅ EPIDetector importado com sucesso")
        
        # 4. Testar inicialização com diferentes configurações
        print("\n🧪 Testando diferentes configurações:")
        
        # Teste 1: CPU Only
        print("\n   Teste 1: CPU Only")
        detector_cpu = EPIDetector(
            model_path="yolov5/yolov5n.pt",
            force_cpu_only=True,
            device_preference="cpu"
        )
        device_cpu = detector_cpu._select_device()
        print(f"   Dispositivo selecionado: {device_cpu}")
        
        # Teste 2: Auto (deve usar GPU se disponível)
        print("\n   Teste 2: Auto")
        detector_auto = EPIDetector(
            model_path="yolov5/yolov5n.pt",
            force_cpu_only=False,
            device_preference="auto"
        )
        device_auto = detector_auto._select_device()
        print(f"   Dispositivo selecionado: {device_auto}")
        
        # Teste 3: CUDA (se disponível)
        if torch.cuda.is_available():
            print("\n   Teste 3: CUDA")
            detector_cuda = EPIDetector(
                model_path="yolov5/yolov5n.pt",
                force_cpu_only=False,
                device_preference="cuda"
            )
            device_cuda = detector_cuda._select_device()
            print(f"   Dispositivo selecionado: {device_cuda}")
        else:
            print("\n   Teste 3: CUDA (não disponível)")
        
        print("\n✅ Todos os testes de configuração passaram!")
        
    except ImportError as e:
        print(f"\n❌ Erro ao importar EPIDetector: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Erro durante teste: {e}")
        return False
    
    # 5. Testar configurações do backend
    print("\n🔧 Testando Configurações do Backend:")
    try:
        from backend.config import CONFIG
        
        model_config = CONFIG.get_model_config()
        print(f"   Model Path: {model_config['model_path']}")
        print(f"   Force CPU Only: {model_config['force_cpu_only']}")
        print(f"   Device Preference: {model_config['device_preference']}")
        
        print("\n✅ Configurações do backend OK!")
        
    except Exception as e:
        print(f"\n❌ Erro nas configurações do backend: {e}")
        return False
    
    print("\n🎉 Todos os testes passaram com sucesso!")
    return True

def test_model_loading():
    """Testa o carregamento do modelo"""
    
    print("\n🤖 Testando Carregamento do Modelo...")
    print("=" * 50)
    
    try:
        from src.epi_detector import EPIDetector
        
        # Verificar se existe um modelo
        model_path = "yolov5/yolov5n.pt"
        if not Path(model_path).exists():
            print(f"❌ Modelo não encontrado: {model_path}")
            print("💡 Baixando modelo padrão...")
            
            # Tentar baixar modelo padrão
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            model.save(model_path)
            print(f"✅ Modelo salvo em: {model_path}")
        
        # Testar carregamento
        detector = EPIDetector(
            model_path=model_path,
            force_cpu_only=True,  # Forçar CPU para teste
            device_preference="cpu"
        )
        
        print("🔄 Inicializando modelo...")
        detector.initialize_model()
        
        if detector.is_initialized:
            print("✅ Modelo carregado com sucesso!")
            print(f"   Dispositivo: {detector.device}")
            print(f"   Modelo: {detector.model}")
            return True
        else:
            print("❌ Falha ao inicializar modelo")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando Testes do Sistema CPU/GPU...")
    print("=" * 60)
    
    # Executar testes
    config_ok = test_cpu_gpu_config()
    model_ok = test_model_loading()
    
    print("\n" + "=" * 60)
    print("📊 RESULTADO DOS TESTES:")
    print(f"   Configurações CPU/GPU: {'✅ OK' if config_ok else '❌ FALHOU'}")
    print(f"   Carregamento do Modelo: {'✅ OK' if model_ok else '❌ FALHOU'}")
    
    if config_ok and model_ok:
        print("\n🎉 SISTEMA CPU/GPU FUNCIONANDO PERFEITAMENTE!")
        print("\n💡 Para usar CPU only:")
        print("   export FORCE_CPU_ONLY=true")
        print("   python start_api.py")
        print("\n💡 Para usar GPU:")
        print("   export DEVICE_PREFERENCE=cuda")
        print("   python start_api.py")
    else:
        print("\n❌ ALGUNS TESTES FALHARAM!")
        print("   Verifique os erros acima e corrija os problemas.")
