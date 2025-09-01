#!/usr/bin/env python3
"""
Teste simples para verificar a estrutura modular do backend
"""

import sys
from pathlib import Path

# Adicionar backend ao path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def test_imports():
    """Testa se todos os módulos podem ser importados"""
    try:
        print("🧪 Testando imports dos módulos...")
        
        # Testar configuração
        from config import CONFIG
        print("✅ config.py importado com sucesso")
        
        # Testar utilitários
        from utils import setup_logging, format_timestamp
        print("✅ utils.py importado com sucesso")
        
        # Testar detecção
        from detection import EPIDetectionSystem
        print("✅ detection.py importado com sucesso")
        
        # Testar snapshot
        from snapshot import EPISnapshotSystem
        print("✅ snapshot.py importado com sucesso")
        
        # Testar histórico
        from history import EPIHistorySystem
        print("✅ history.py importado com sucesso")
        
        # Testar API
        from api import app
        print("✅ api.py importado com sucesso")
        
        print("\n🎉 Todos os módulos importados com sucesso!")
        return True
        
    except ImportError as e:
        print(f"❌ Erro ao importar módulo: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def test_config():
    """Testa configurações básicas"""
    try:
        print("\n🔧 Testando configurações...")
        
        from config import CONFIG
        
        # Verificar configurações básicas
        assert CONFIG.API_HOST == "0.0.0.0"
        assert CONFIG.API_PORT == 8000
        assert CONFIG.VIDEO_FPS == 30
        
        print("✅ Configurações básicas válidas")
        
        # Verificar configurações do modelo
        model_config = CONFIG.get_model_config()
        assert "model_path" in model_config
        assert "conf_thresh" in model_config
        
        print("✅ Configurações do modelo válidas")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas configurações: {e}")
        return False

def test_utils():
    """Testa funções utilitárias"""
    try:
        print("\n🛠️ Testando utilitários...")
        
        from utils import format_timestamp, format_uptime, calculate_iou
        
        # Testar formatação de timestamp
        timestamp = format_timestamp()
        assert isinstance(timestamp, str)
        print("✅ Formatação de timestamp funcionando")
        
        # Testar formatação de uptime
        uptime = format_uptime(3661)  # 1 hora, 1 minuto, 1 segundo
        assert "1h" in uptime
        print("✅ Formatação de uptime funcionando")
        
        # Testar cálculo de IoU
        box1 = {"x": 0, "y": 0, "w": 10, "h": 10}
        box2 = {"x": 5, "y": 5, "w": 10, "h": 10}
        iou = calculate_iou(box1, box2)
        assert 0 <= iou <= 1
        print("✅ Cálculo de IoU funcionando")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nos utilitários: {e}")
        return False

def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes do backend Athena...\n")
    
    tests = [
        test_imports,
        test_config,
        test_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Teste falhou com erro: {e}")
    
    print(f"\n📊 Resultados: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! Backend está funcionando corretamente.")
        return 0
    else:
        print("❌ Alguns testes falharam. Verifique os erros acima.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
