#!/usr/bin/env python3
"""
Script de teste para o sistema de detecção de vídeos
Testa a funcionalidade básica do módulo de detecção de vídeos
"""

import sys
import os
from pathlib import Path

# Adicionar o diretório do projeto ao path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_video_detection():
    """Testa o módulo de detecção de vídeos"""
    try:
        from backend.video_detection import VideoAIDetector, VideoProcessingQueue
        
        print("🧪 Testando módulo de detecção de vídeos...")
        
        # Testar inicialização do detector
        detector = VideoAIDetector()
        print("✅ Detector inicializado com sucesso")
        
        # Testar formatos suportados
        formats = detector.get_supported_formats()
        print(f"✅ Formatos suportados: {formats}")
        
        # Testar inicialização da fila
        queue = VideoProcessingQueue(max_workers=1)
        print("✅ Fila de processamento inicializada")
        
        # Testar validação de vídeo (arquivo inexistente)
        validation = detector.validate_video("test_video.mp4")
        print(f"✅ Validação de vídeo inexistente: {validation}")
        
        print("🎉 Todos os testes passaram!")
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

def test_api_endpoints():
    """Testa se os endpoints da API estão configurados"""
    try:
        print("\n🧪 Testando configuração da API...")
        
        # Verificar se o arquivo da API existe
        api_file = project_root / "backend" / "api_optimized.py"
        if not api_file.exists():
            print("❌ Arquivo da API não encontrado")
            return False
        
        # Ler o arquivo e verificar se contém os endpoints de vídeo
        with open(api_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_endpoints = [
            "/api/videos/upload",
            "/api/videos/list", 
            "/api/videos/{video_id}/status",
            "/api/videos/{video_id}/results",
            "/api/videos/{video_id}/download"
        ]
        
        for endpoint in required_endpoints:
            if endpoint in content:
                print(f"✅ Endpoint encontrado: {endpoint}")
            else:
                print(f"❌ Endpoint não encontrado: {endpoint}")
                return False
        
        print("🎉 Todos os endpoints estão configurados!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste da API: {e}")
        return False

def test_frontend():
    """Testa se o frontend está configurado"""
    try:
        print("\n🧪 Testando configuração do frontend...")
        
        # Verificar se o arquivo HTML existe
        html_file = project_root / "frontend" / "index.html"
        if not html_file.exists():
            print("❌ Arquivo HTML não encontrado")
            return False
        
        # Verificar se o arquivo JS existe
        js_file = project_root / "frontend" / "js" / "app.js"
        if not js_file.exists():
            print("❌ Arquivo JavaScript não encontrado")
            return False
        
        # Ler o HTML e verificar se contém a seção de vídeos
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        if 'activeView === \'videos\'' in html_content:
            print("✅ Seção de vídeos encontrada no HTML")
        else:
            print("❌ Seção de vídeos não encontrada no HTML")
            return False
        
        # Ler o JS e verificar se contém as funções de vídeo
        with open(js_file, 'r', encoding='utf-8') as f:
            js_content = f.read()
        
        required_functions = [
            "handleVideoUpload",
            "loadVideoList",
            "viewVideoResults",
            "downloadVideo",
            "deleteVideo"
        ]
        
        for function in required_functions:
            if function in js_content:
                print(f"✅ Função encontrada: {function}")
            else:
                print(f"❌ Função não encontrada: {function}")
                return False
        
        print("🎉 Frontend está configurado corretamente!")
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste do frontend: {e}")
        return False

def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes do sistema de detecção de vídeos...")
    print("=" * 60)
    
    tests = [
        test_video_detection,
        test_api_endpoints,
        test_frontend
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Resultado dos testes: {passed}/{total} passaram")
    
    if passed == total:
        print("🎉 Todos os testes passaram! Sistema pronto para uso.")
        print("\n📋 Próximos passos:")
        print("1. Instalar dependências: pip install -r requirements_simple.txt")
        print("2. Iniciar a API: python start_api_optimized.py")
        print("3. Abrir o dashboard no navegador")
        print("4. Navegar para a seção 'Vídeos' para testar o upload")
    else:
        print("❌ Alguns testes falharam. Verifique os erros acima.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
