#!/bin/bash

# Script para testar conexões RTSP
# Uso: ./test_rtsp.sh [URL_RTSP]

RTSP_URL=${1:-"rtsp://100.116.238.18:8554/cam1"}

echo "🔍 Testando conexão RTSP: $RTSP_URL"
echo ""

cd /home/ubuntu/athena_project
source venv/bin/activate

# Teste com timeout
timeout 10 python -c "
import cv2
import sys

rtsp_url = '$RTSP_URL'
print(f'📹 Testando: {rtsp_url}')

try:
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_TIMEOUT, 5000)
    
    if cap.isOpened():
        print('✅ Conexão estabelecida')
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f'✅ Frame recebido: {frame.shape}')
            print('✅ RTSP funcionando perfeitamente!')
        else:
            print('⚠️ Conexão OK mas sem frame')
    else:
        print('❌ Falha na conexão')
    
    cap.release()
    
except Exception as e:
    print(f'❌ Erro: {e}')
"

exit_code=$?

if [ $exit_code -eq 124 ]; then
    echo ""
    echo "⏰ Timeout - RTSP não respondeu em 10 segundos"
    echo "💡 Possíveis problemas:"
    echo "   • URL incorreta"
    echo "   • Câmera offline"
    echo "   • Firewall bloqueando"
    echo "   • Credenciais necessárias"
elif [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 Teste concluído com sucesso!"
else
    echo ""
    echo "❌ Erro no teste (código: $exit_code)"
fi
