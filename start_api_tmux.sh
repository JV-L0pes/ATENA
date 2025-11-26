#!/bin/bash

# Script para iniciar a API Athena na porta 3000 via tmux
# Uso: ./start_api_tmux.sh

echo "🚀 Iniciando API Athena na porta 3000 via tmux..."

# Verificar se tmux está instalado
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux não está instalado. Instalando..."
    sudo apt update && sudo apt install -y tmux
fi

# Verificar se o ambiente virtual existe
if [ ! -d "venv" ]; then
    echo "❌ Ambiente virtual não encontrado. Criando..."
    python3 -m venv venv
fi

# Ativar ambiente virtual
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

# Verificar se a sessão tmux já existe
if tmux has-session -t athena-api 2>/dev/null; then
    echo "⚠️  Sessão 'athena-api' já existe. Parando sessão anterior..."
    tmux kill-session -t athena-api
fi

# Criar nova sessão tmux
echo "📱 Criando sessão tmux 'athena-api'..."
tmux new-session -d -s athena-api

# Enviar comandos para a sessão tmux
tmux send-keys -t athena-api "cd /home/ubuntu/athena_project" Enter
tmux send-keys -t athena-api "source venv/bin/activate" Enter
tmux send-keys -t athena-api "python start_rtsp.py" Enter

echo "✅ API iniciada na sessão tmux 'athena-api'"
echo ""
echo "📋 Comandos úteis:"
echo "  • Ver sessão: tmux attach -t athena-api"
echo "  • Listar sessões: tmux list-sessions"
echo "  • Parar sessão: tmux kill-session -t athena-api"
echo ""
echo "🌐 API disponível em: http://localhost:3000"
echo "📱 Dashboard: http://localhost:3000/athena"
echo "📺 Stream: http://localhost:3000/stream.mjpg"
echo ""
echo "🔍 Para ver os logs em tempo real:"
echo "  tmux attach -t athena-api"
