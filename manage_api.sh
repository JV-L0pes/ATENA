#!/bin/bash

# Script de gerenciamento da API Athena via tmux
# Uso: ./manage_api.sh [start|stop|status|logs|attach]

SESSION_NAME="athena-api"
API_PORT="3000"

case "$1" in
    "start")
        echo "🚀 Iniciando API Athena na porta $API_PORT..."
        
        # Verificar se já está rodando
        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            echo "⚠️  API já está rodando na sessão '$SESSION_NAME'"
            echo "   Use './manage_api.sh stop' para parar primeiro"
            exit 1
        fi
        
        # Ativar ambiente virtual e iniciar
        cd /home/ubuntu/athena_project
        source venv/bin/activate
        
        tmux new-session -d -s $SESSION_NAME
        tmux send-keys -t $SESSION_NAME "cd /home/ubuntu/athena_project" Enter
        tmux send-keys -t $SESSION_NAME "source venv/bin/activate" Enter
        tmux send-keys -t $SESSION_NAME "python start_rtsp.py" Enter
        
        echo "✅ API iniciada na sessão '$SESSION_NAME'"
        echo "🌐 Disponível em: http://localhost:$API_PORT"
        ;;
        
    "stop")
        echo "🛑 Parando API Athena..."
        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            tmux kill-session -t $SESSION_NAME
            echo "✅ API parada com sucesso"
        else
            echo "⚠️  API não estava rodando"
        fi
        ;;
        
    "status")
        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            echo "✅ API está rodando na sessão '$SESSION_NAME'"
            echo "🌐 Porta: $API_PORT"
            echo "📱 Dashboard: http://localhost:$API_PORT/athena"
        else
            echo "❌ API não está rodando"
        fi
        ;;
        
    "logs")
        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            echo "📋 Mostrando logs da API..."
            tmux capture-pane -t $SESSION_NAME -p
        else
            echo "❌ API não está rodando"
        fi
        ;;
        
    "attach")
        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            echo "🔗 Conectando à sessão '$SESSION_NAME'..."
            echo "   Pressione Ctrl+B, depois D para desconectar"
            tmux attach -t $SESSION_NAME
        else
            echo "❌ API não está rodando"
        fi
        ;;
        
    *)
        echo "📋 Uso: $0 [start|stop|status|logs|attach]"
        echo ""
        echo "Comandos disponíveis:"
        echo "  start  - Inicia a API na porta $API_PORT"
        echo "  stop   - Para a API"
        echo "  status - Mostra status da API"
        echo "  logs   - Mostra logs da API"
        echo "  attach - Conecta à sessão tmux"
        ;;
esac
