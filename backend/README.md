# Backend Athena - Sistema de Detecção de EPIs

Backend modular para o dashboard Athena, integrando modelo YOLOv5 para detecção de Equipamentos de Proteção Individual (EPIs).

## 🏗️ Arquitetura Modular

```
backend/
├── __init__.py          # Inicialização do módulo
├── config.py            # Configurações centralizadas
├── utils.py             # Utilitários e helpers
├── detection.py         # Sistema de detecção YOLOv5
├── snapshot.py          # Sistema de captura de snapshots
├── history.py           # Sistema de histórico e analytics
├── api.py               # API principal FastAPI
├── .env.example         # Exemplo de variáveis de ambiente
└── README.md            # Esta documentação
```

## 🚀 Módulos

### 1. Config (`config.py`)
- Centraliza todas as configurações da aplicação
- Suporte a variáveis de ambiente
- Validação automática de configurações
- Configurações para modelo, vídeo, detecção e performance

### 2. Utils (`utils.py`)
- Funções utilitárias para formatação
- Cálculos de IoU (Intersection over Union)
- Validação de dados
- Manipulação de frames e imagens
- Decorators para debounce e throttle

### 3. Detection (`detection.py`)
- **EPIDetector**: Classe principal para detecção YOLOv5
- **EPIDetectionSystem**: Sistema completo de detecção
- Threading para processamento assíncrono
- Cache de resultados e estatísticas em tempo real
- Suporte a GPU CUDA e CPU

### 4. Snapshot (`snapshot.py`)
- **SnapshotManager**: Gerenciamento de snapshots
- **EPISnapshotSystem**: Sistema de captura para EPIs
- Nomenclatura automática com timestamp e metadados
- Limpeza automática de snapshots antigos
- Estatísticas de uso

### 5. History (`history.py`)
- **DetectionRecord**: Registro individual de detecção
- **HistoryManager**: Gerenciamento de histórico
- **EPIHistorySystem**: Sistema de histórico para EPIs
- Análise de compliance ao longo do tempo
- Busca e filtros avançados
- Exportação de dados

### 6. API (`api.py`)
- **FastAPI** com todos os endpoints necessários
- **AppState**: Estado global da aplicação
- **ConnectionManager**: Gerenciamento de WebSockets
- Middleware para estatísticas
- Integração com todos os módulos

## 🔧 Configuração

### Variáveis de Ambiente
Copie `.env.example` para `.env` e configure:

```bash
# Configurações básicas
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=yolov5/yolov5n.pt
VIDEO_SOURCE=0

# Performance
VIDEO_FPS=30
MODEL_CONF_THRESH=0.35
MODEL_IOU_THRESH=0.45
```

### Dependências
```bash
pip install -r requirements.txt
```

## 🚀 Inicialização

### Script Principal
```bash
python start_api.py --host 0.0.0.0 --port 8000 --model yolov5/yolov5n.pt --video 0
```

### Parâmetros Disponíveis
- `--host`: Host para bind (padrão: 0.0.0.0)
- `--port`: Porta (padrão: 8000)
- `--model`: Caminho para modelo YOLOv5
- `--video`: Fonte de vídeo (padrão: 0 para webcam)
- `--reload`: Habilitar reload automático
- `--log-level`: Nível de log (DEBUG, INFO, WARNING, ERROR)

### Inicialização Direta
```python
from backend.api import start_server
start_server(host="0.0.0.0", port=8000)
```

## 📡 Endpoints da API

### Básicos
- `GET /` - Informações da API
- `GET /health` - Verificação de saúde
- `GET /status` - Status do sistema

### Vídeo e Detecção
- `GET /stream.mjpg` - Stream MJPEG
- `GET /events/detections` - SSE para detecções
- `GET /stats` - Estatísticas atuais

### Histórico
- `GET /history` - Lista de detecções
- `GET /history/stats` - Estatísticas do histórico
- `GET /history/trend` - Tendência de compliance
- `GET /history/search` - Busca no histórico

### Snapshots
- `POST /snapshot` - Capturar snapshot
- `GET /snapshots/{filename}` - Download de snapshot

### Configuração
- `GET /config` - Configuração atual
- `PUT /config` - Atualizar configuração

### WebSocket
- `WS /ws/detections` - Conexão WebSocket para detecções

## 🔍 Funcionalidades

### Detecção em Tempo Real
- Processamento assíncrono de frames
- Detecção YOLOv5 com suporte a GPU
- Estatísticas em tempo real
- Cache de resultados para performance

### Sistema de Snapshots
- Captura automática com metadados
- Nomenclatura inteligente
- Limpeza automática
- Estatísticas de uso

### Histórico e Analytics
- Registro de todas as detecções
- Análise de compliance
- Tendências temporais
- Busca e filtros avançados
- Exportação de dados

### Performance
- Threading para operações assíncronas
- Cache inteligente
- Debouncing e throttling
- Configurações de FPS ajustáveis

## 🐛 Troubleshooting

### Problemas Comuns

1. **Modelo não encontrado**
   - Verifique `MODEL_PATH` em `.env`
   - Confirme se o arquivo `.pt` existe

2. **Erro de CUDA**
   - Verifique se PyTorch com CUDA está instalado
   - O sistema automaticamente usa CPU se CUDA não estiver disponível

3. **Vídeo não carrega**
   - Verifique `VIDEO_SOURCE` (0 para webcam)
   - Confirme permissões de acesso à câmera

4. **Performance lenta**
   - Ajuste `VIDEO_FPS` e `MODEL_CONF_THRESH`
   - Use GPU se disponível
   - Ajuste tamanhos de fila em configurações

### Logs
- Logs são salvos em `athena_backend.log`
- Configure `LOG_LEVEL` para mais detalhes
- Use `--log-level DEBUG` para troubleshooting

## 🔧 Desenvolvimento

### Estrutura de Classes
- Cada módulo é independente e testável
- Interfaces claras entre módulos
- Padrão de nomenclatura consistente
- Documentação inline completa

### Extensibilidade
- Fácil adição de novos tipos de detecção
- Sistema de plugins para modelos
- Configurações flexíveis via ambiente
- APIs padronizadas para integração

### Testes
```bash
# Executar testes (quando implementados)
python -m pytest backend/tests/

# Verificar cobertura
python -m pytest --cov=backend
```

## 📊 Monitoramento

### Métricas Disponíveis
- FPS de detecção
- Uso de memória e CPU
- Estatísticas de compliance
- Performance de inferência
- Status de conexões

### Health Checks
- Verificação automática de sistemas
- Status de dependências
- Métricas de performance
- Alertas de falha

## 🤝 Contribuição

1. Mantenha a arquitetura modular
2. Documente novas funcionalidades
3. Siga padrões de nomenclatura
4. Implemente testes para novos módulos
5. Atualize configurações e documentação

## 📝 Licença

Este projeto é um sistema completo de detecção de EPIs em tempo real.
