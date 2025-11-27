# 🛡️ Athena - Sistema de Detecção de EPIs

Sistema inteligente de detecção de Equipamentos de Proteção Individual (EPIs) usando YOLOv11 e interface web moderna.

## ✨ Características

- **Detecção em Tempo Real**: Identifica pessoas e EPIs (17 classes) via RTSP/P2P ou webcam
- **Processamento de Vídeos**: Upload e análise de vídeos com detecção frame a frame
- **Interface Web Responsiva**: Dashboard moderno com Alpine.js e Tailwind CSS
- **Modelo Treinado**: Utiliza modelo YOLOv11 customizado (best.pt) com 17 classes
- **Relatórios Dinâmicos**: Geração automática de relatórios baseados em todas as classes detectadas
- **Histórico e Snapshots**: Sistema completo de registro e análise
- **GPU Otimizado**: Requer GPU CUDA para melhor performance

## 📁 Estrutura do Projeto

```
athena_project/
├── core/                    # Código principal de detecção
│   ├── detector.py         # Sistema de detecção consolidado
│   ├── config.py           # Configurações centralizadas
│   └── __init__.py
│
├── backend/                 # Backend API (legado - será migrado)
│   ├── api_optimized.py    # API FastAPI principal
│   ├── config.py           # Configurações do backend
│   ├── video_detection.py  # Detecção em vídeos (legado)
│   ├── video_report.py     # Sistema de relatórios
│   ├── history.py          # Histórico de detecções
│   └── snapshot.py         # Sistema de snapshots
│
├── api/                     # Nova estrutura de API (em desenvolvimento)
│   ├── main.py            # FastAPI app principal
│   └── routes/            # Rotas organizadas por funcionalidade
│
├── frontend/                # Interface web
│   ├── index.html         # Página principal
│   ├── js/
│   │   ├── app.js        # Lógica principal (Alpine.js)
│   │   └── utils.js      # Utilitários
│   └── styles/
│       └── main.css      # Estilos consolidados
│
├── models/                  # Modelos treinados
│   └── best.pt            # Modelo principal (YOLOv11)
│
├── storage/                 # Dados de produção
│   ├── videos/            # Vídeos processados
│   ├── uploads/           # Vídeos enviados
│   ├── reports/           # Relatórios gerados
│   ├── snapshots/         # Snapshots
│   └── logs/              # Logs de produção
│
├── archive/                 # Dados arquivados
│   └── training_data/     # Dados de treinamento
│
├── dev/                    # Ferramentas de desenvolvimento
│   ├── tools/             # Ferramentas de processamento
│   ├── scripts/           # Scripts de treinamento
│   └── tests/             # Testes
│
├── start_api_optimized.py  # Script de inicialização
└── requirements.txt        # Dependências Python
```

## 🚀 Início Rápido

### Pré-requisitos

- Python 3.8+
- GPU NVIDIA com CUDA (obrigatório)
- PyTorch com suporte CUDA
- Webcam ou fonte RTSP/P2P

### Instalação

1. **Clone o repositório**
```bash
git clone <repository-url>
cd athena_project
```

2. **Instale PyTorch com CUDA** (IMPORTANTE: faça isso primeiro)
```bash
# Para CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Instale as dependências**
```bash
pip install -r requirements.txt
```

4. **Verifique o modelo**
```bash
# O modelo deve estar em models/best.pt ou no caminho padrão
ls models/best.pt
```

### Execução

1. **Inicie o backend**
```bash
python start_api_optimized.py
```

2. **Acesse o frontend**
```bash
# Abra o arquivo frontend/index.html no navegador
# Ou use um servidor local:
cd frontend
python -m http.server 8080
# Acesse: http://localhost:8080
```

3. **Acesse a API**
```
http://localhost:3000
```

## 🔧 Configuração

### Variáveis de Ambiente

```bash
# Modelo
MODEL_PATH=models/best.pt
MODEL_CONF_THRESH=0.25
MODEL_IOU_THRESH=0.45

# API
API_HOST=0.0.0.0
API_PORT=3000

# Vídeo
VIDEO_SOURCE=0  # 0 = webcam padrão
RTSP_URL=rtsp://user:pass@ip:port/stream  # Para RTSP
VIDEO_FPS=30

# EPIs Requeridos
REQUIRED_EPIS=helmet,safety-vest,gloves,glasses
```

### Parâmetros do Modelo

- **Confidence Threshold**: Limite mínimo de confiança para detecções (padrão: 0.25)
- **IoU Threshold**: Limite de sobreposição para supressão de detecções duplicadas (padrão: 0.45)
- **Max Detections**: Número máximo de detecções por frame (padrão: 300)

## 🎯 Modelo Treinado

- **Nome**: `best.pt` (Fase 1 Otimizado)
- **Arquivo**: `models/best.pt`
- **Classes**: 17 classes de EPIs + person
- **Performance**: 
  - Precision: 89.9%
  - Recall: 76.4%
  - mAP50: 83.2%
  - mAP50-95: 63.6%

## 🌐 Endpoints da API

### Principais
- `GET /` - Redireciona para frontend
- `GET /health` - Verificação de saúde
- `GET /status` - Status do sistema
- `GET /stream.mjpg` - Stream de vídeo MJPEG
- `GET /events/detections` - SSE para detecções em tempo real
- `GET /stats` - Estatísticas atuais
- `POST /api/detect-frame` - Detecção em frame individual

### Vídeos
- `POST /api/videos/upload` - Upload de vídeo
- `GET /api/videos/list` - Lista de vídeos
- `GET /api/videos/{id}/status` - Status do processamento
- `GET /api/videos/{id}/results` - Resultados da detecção
- `GET /api/videos/{id}/report` - Relatório do vídeo
- `GET /api/videos/{id}/report/csv` - Exportar relatório CSV
- `POST /api/videos/realtime/report` - Gerar relatório em tempo real

### Configuração
- `GET /config` - Configurações atuais
- `PUT /config` - Atualizar configurações
- `GET /classes` - Classes disponíveis do modelo
- `GET /classes/enabled` - Classes habilitadas
- `PUT /classes/enabled` - Atualizar classes habilitadas

### Histórico
- `GET /history` - Histórico de detecções

## 🎨 Interface Web

### Views Disponíveis
1. **Dashboard**: Monitoramento em tempo real com stream de vídeo
2. **Vídeos**: Upload e visualização de vídeos processados
3. **Relatório**: Análise estatística e relatórios dinâmicos
4. **Histórico**: Registro de detecções
5. **Status**: Monitoramento do sistema (FPS, GPU, uptime)
6. **Config**: Configurações do sistema

## 🔍 Sistema de Detecção

O sistema detecta:
- **Pessoas**: Detecção de pessoas no frame
- **EPIs Presentes**: EPIs detectados e associados a pessoas
- **EPIs Ausentes**: EPIs faltando (detecções virtuais "missing-*")

### Classes Suportadas (17 classes)
- person, helmet, safety-vest, gloves, glasses
- ear, ear-mufs, face, face-guard, face-mask-medical
- foot, tools, hands, head
- medical-suit, shoes, safety-suit

### Filtragem de EPIs Soltos
O sistema filtra automaticamente EPIs que não estão associados a pessoas, garantindo que apenas EPIs usados por pessoas sejam contabilizados.

## 📊 Relatórios Dinâmicos

Os relatórios são gerados dinamicamente baseados em todas as classes detectadas pelo modelo, sem hardcoding de EPIs específicos. Incluem:
- Estatísticas por classe (positivas e negativas)
- Compliance score
- Exportação para CSV

## 🛠️ Desenvolvimento

### Ferramentas de Desenvolvimento
- `dev/tools/` - Ferramentas de processamento de dados
- `dev/scripts/` - Scripts de treinamento
- `dev/tests/` - Testes

### Dados de Treinamento
Arquivados em `archive/training_data/` para referência.

## 📝 Notas

- **GPU Obrigatória**: Este projeto requer GPU CUDA para funcionar adequadamente
- **Modelo**: O modelo `best.pt` deve estar disponível em `models/best.pt` ou no caminho configurado
- **RTSP**: Configure `RTSP_URL` para usar câmeras IP via RTSP
- **Performance**: Ajuste `MODEL_CONF_THRESH` e thresholds por classe conforme necessário

## 📄 Licença

[Adicione informações de licença aqui]
