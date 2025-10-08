# 🛡️ Athena - Sistema de Detecção de EPIs

Sistema inteligente de detecção de Equipamentos de Proteção Individual (EPIs) usando YOLOv5 e interface web moderna.

## ✨ Características

- **Detecção em Tempo Real**: Identifica capacetes e coletes de segurança
- **Interface Web Responsiva**: Dashboard moderno com Alpine.js e Tailwind CSS
- **Modelo Treinado**: Utiliza modelo YOLOv5 customizado para detecção de EPIs
- **Stream de Vídeo**: Transmissão ao vivo da webcam com overlay de detecções
- **Histórico e Relatórios**: Sistema completo de registro e análise
- **Configurações Flexíveis**: Ajuste de parâmetros de detecção em tempo real

## 🚀 Início Rápido

### Pré-requisitos

- Python 3.8+
- Webcam funcional
- CUDA (opcional, para aceleração GPU)

### Instalação

1. **Clone o repositório**
```bash
git clone <repository-url>
cd Atena
```

2. **Instale as dependências**
```bash
pip install -r requirements.txt
```

3. **Verifique se há modelos treinados**
```bash
ls yolov5/runs/train/
```

### Execução

1. **Inicie o backend**
```bash
python start_api.py
```

2. **Abra o frontend**
```bash
# Abra o arquivo frontend/index.html no navegador
# Ou use um servidor local simples:
cd frontend
python -m http.server 8080
# Acesse: http://localhost:8080
```

3. **Acesse a API**
```
http://localhost:8000
```

## 🔧 Configuração

### Variáveis de Ambiente

```bash
# Modelo
MODEL_PATH=athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt

# API
API_HOST=0.0.0.0
API_PORT=8000

# Vídeo
VIDEO_SOURCE=0  # 0 = webcam padrão
VIDEO_FPS=30
VIDEO_WIDTH=640
VIDEO_HEIGHT=480

# Detecção
MODEL_CONF_THRESH=0.35
MODEL_IOU_THRESH=0.45
MODEL_MAX_DETECTIONS=50
```

### Parâmetros do Modelo

- **Confidence Threshold**: Limite mínimo de confiança para detecções
- **IoU Threshold**: Limite de sobreposição para supressão de detecções duplicadas
- **Max Detections**: Número máximo de detecções por frame

## 📁 Estrutura do Projeto

```
Atena/
├── backend/                 # Backend Python/FastAPI
│   ├── api.py             # API principal
│   ├── detection.py       # Sistema de detecção
│   ├── config.py          # Configurações
│   └── utils.py           # Utilitários
├── frontend/               # Interface web
│   ├── index.html         # Página principal
│   ├── js/                # JavaScript
│   └── styles/            # CSS
├── yolov5/                 # Framework YOLOv5
│   └── runs/train/        # Modelos treinados
├── start_api.py           # Script de inicialização
└── requirements.txt        # Dependências Python
```

## 🎯 Modelos Treinados

O sistema automaticamente detecta e usa o melhor modelo disponível em `yolov5/runs/train/`.

### Modelo Atual
- **Nome**: `athena_phase1_tesla_t4` (Fase 1 Otimizado)
- **Arquivo**: `best.pt`
- **Tamanho**: ~14MB
- **Classes**: 17 classes de EPIs + person
- **Performance**: 99.58% taxa de detecção, 84.18% confiança média

## 🌐 Endpoints da API

### Principais
- `GET /` - Status da API
- `GET /health` - Verificação de saúde
- `GET /stream.mjpg` - Stream de vídeo MJPEG
- `GET /events/detections` - SSE para detecções
- `GET /stats` - Estatísticas atuais
- `POST /snapshot` - Capturar snapshot

### Histórico
- `GET /history` - Histórico de detecções
- `GET /history/stats` - Estatísticas do histórico
- `GET /history/trend` - Tendências de compliance

### Configuração
- `GET /config` - Configurações atuais
- `PUT /config` - Atualizar configurações

## 🎨 Interface Web

### Views Disponíveis
1. **Dashboard**: Monitoramento em tempo real
2. **Relatório**: Análise estatística
3. **Histórico**: Registro de detecções
4. **Status**: Performance do sistema
5. **Configurações**: Ajuste de parâmetros

### Funcionalidades
- Stream de vídeo ao vivo
- Overlay de detecções em tempo real
- Contadores de EPIs detectados
- Indicador de status de conexão
- Sistema de snapshots
- Navegação responsiva

## 🔍 Solução de Problemas

### Erros Comuns

#### 1. Webcam não inicializa
```bash
# Verifique se a webcam está disponível
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

#### 2. Modelo não carrega
```bash
# Verifique se o arquivo existe
ls -la athena_training_2phase_optimized/models/phase1_complete/athena_phase1_tesla_t4/weights/best.pt
```

#### 3. Frontend não conecta
```bash
# Verifique se a API está rodando
curl http://localhost:8000/health
```

#### 4. Erros Alpine.js
- Certifique-se de que o Alpine.js está carregando com `defer`
- Verifique o console do navegador para erros JavaScript

### Logs

Os logs são exibidos no terminal onde a API está rodando. Use `LOG_LEVEL=DEBUG` para mais detalhes.

## 📊 Performance

### Métricas Típicas
- **FPS**: 25-30 (dependendo do hardware)
- **Latência**: <100ms
- **Precisão**: >90% (com modelo treinado)
- **Uso de Memória**: ~500MB-1GB

### Otimizações
- Use GPU CUDA se disponível
- Ajuste `VIDEO_FPS` conforme necessário
- Configure `MODEL_MAX_DETECTIONS` adequadamente

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## 🙏 Agradecimentos

- [Ultralytics](https://github.com/ultralytics/yolov5) - Framework YOLOv5
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web Python
- [Alpine.js](https://alpinejs.dev/) - Framework JavaScript minimalista
- [Tailwind CSS](https://tailwindcss.com/) - Framework CSS utilitário

## 📞 Suporte

Para suporte ou dúvidas:
- Abra uma issue no GitHub
- Consulte a documentação da API em `/docs`
- Verifique os logs do sistema

---

**Athena** - Protegendo vidas através da tecnologia 🤖🛡️
