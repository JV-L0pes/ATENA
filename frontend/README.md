# Athena Dashboard - Sistema de Detecção de EPIs

## Visão Geral

O Athena Dashboard é uma interface web moderna e responsiva para monitoramento em tempo real de detecção de EPIs (Equipamentos de Proteção Individual) como capacetes e coletes. O sistema utiliza YOLOv5 para detecção e oferece uma interface intuitiva com tema escuro.

## Características Principais

### 🎨 **Interface Moderna**
- **Tema escuro** com Tailwind CSS
- **Layout responsivo** (desktop prioritário, mobile aceitável)
- **Sidebar fixa** à esquerda com navegação
- **Player principal** ao centro para stream de vídeo
- **Rodapé** com contadores em tempo real

### 📊 **Funcionalidades**
- **Dashboard**: Visualização do stream + detecções em tempo real
- **Relatório**: Gráficos Chart.js com filtros de data
- **Histórico**: Tabela paginada com histórico de detecções
- **Status**: Monitoramento do sistema (FPS, GPU, uptime)
- **Config**: Configurações do sistema (thresholds, IOU, etc.)

### 🔄 **Integração em Tempo Real**
- **SSE (Server-Sent Events)** para detecções
- **WebSocket** como fallback
- **Stream MJPEG** para vídeo
- **Contadores atualizados** automaticamente

### 📸 **Recursos Avançados**
- **Snapshot** com botão redondo dedicado
- **Bounding boxes** coloridos por status dos EPIs
- **Sistema de toast** para notificações
- **Interface responsiva** para diferentes dispositivos

## Estrutura dos Arquivos

```
frontend/
├── index.html          # Interface principal (único arquivo HTML)
├── js/
│   ├── app.js         # Lógica principal da aplicação
│   ├── api.js         # Comunicação com backend
│   ├── config.js      # Configurações
│   ├── detection.js   # Sistema de detecção
│   └── utils.js       # Funções utilitárias
└── README.md          # Este arquivo
```

## Como Usar

### 1. **Abrir o Dashboard**
Simplesmente abra o arquivo `index.html` em qualquer navegador moderno. Não é necessário servidor web ou build.

### 2. **Navegação**
- **Dashboard**: Visualização principal com stream e detecções
- **Relatório**: Gráficos e análises com filtros de data
- **Histórico**: Tabela com histórico paginado
- **Status**: Monitoramento do sistema
- **Config**: Configurações e parâmetros

### 3. **Controles de Detecção**
- **▶️ Iniciar**: Ativa o sistema de detecção
- **⏸️ Ativo**: Indica que a detecção está rodando
- **⏹️ Parar**: Para o sistema de detecção
- **📸 Snapshot**: Captura imagem atual

### 4. **Contadores em Tempo Real**
O rodapé exibe 4 contadores principais:
- ✅ **Com Capacete** (verde)
- ❌ **Sem Capacete** (vermelho)
- ✅ **Com Colete** (verde)
- ❌ **Sem Colete** (vermelho)

## Endpoints do Backend

O dashboard espera os seguintes endpoints:

### **Stream de Vídeo**
```
GET /stream.mjpg
```
Retorna stream MJPEG para exibição.

### **Detecções em Tempo Real**
```
GET /events/detections
```
SSE para receber detecções em tempo real.

**Payload esperado:**
```json
{
  "frame_id": 12345,
  "boxes": [
    { "x": 120, "y": 80, "w": 140, "h": 260, "label": "person", "conf": 0.88, "track_id": 7 },
    { "x": 150, "y": 60, "w": 80, "h": 80, "label": "helmet", "conf": 0.91, "track_id": 7 },
    { "x": 150, "y": 210, "w": 100, "h": 120, "label": "vest", "conf": 0.87, "track_id": 7 }
  ],
  "epi_summary": { "com_capacete": 5, "sem_capacete": 1, "com_colete": 4, "sem_colete": 2 }
}
```

### **Snapshot**
```
POST /snapshot
```
Captura snapshot da câmera.

### **Estatísticas**
```
GET /stats
```
Retorna contadores agregados.

### **Histórico**
```
GET /history?offset=0&limit=50
```
Histórico paginado de detecções.

### **Status do Sistema**
```
GET /status
```
Informações do sistema (FPS, GPU, uptime).

### **Configurações**
```
GET /config
PUT /config
```
Leitura e escrita de configurações.

## Regras de Cores dos Boxes

### **Verde** 🟢
- **Pessoa com capacete E colete**
- Status: Conforme

### **Amarelo** 🟡
- **Pessoa com capacete OU colete** (falta um)
- Status: Parcialmente conforme

### **Vermelho** 🔴
- **Pessoa sem capacete E sem colete**
- Status: Não conforme

## Tecnologias Utilizadas

- **HTML5** + **CSS3** + **JavaScript ES6+**
- **Tailwind CSS** (via CDN) - Tema escuro
- **Alpine.js** (via CDN) - Reatividade
- **Chart.js** (via CDN) - Gráficos
- **Sem bundler** - Tudo via CDN

## Desenvolvimento

### **Conexão com Backend**
O frontend se conecta automaticamente com o backend via WebSocket para receber dados de detecção em tempo real.

### **Personalização**
- **Cores**: Edite `config.js` → `DETECTION.COLORS`
- **Endpoints**: Modifique `api.js` → `endpoints`
- **UI**: Personalize classes Tailwind no `index.html`

### **Debug**
- Abra o console do navegador para logs detalhados
- Use `Utils.log()` para logging estruturado
- Verifique status da conexão na sidebar

## Compatibilidade

### **Navegadores Suportados**
- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

### **Funcionalidades**
- ✅ SSE (Server-Sent Events)
- ✅ WebSocket (fallback)
- ✅ Canvas 2D
- ✅ Fetch API
- ✅ ES6+ Features

## Troubleshooting

### **Stream não carrega**
- Verifique se `/stream.mjpg` está acessível
- Confirme se o backend está rodando
- Verifique logs no console

### **Detecções não aparecem**
- Teste conexão SSE em `/events/detections`
- Verifique se o backend está enviando dados
- Verifique se o backend está rodando

### **Performance lenta**
- Reduza FPS máximo em `config.js`
- Verifique se há muitas detecções simultâneas
- Monitore uso de CPU/GPU

## Contribuição

1. Mantenha a estrutura modular
2. Use funções utilitárias do `utils.js`
3. Siga padrões de nomenclatura
4. Teste em diferentes navegadores
5. Documente novas funcionalidades

## Licença

Este projeto é parte do sistema Athena para detecção de EPIs.

---

**Desenvolvido com ❤️ para segurança no trabalho**
