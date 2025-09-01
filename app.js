/**
 * Athena - Dashboard de Detecção de EPIs
 * Aplicação principal com Alpine.js
 */

document.addEventListener('alpine:init', () => {
    Alpine.data('athenaApp', () => ({
      // Estado da aplicação
      view: 'dashboard',
      connectionStatus: 'disconnected',
        isDetectionRunning: false,
        videoLoaded: false,
        videoError: false,

      
      // Dados das views
      stats: {
        com_capacete: 0,
        sem_capacete: 0,
        com_colete: 0,
        sem_colete: 0
      },
      
        history: [],
        currentPage: 1,
        historyLimit: 50,
        
        systemStatus: {
            fps: 0,
            gpu: 'N/A',
            uptime: '0s',
            last_update: 'N/A'
      },
      
      systemLogs: [],
      
      config: {
        conf_thresh: 0.5,
        iou: 0.45,
        max_detections: 50,
        batch_size: 1,
        enable_tracking: true,
        device_preference: 'auto',
        force_cpu_only: false
      },
      
      filters: {
        startDate: '',
        endDate: ''
      },
      

        
        // Sistema de detecções
        eventSource: null,
        websocket: null,
        lastFrameId: null,
        drawRequestId: null,
        
        // Gráfico
        reportChart: null,
        
        // Inicialização
        init() {
            console.log('Inicializando Athena Dashboard');
            
            // Configurar data inicial
            const today = new Date();
            this.filters.startDate = new Date(today.getFullYear(), today.getMonth(), 1).toISOString().split('T')[0];
            this.filters.endDate = today.toISOString().split('T')[0];
            
            // Carregar dados iniciais
            this.loadConfig();
            this.loadStatus();
            this.loadHistory();
            
            // Inicializar gráfico
            this.initChart();
            
            // Conectar com backend
            this.connectToBackend();
            
            // Inicializar stream
            this.initStream();
            
            console.log('Athena Dashboard inicializado');
      },

      // Navegação entre views
        navigate(viewName) {
            console.log('Navegando para:', viewName);
        this.view = viewName;
            
            // Carregar dados específicos da view
            switch(viewName) {
                case 'relatorio':
                    this.loadReportData();
                    break;
                case 'historico':
                    this.loadHistory();
                    break;
                case 'status':
                    this.loadStatus();
                    this.loadLogs();
                    break;
                case 'config':
                    this.loadConfig();
                    break;
            }
        },
        
        // Títulos das views
        getViewTitle() {
            const titles = {
                dashboard: 'Dashboard',
                relatorio: 'Relatório',
                historico: 'Histórico',
                status: 'Status do Sistema',
                config: 'Configurações'
            };
            return titles[this.view] || 'Dashboard';
        },
        
        // Sistema de detecções
        startDetection() {
            if (this.isDetectionRunning) return;
            
            this.isDetectionRunning = true;
            this.connectDetections();
            console.log('Detecção iniciada com sucesso');
        },
        
        stopDetection() {
            if (!this.isDetectionRunning) return;
            
            this.isDetectionRunning = false;
            this.disconnectDetections();
            console.log('Detecção parada');
        },
        
        // Conexão com backend
        connectToBackend() {
            // Tentar conectar via SSE primeiro
            this.connectSSE();
            
            // Fallback para WebSocket se SSE falhar
            setTimeout(() => {
                if (this.connectionStatus !== 'connected') {
                    this.connectWebSocket();
                }
            }, 2000);
        },
        
        // Conexão SSE
        connectSSE() {
            try {
                this.eventSource = new EventSource(CONFIG.API.BASE_URL + CONFIG.API.ENDPOINTS.DETECTIONS);
                
                this.eventSource.onopen = () => {
                    console.log('SSE conectado');
                    this.connectionStatus = 'connected';
                };
                
                this.eventSource.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.processDetectionData(data);
    } catch (error) {
                        console.error('Erro ao processar dados SSE:', error);
                    }
                };
                
                this.eventSource.onerror = (error) => {
                    console.error('Erro SSE:', error);
                    this.connectionStatus = 'disconnected';
                    this.eventSource.close();
                };
                
            } catch (error) {
                console.error('Erro ao conectar SSE:', error);
                this.connectionStatus = 'disconnected';
            }
        },
        
        // Conexão WebSocket
        connectWebSocket() {
            try {
                this.websocket = new WebSocket('ws://localhost:8000/ws/detections');
                
                this.websocket.onopen = () => {
                    console.log('WebSocket conectado');
                    this.connectionStatus = 'connected';
                };
                
                this.websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.processDetectionData(data);
                    } catch (error) {
                        console.error('Erro ao processar dados WebSocket:', error);
                    }
                };
                
                this.websocket.onerror = (error) => {
                    console.error('Erro WebSocket:', error);
                    this.connectionStatus = 'disconnected';
                };
                
                this.websocket.onclose = () => {
                    console.log('WebSocket desconectado');
                    this.connectionStatus = 'disconnected';
                };
                
    } catch (error) {
                console.error('Erro ao conectar WebSocket:', error);
                this.connectionStatus = 'disconnected';
            }
        },
        
        // Desconectar detecções
        disconnectDetections() {
            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }
            
            if (this.websocket) {
                this.websocket.close();
                this.websocket = null;
            }
            
            this.connectionStatus = 'disconnected';
        },
        
        // Processar dados de detecção
        processDetectionData(data) {
            if (!data || !data.frame_id) return;
            
            // Atualizar contadores se disponível
            if (data.epi_summary) {
                this.updateCounters(data.epi_summary);
            }
            
            // Desenhar boxes apenas se frame_id mudou
            if (this.lastFrameId !== data.frame_id) {
                this.lastFrameId = data.frame_id;
                this.drawBoxes(data.boxes);
            }
        },
        
        // Atualizar contadores
        updateCounters(stats) {
            this.stats = { ...stats };
            
            // Atualizar elementos do DOM
            document.getElementById('cnt-capacete-ok-value').textContent = stats.com_capacete || 0;
            document.getElementById('cnt-capacete-nok-value').textContent = stats.sem_capacete || 0;
            document.getElementById('cnt-colete-ok-value').textContent = stats.com_colete || 0;
            document.getElementById('cnt-colete-nok-value').textContent = stats.sem_colete || 0;
        },
        
        // Desenhar bounding boxes
        drawBoxes(boxes) {
            if (!boxes || !Array.isArray(boxes)) return;
            
            // Cancelar draw anterior se ainda não executou
            if (this.drawRequestId) {
                cancelAnimationFrame(this.drawRequestId);
            }
            
            // Agendar draw para próximo frame (limitar a 30 FPS)
            this.drawRequestId = requestAnimationFrame(() => {
                this.drawBoxesImmediate(boxes);
            });
        },
        
        // Desenhar boxes imediatamente
        drawBoxesImmediate(boxes) {
            const canvas = document.getElementById('overlay');
            const ctx = canvas.getContext('2d');
            
            if (!canvas || !ctx) return;
            
            // Ajustar tamanho do canvas
            const img = document.getElementById('mjpeg');
            if (img) {
                canvas.width = img.offsetWidth;
                canvas.height = img.offsetHeight;
            }
            
            // Limpar canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Agrupar boxes por track_id
            const trackGroups = {};
            boxes.forEach(box => {
                if (!trackGroups[box.track_id]) {
                    trackGroups[box.track_id] = [];
                }
                trackGroups[box.track_id].push(box);
            });
            
            // Desenhar boxes para cada track
            Object.values(trackGroups).forEach(trackBoxes => {
                const personBox = trackBoxes.find(box => box.label === 'person');
                const helmetBox = trackBoxes.find(box => box.label === 'helmet');
                const vestBox = trackBoxes.find(box => box.label === 'vest');
                
                if (personBox) {
                    // Determinar cor baseada nos EPIs
                    let color = '#ef4444'; // Vermelho - sem EPIs
                    
                    if (helmetBox && vestBox) {
                        color = '#10b981'; // Verde - com ambos
                    } else if (helmetBox || vestBox) {
                        color = '#d5b481'; // Amarelo-bege - com um
                    }
                    
                    // Desenhar box da pessoa
                    this.drawBox(ctx, personBox, color, 'person');
                    
                    // Desenhar boxes dos EPIs
                    if (helmetBox) {
                        this.drawBox(ctx, helmetBox, '#10b981', 'helmet');
                    }
                    if (vestBox) {
                        this.drawBox(ctx, vestBox, '#10b981', 'vest');
                    }
                }
            });
        },
        
        // Desenhar box individual
        drawBox(ctx, box, color, label) {
            const x = box.x;
            const y = box.y;
            const w = box.w;
            const h = box.h;
            const conf = box.conf;
            
            // Desenhar retângulo
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            
            // Desenhar rótulo
            ctx.fillStyle = color;
            ctx.fillRect(x, y - 20, 120, 20);
            
            // Texto do rótulo
            ctx.fillStyle = '#ffffff';
            ctx.font = '12px Arial';
            ctx.fillText(`${label}:${Math.round(conf * 100)}%`, x + 5, y - 5);
        },
        
        // Inicializar stream
        initStream() {
            const img = document.getElementById('mjpeg');
            if (img) {
                img.src = '/stream.mjpg';
            }
        },
        
        // Eventos do vídeo
        onVideoLoad() {
            this.videoLoaded = true;
            this.videoError = false;
            console.log('Vídeo carregado');
        },
        
        onVideoError() {
            this.videoLoaded = false;
            this.videoError = true;
            console.error('Erro ao carregar vídeo');
        },
        
        // Snapshot
        async takeSnapshot() {
            try {
                const response = await fetch('/snapshot', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                const result = await response.json();
                
                if (result.saved) {
                    console.log('Snapshot capturado com sucesso');
                    
                    // Mostrar preview se disponível
                    if (result.url) {
                        setTimeout(() => {
                            window.open(result.url, '_blank');
                        }, 1000);
                    }
      } else {
                    console.error('Erro ao capturar snapshot');
                }
                
    } catch (error) {
                console.error('Erro ao capturar snapshot:', error);
                console.error('Erro na comunicação com o servidor');
            }
        },
        
        // Carregar configurações
        async loadConfig() {
            try {
                const response = await fetch('/config');
                const config = await response.json();
                this.config = { ...this.config, ...config };
    } catch (error) {
                console.error('Erro ao carregar configurações:', error);
                console.error('Erro ao carregar configurações');
            }
        },
        
        // Salvar configurações
        async saveConfig() {
            try {
                const response = await fetch('/config', {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.config)
                });
                
                if (response.ok) {
                    console.log('Configurações salvas com sucesso');
                } else {
                    throw new Error('Falha ao salvar');
                }
                
    } catch (error) {
                console.error('Erro ao salvar configurações:', error);
                console.error('Erro ao salvar configurações');
            }
        },
        
        // Carregar status
        async loadStatus() {
            try {
                const response = await fetch('/status');
                const status = await response.json();
                
                this.systemStatus = {
                    fps: status.fps || 0,
                    gpu: status.gpu || 'N/A',
                    uptime: this.formatUptime(status.uptime_s || 0),
                    last_update: new Date().toLocaleTimeString(),
                    status: status.status || 'offline',
                    version: status.version || 'N/A',
                    api_version: status.api_version || 'N/A',
                    model_loaded: status.model_loaded || false
                };
                
                // Atualizar status de conexão
                this.connectionStatus = status.status || 'offline';
                
    } catch (error) {
                console.error('Erro ao carregar status:', error);
                this.systemStatus.last_update = 'Erro ao carregar';
            }
        },
        
        // Carregar histórico
        async loadHistory(direction = null) {
            try {
                if (direction === 'next') {
                    this.currentPage++;
                } else if (direction === 'prev') {
                    this.currentPage = Math.max(1, this.currentPage - 1);
                }
                
                const offset = (this.currentPage - 1) * this.historyLimit;
                
                // Construir query com filtros
                const params = new URLSearchParams({
                    offset: offset.toString(),
                    limit: this.historyLimit.toString()
                });
                
                if (this.filters.startDate) {
                    params.append('start_date', this.filters.startDate);
                }
                if (this.filters.endDate) {
                    params.append('end_date', this.filters.endDate);
                }
                
                const response = await fetch(`/history?${params.toString()}`);
                const data = await response.json();
                
                this.history = data.data || [];
                
    } catch (error) {
                console.error('Erro ao carregar histórico:', error);
                console.error('Erro ao carregar histórico');
            }
        },
        
        // Carregar logs do sistema
        async loadLogs() {
            try {
                const response = await fetch('/logs?limit=20');
                if (response.ok) {
                    const data = await response.json();
                    this.systemLogs = data.logs || [];
                }
            } catch (error) {
                console.error('Erro ao carregar logs:', error);
            }
        },
        
        // Carregar dados para relatórios
        async loadReportData() {
            try {
                const response = await fetch(CONFIG.API.BASE_URL + '/api/reports/daily');
                if (response.ok) {
                    const data = await response.json();
                    this.updateChart(data);
                } else {
                    console.error('Erro ao carregar dados do relatório');
                }
            } catch (error) {
                console.error('Erro ao carregar dados do relatório:', error);
            }
        },
        
        // Inicializar gráfico
        initChart() {
            const ctx = document.getElementById('reportChart');
            if (!ctx) return;
            
            this.reportChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Com Capacete', 'Sem Capacete', 'Com Colete', 'Sem Colete'],
                    datasets: [{
                        label: 'Detecções',
                        data: [0, 0, 0, 0],
                        backgroundColor: [
                            '#1e40af', // Azul ATHENA
                            '#dc2626', // Vermelho
                            '#1e40af', // Azul ATHENA
                            '#dc2626'  // Vermelho
                        ],
                        borderColor: [
                            '#1d4ed8',
                            '#dc2626',
                            '#1d4ed8',
                            '#dc2626'
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: '#374151'
                            },
                            ticks: {
                                color: '#9ca3af'
                            }
                        },
                        x: {
                            grid: {
                                color: '#374151'
                            },
                            ticks: {
                                color: '#9ca3af'
                            }
                        }
          }
        }
      });
        },
        
        // Atualizar gráfico
        updateChart(data = null) {
            if (!this.reportChart) return;
            
            let chartData;
            
            if (data && Array.isArray(data)) {
                // Usar dados do relatório se disponível
                const summary = {
                    com_capacete: 0,
                    sem_capacete: 0,
                    com_colete: 0,
                    sem_colete: 0
                };
                
                data.forEach(item => {
                    summary.com_capacete += item.com_capacete || 0;
                    summary.sem_capacete += item.sem_capacete || 0;
                    summary.com_colete += item.com_colete || 0;
                    summary.sem_colete += item.sem_colete || 0;
                });
                
                chartData = [
                    summary.com_capacete,
                    summary.sem_capacete,
                    summary.com_colete,
                    summary.sem_colete
                ];
            } else {
                // Usar dados das estatísticas atuais
                chartData = [
                    this.stats.com_capacete || 0,
                    this.stats.sem_capacete || 0,
                    this.stats.com_colete || 0,
                    this.stats.sem_colete || 0
                ];
            }
            
            this.reportChart.data.datasets[0].data = chartData;
            this.reportChart.update();
        },
        
        // Formatar uptime
        formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            
            if (hours > 0) {
                return `${hours}h ${minutes}m ${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
      } else {
                return `${secs}s`;
            }
        },
        

        

        
        // Atualizar contadores periodicamente (fallback)
        startStatsPolling() {
            setInterval(async () => {
                if (this.connectionStatus !== 'connected') {
                    try {
                        const response = await fetch('/stats');
                        const stats = await response.json();
                        this.updateCounters(stats);
                    } catch (error) {
                        console.error('Erro ao carregar stats:', error);
                    }
                }
            }, 2000);
        }
    }));
});

// Inicializar quando DOM estiver pronto
document.addEventListener('DOMContentLoaded', () => {
    // Iniciar polling de stats
    if (window.Alpine && window.Alpine.data('athenaApp')) {
        const app = document.querySelector('[x-data="athenaApp"]').__x.$data;
        app.startStatsPolling();
    }
});
