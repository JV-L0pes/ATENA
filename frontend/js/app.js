/* Athena Dashboard - Aplicação Principal Alpine.js */

// Configuração global
const CONFIG = {
    API: {
        BASE_URL: window.location.origin,
        ENDPOINTS: {
            HEALTH: '/health',
            STATUS: '/status',
            DETECTIONS: '/events/detections',
            STREAM: '/stream.mjpg',
            CONFIG: '/config',
            CAMERA_CONFIG: '/camera/config',
            HISTORY: '/history',
            CLASSES: '/classes',
            VIDEOS_UPLOAD: '/api/videos/upload',
            VIDEOS_LIST: '/api/videos/list',
            VIDEOS_STATUS: '/api/videos',
            VIDEOS_RESULTS: '/api/videos',
            VIDEOS_DOWNLOAD: '/api/videos',
            VIDEOS_FORMATS: '/api/videos/formats'
        }
    },
    UI: {
        REFRESH_INTERVAL: 5000,
        CHART_UPDATE_INTERVAL: 2000,
        SSE_RECONNECT_DELAY: 3000,
        STREAM_TIMEOUT: 10000
    },
    DETECTION: {
        CONFIDENCE_THRESHOLD: 0.25,
        MAX_DETECTIONS: 100,
        BOX_COLOR: '#00ff00',
        LABEL_BACKGROUND: 'rgba(0, 0, 0, 0.8)',
        MAX_FRAME_WIDTH: 640,
        TARGET_FPS: 2,
        JPEG_QUALITY: 0.5
    }
};

// Função principal Alpine.js
function athenaApp() {
    return {
        // Estado da aplicação
        activeView: 'dashboard',
      connectionStatus: 'disconnected',
        videoLoaded: false,
        isDetectionRunning: false,
        configSaving: false,
        initialized: false,
        chartUpdating: false,
        // Modo de exibição das detecções: 'negatives' | 'positives' | 'both'
        detectionViewMode: 'negatives',
      
        // Dados
      stats: {
        com_capacete: 0,
        sem_capacete: 0,
        com_colete: 0,
        sem_colete: 0,
        total_pessoas: 0
      },
      
      // Histórico inicializado como array vazio
        history: [],
        
        // Informações das classes
        classesInfo: {
            class_names: [],
            active_classes: [],
            colors: {},
            compliance_mapping: {},
            total_classes: 0
        },
        
        systemStatus: {
            fps: 0,
            uptime_s: 0,
            version: '1.0.0',
            api_version: '1.0.0',
            status: 'offline'
        },
      
      config: {
            conf_thresh: 0.35,
        iou: 0.45,
        max_detections: 50,
        batch_size: 1,
            enable_tracking: true
        },

        cameraConfig: {
            type: 'usb', // 'usb' ou 'ip'
            usb: {
                index: 0
            },
            ip: {
                url: '',
                username: '',
                password: '',
                timeout: 10
            }
        },

        cameraTesting: false,
        cameraTestResult: null,
        
        // Sistema de vídeos em tempo real
        currentVideo: null,
        currentVideoUrl: null,
        videoDimensions: { width: 0, height: 0 },
        detectionActive: false,
        detectionFPS: 0,
        currentDetections: [],
        realtimeStats: {
            total_pessoas: 0,
            com_capacete: 0,
            com_colete: 0,
            compliance_score: 0
        },
        detectionInterval: null,
        lastDetectionTime: 0,
        inFlightDetection: false,
        lastFpsTick: 0,
        framesThisSecond: 0,
        videoPlayer: null,
        videoOverlay: null,
        lastSentFrameSize: { width: 0, height: 0 },
        // WebSocket métricas/adaptação
        wsLastSendAt: 0,
        wsRTTms: 0,
        dynamicTargetFps: 2,
        dynamicJpegQuality: 0.5,
        
        // Sistema de detecções
        eventSource: null,
        reportChart: null,
        lastDetections: null,  // Para evitar redesenhar caixas desnecessariamente
        lastFilteredCount: 0,  // Para logs otimizados
        
        // Inicialização
        init() {
            // Proteção contra múltiplas inicializações
            if (this.initialized) return;
            this.initialized = true;
            
            // Inicialização silenciosa para performance
            
            // Carregar dados iniciais
            this.loadSystemStatus();
            this.loadConfig();
            this.loadCameraConfig();
            this.loadHistory();
            
            // Conectar com backend
            this.connectToBackend();
            
            // Timer para limpar caixas antigas
            setInterval(() => {
                this.clearOldDetections();
            }, 2000); // A cada 2 segundos
            
            // Gráfico desabilitado temporariamente
            
            // Verificar se o stream está funcionando após 3 segundos
            setTimeout(() => {
                if (!this.videoLoaded) {
                    console.log('🔄 Forçando carregamento do stream...');
                    this.videoLoaded = true;
                }
            }, 3000);
            
            console.log('✅ Athena Dashboard inicializado');

            // redimensionar overlay quando janela muda / fullscreen
            window.addEventListener('resize', () => this.resizeOverlayToVideo());
            document.addEventListener('fullscreenchange', () => this.resizeOverlayToVideo());
        },

        // Navegação
        setActiveView(viewName) {
            console.log('📍 Navegando para:', viewName);
            this.activeView = viewName;
            
            // Carregar dados específicos da view
            switch(viewName) {
                case 'relatorio':
                    this.loadReportData();
                    break;
                case 'historico':
                    this.loadHistory();
                    break;
                case 'status':
                    this.loadSystemStatus();
                    break;
                case 'config':
                    this.loadConfig();
                    this.loadCameraConfig();
                    break;
                case 'classes':
                    this.loadClassesInfo();
                    break;
                case 'videos':
                    // Não precisa carregar nada - detecção em tempo real
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
                config: 'Configurações',
                classes: 'Classes de Detecção',
                videos: 'Player de Vídeo com IA'
            };
            return titles[this.activeView] || 'Dashboard';
        },
        
        // Conexão com backend
        connectToBackend() {
            this.connectSSE();
            this.checkStreamStatus();
        },

        // Verificar status do stream
        checkStreamStatus() {
            const img = document.getElementById('mjpeg');
            if (img) {
                // Verificar se a imagem está carregando
                img.onload = () => {
                    console.log('✅ Stream carregado com sucesso');
                    this.videoLoaded = true;
                };
                
                img.onerror = () => {
                    console.log('❌ Erro ao carregar stream');
                    this.videoLoaded = false;
                };
                
                // Timeout de segurança
            setTimeout(() => {
                    if (!this.videoLoaded) {
                        console.log('🔄 Timeout - assumindo que stream está funcionando');
                        this.videoLoaded = true;
                    }
                }, 5000);
            }
        },
        
        // Conexão SSE
        connectSSE() {
            try {
                this.eventSource = new EventSource(CONFIG.API.BASE_URL + CONFIG.API.ENDPOINTS.DETECTIONS);
                
                this.eventSource.onopen = () => {
                    console.log('✅ SSE conectado');
                    this.connectionStatus = 'connected';
                };
                
                this.eventSource.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        this.processDetectionData(data);
    } catch (error) {
                        console.error('❌ Erro ao processar dados SSE:', error);
                    }
                };
                
                this.eventSource.onerror = (error) => {
                    console.error('❌ Erro SSE:', error);
                    this.connectionStatus = 'disconnected';
                    this.eventSource.close();
                    
                    // Tentar reconectar após 5 segundos
                    setTimeout(() => {
                        this.connectSSE();
                    }, 5000);
                };
                
            } catch (error) {
                console.error('❌ Erro ao conectar SSE:', error);
                this.connectionStatus = 'disconnected';
            }
        },
        
        // Processar dados de detecção
        processDetectionData(data) {
            if (!data || !data.frame_id) return;
            
            // Atualizar estatísticas se disponível
            if (data.epi_summary) {
                this.stats = { ...data.epi_summary };
                this.isDetectionRunning = true;
            }
            
            // Processar detecções (boxes) se disponível
            if (data.boxes && Array.isArray(data.boxes)) {
                this.currentDetections = data.boxes;
                
                // Desenhar caixas de detecção
                this.drawDetectionBoxes(data.boxes);
            }
            
            // Gráfico desabilitado temporariamente
            // if (this.activeView === 'relatorio' && this.reportChart) {
            //     this.updateChart();
            // }
        },

        // Desenhar caixas de detecção
        drawDetectionBoxes(detections) {
            // Verificar se as detecções mudaram significativamente
            if (this.detectionsChanged(detections)) {
                // Limpar detecções anteriores apenas se mudaram
                this.clearDetectionBoxes();
                this.lastDetections = JSON.stringify(detections);
            } else {
                // Detecções não mudaram, não redesenhar
                return;
            }
            
            if (!detections || detections.length === 0) return;
            
            // Criar container para as caixas se não existir
            let detectionContainer = document.getElementById('detection-container');
            if (!detectionContainer) {
                detectionContainer = document.createElement('div');
                detectionContainer.id = 'detection-container';
                detectionContainer.style.position = 'absolute';
                detectionContainer.style.top = '0';
                detectionContainer.style.left = '0';
                detectionContainer.style.width = '100%';
                detectionContainer.style.height = '100%';
                detectionContainer.style.pointerEvents = 'none';
                detectionContainer.style.zIndex = '10';
                
                const videoContainer = document.getElementById('mjpeg').parentElement;
                videoContainer.style.position = 'relative';
                videoContainer.appendChild(detectionContainer);
            }
            
            // Filtragem conforme modo selecionado - LÓGICA CORRIGIDA
            const isMissing = d => {
                // AUSÊNCIAS = Detecções virtuais (pessoa sem capacete detectado pelo modelo)
                return Array.isArray(d.missing_epis) && d.missing_epis.includes('helmet');
            };
            const isPositiveEPI = d => {
                // DETECÇÕES = Detecções reais do modelo de IA
                const cn = d.class_name;
                return ['helmet','hardhat','person','head','face','ear','ear-mufs','glasses','gloves','hands'].includes(cn);
            };
            const isCompliant = d => {
                // DETECÇÕES COMPLIANT = Detecções reais do modelo que são compliant
                const cn = d.class_name;
                return ['helmet','hardhat','ear','ear-mufs','glasses','gloves'].includes(cn) || 
                       (cn === 'person' && Array.isArray(d.missing_epis) && d.missing_epis.length === 0);
            };
            
            let filtered = detections;
            if (this.detectionViewMode === 'negatives') {
                // Mostrar apenas AUSÊNCIAS (detecções virtuais - pessoa sem capacete)
                filtered = detections.filter(isMissing);
                console.log(`🔍 Modo 'ausências': ${filtered.length} detecções virtuais (pessoas sem capacete)`);
            } else if (this.detectionViewMode === 'positives') {
                // Mostrar apenas DETECÇÕES REAIS do modelo de IA
                filtered = detections.filter(isPositiveEPI);
                console.log(`🔍 Modo 'detecções': ${filtered.length} detecções reais do modelo de IA`);
            } else {
                // Mostrar todas as detecções (virtuais + reais)
                filtered = detections;
                console.log(`🔍 Modo 'ambos': ${filtered.length} detecções (virtuais + reais)`);
            }

            // Log apenas quando há mudanças significativas
            if (filtered.length !== this.lastFilteredCount) {
                console.log(`🔍 ${filtered.length} detecções ativas`);
                this.lastFilteredCount = filtered.length;
            }

            // Desenhar cada detecção
            filtered.forEach((detection, index) => {
                // Verificar se a detecção tem os campos necessários
                if (!detection.bbox || !Array.isArray(detection.bbox) || detection.bbox.length < 4) {
                    console.warn('⚠️ Detecção inválida:', detection);
                    return;
                }
                
                const box = detection.bbox;
                const confidence = detection.confidence || 0;
                const className = detection.class_name || 'unknown';
                const classId = detection.class_id || 0;
                
                // Criar elemento da caixa
                const boxElement = document.createElement('div');
                boxElement.className = 'detection-box';
                boxElement.style.position = 'absolute';
                boxElement.style.transition = 'opacity 0.3s ease-in-out';  // Transição suave
                boxElement.style.opacity = '1';
                
                // Cor por compliance: pessoa sem capacete => vermelho; caso contrário, usa verde
                const hasMissingHelmet = Array.isArray(detection.missing_epis) && detection.missing_epis.includes('helmet');
                const isCompliant = detection.compliant === true || !hasMissingHelmet;
                const isMissingHelmetClass = className.startsWith('missing-helmet');
                
                // Cores baseadas no status do capacete
                let borderColor, bgColor, labelText;
                
                if (isMissingHelmetClass) {
                    // Detecção virtual de capacete faltando = VERMELHO
                    borderColor = '#ff0000';
                    bgColor = 'rgba(255, 0, 0, 0.2)';
                    labelText = `Falta: Capacete`;
                } else if (hasMissingHelmet) {
                    // Pessoa sem capacete = VERMELHO
                    borderColor = '#ff0000';
                    bgColor = 'rgba(255, 0, 0, 0.15)';
                    labelText = `Pessoa - Sem Capacete`;
                } else if (isCompliant) {
                    // Pessoa com capacete = VERDE
                    borderColor = '#00ff00';
                    bgColor = 'rgba(0, 255, 0, 0.15)';
                    labelText = `Pessoa - Com Capacete`;
                } else {
                    // Outras detecções (face, head, etc.) = AZUL
                    borderColor = '#0080ff';
                    bgColor = 'rgba(0, 128, 255, 0.15)';
                    labelText = `${className} (${(confidence * 100).toFixed(1)}%)`;
                }
                
                boxElement.style.border = `2px solid ${borderColor}`;
                boxElement.style.backgroundColor = bgColor;
                boxElement.style.pointerEvents = 'none';
                boxElement.style.zIndex = '11';
                
                // Adicionar ID único para tracking
                boxElement.id = `detection-${index}-${Date.now()}`;
                
                // Calcular posição baseada no tamanho da imagem
                const img = document.getElementById('mjpeg');
                if (img && img.naturalWidth && img.naturalHeight) {
                    const scaleX = img.clientWidth / img.naturalWidth;
                    const scaleY = img.clientHeight / img.naturalHeight;
                    
                    const x = box[0] * scaleX;
                    const y = box[1] * scaleY;
                    const width = (box[2] - box[0]) * scaleX;
                    const height = (box[3] - box[1]) * scaleY;
                    
                    boxElement.style.left = `${x}px`;
                    boxElement.style.top = `${y}px`;
                    boxElement.style.width = `${width}px`;
                    boxElement.style.height = `${height}px`;
                    
                    // Log removido para performance
                } else {
                    console.warn('⚠️ Imagem não encontrada ou sem dimensões');
                    return;
                }
                
                // Criar label
                const label = document.createElement('div');
                label.style.position = 'absolute';
                label.style.top = '-20px';
                label.style.left = '0';
                label.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
                label.style.color = borderColor;
                label.style.padding = '2px 6px';
                label.style.fontSize = '12px';
                label.style.fontWeight = 'bold';
                label.style.borderRadius = '3px';
                label.style.whiteSpace = 'nowrap';
                label.textContent = labelText;
                
                boxElement.appendChild(label);
                detectionContainer.appendChild(boxElement);
            });
            
            // Log removido para performance
        },

        // Verificar se as detecções mudaram significativamente
        detectionsChanged(newDetections) {
            if (!this.lastDetections) return true;
            
            try {
                const lastDetections = JSON.parse(this.lastDetections);
                
                // Comparar número de detecções
                if (newDetections.length !== lastDetections.length) {
                    return true;
                }
                
                // Comparar posições das caixas (com tolerância)
                for (let i = 0; i < newDetections.length; i++) {
                    const newDet = newDetections[i];
                    const lastDet = lastDetections[i];
                    
                    if (!newDet.bbox || !lastDet.bbox) return true;
                    
                    // Tolerância de 20 pixels para movimento
                    const tolerance = 20;
                    for (let j = 0; j < 4; j++) {
                        if (Math.abs(newDet.bbox[j] - lastDet.bbox[j]) > tolerance) {
                            return true;
                        }
                    }
                    
                    // Comparar classe e confiança
                    if (newDet.class_name !== lastDet.class_name) return true;
                    if (Math.abs(newDet.confidence - lastDet.confidence) > 0.1) return true;
                }
                
                return false; // Não mudou significativamente
            } catch (error) {
                console.warn('Erro ao comparar detecções:', error);
                return true; // Em caso de erro, redesenhar
            }
        },

        // Limpar caixas de detecção com fade-out
        clearDetectionBoxes() {
            const container = document.getElementById('detection-container');
            if (container) {
                // Fade-out gradual em vez de remoção instantânea
                const boxes = container.querySelectorAll('.detection-box');
                boxes.forEach(box => {
                    box.style.opacity = '0';
                    setTimeout(() => {
                        if (box.parentNode) {
                            box.remove();
                        }
                    }, 300); // Aguardar transição de 300ms
                });
            }
        },

        // Limpar caixas antigas (sem detecção por muito tempo)
        clearOldDetections() {
            const container = document.getElementById('detection-container');
            if (container) {
                const boxes = container.querySelectorAll('.detection-box');
                const now = Date.now();
                boxes.forEach(box => {
                    const boxId = box.id;
                    const timestamp = parseInt(boxId.split('-').pop());
                    const age = now - timestamp;
                    
                    // Remover caixas com mais de 5 segundos
                    if (age > 5000) {
                        box.style.opacity = '0';
                        setTimeout(() => {
                            if (box.parentNode) {
                                box.remove();
                            }
                        }, 300);
                    }
                });
            }
        },

        // Carregar status do sistema
        async loadSystemStatus() {
            try {
                const response = await fetch(CONFIG.API.BASE_URL + CONFIG.API.ENDPOINTS.STATUS);
                if (response.ok) {
                    const data = await response.json();
                    this.systemStatus = { ...this.systemStatus, ...data };
                    this.connectionStatus = 'connected';
                }
    } catch (error) {
                console.error('❌ Erro ao carregar status:', error);
                this.connectionStatus = 'disconnected';
            }
        },
        
        // Carregar configurações
        async loadConfig() {
            try {
                const response = await fetch(CONFIG.API.BASE_URL + CONFIG.API.ENDPOINTS.CONFIG);
                if (response.ok) {
                    const data = await response.json();
                    this.config = { ...this.config, ...data };
                }
    } catch (error) {
                console.error('❌ Erro ao carregar configurações:', error);
            }
        },

        // Carregar configurações da câmera
        async loadCameraConfig() {
            try {
                const response = await fetch(CONFIG.API.BASE_URL + CONFIG.API.ENDPOINTS.CAMERA_CONFIG);
                if (response.ok) {
                    const data = await response.json();
                    this.cameraConfig = { ...this.cameraConfig, ...data };
                }
    } catch (error) {
                console.error('❌ Erro ao carregar configurações da câmera:', error);
            }
        },
        
        // Carregar histórico
        async loadHistory() {
            try {
                const response = await fetch(CONFIG.API.BASE_URL + CONFIG.API.ENDPOINTS.HISTORY + '?limit=50');
                if (response.ok) {
                    const data = await response.json();
                    this.history = Array.isArray(data) ? data : [];
                } else {
                    this.history = [];
                }
    } catch (error) {
                console.error('❌ Erro ao carregar histórico:', error);
                this.history = [];
            }
        },
        
        // Carregar informações das classes
        async loadClassesInfo() {
            try {
                console.log('🎯 Carregando informações das classes...');
                const response = await fetch(CONFIG.API.BASE_URL + CONFIG.API.ENDPOINTS.CLASSES);
                if (response.ok) {
                    const data = await response.json();
                    console.log('📋 Dados das classes recebidos:', data);
                    this.classesInfo.class_names = data.class_names || [];
                    this.classesInfo.total_classes = data.total_classes || this.classesInfo.class_names.length;
                    this.classesInfo.active_classes = data.enabled_classes || [...this.classesInfo.class_names];
                    console.log('✅ Classes carregadas:', this.classesInfo);
                } else {
                    console.error('❌ Erro na resposta do servidor:', response.status, response.statusText);
                }
            } catch (error) {
                console.error('❌ Erro ao carregar classes:', error);
            }
        },

        // Alternar classe habilitada/desabilitada
        toggleClass(className) {
            const idx = this.classesInfo.active_classes.indexOf(className);
            if (idx >= 0) {
                // não permitir desabilitar 'person'
                if (className === 'person') return;
                this.classesInfo.active_classes.splice(idx, 1);
            } else {
                this.classesInfo.active_classes.push(className);
            }
        },

        // Salvar classes habilitadas no backend
        async saveEnabledClasses() {
            try {
                const payload = { enabled_classes: this.classesInfo.active_classes };
                console.log('💾 Salvando classes:', payload);
                const resp = await fetch(CONFIG.API.BASE_URL + '/classes/enabled', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (resp.ok) {
                    const result = await resp.json();
                    this.classesInfo.active_classes = result.enabled_classes || this.classesInfo.active_classes;
                    console.log('✅ Classes salvas:', this.classesInfo.active_classes);
                    this.showToast('Classes atualizadas com sucesso!', 'success');
                } else {
                    console.error('❌ Falha ao salvar classes:', resp.status, resp.statusText);
                    this.showToast('Erro ao salvar classes', 'error');
                }
            } catch (e) {
                console.error('❌ Erro ao salvar classes:', e);
                this.showToast('Erro ao salvar classes', 'error');
            }
        },
        
        // Inicializar gráfico
        initChart() {
            const ctx = document.getElementById('reportChart');
            if (!ctx) return;
            
            // Destruir gráfico existente se houver
            if (this.reportChart) {
                try {
                    this.reportChart.destroy();
                } catch (e) {
                    console.warn('Erro ao destruir gráfico anterior:', e);
                }
                this.reportChart = null;
            }
            
            try {
                this.reportChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Taxa de Conformidade (%)',
                            data: [],
                            borderColor: '#1e40af',
                            backgroundColor: 'rgba(30, 64, 175, 0.1)',
                            tension: 0.4,
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                ticks: {
                                    callback: function(value) {
                                        return value + '%';
                                    }
                                }
                            },
                            x: {
                                display: true,
                                title: {
                                    display: true,
                                    text: 'Tempo'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: true,
                                position: 'top'
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `Conformidade: ${context.parsed.y}%`;
                                    }
                                }
                            }
                        },
                        animation: {
                            duration: 0
                        }
                    }
                });
                
                console.log('📊 Gráfico inicializado');
            } catch (error) {
                console.error('Erro ao inicializar gráfico:', error);
                this.reportChart = null;
            }
        },
        
        // Atualizar gráfico
        updateChart() {
            if (!this.reportChart || !this.reportChart.data || !this.reportChart.data.datasets) return;
            
            try {
                const now = new Date();
                const timeLabel = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
                
                // Verificar se os dados existem
                if (!this.reportChart.data.labels) this.reportChart.data.labels = [];
                if (!this.reportChart.data.datasets[0]) return;
                if (!this.reportChart.data.datasets[0].data) this.reportChart.data.datasets[0].data = [];
                
                // Adicionar novo ponto
                this.reportChart.data.labels.push(timeLabel);
                this.reportChart.data.datasets[0].data.push(this.stats.compliance_score || 0);

                // Manter apenas os últimos 20 pontos
                if (this.reportChart.data.labels.length > 20) {
                    this.reportChart.data.labels.shift();
                    this.reportChart.data.datasets[0].data.shift();
                }

                // Atualizar sem animação para evitar problemas
                this.reportChart.update('none');
            } catch (error) {
                console.error('Erro ao atualizar gráfico:', error);
                // Se houver erro, reinicializar o gráfico
                setTimeout(() => {
                    this.initChart();
                }, 1000);
            }
        },

        // Carregar dados do relatório
        loadReportData() {
            // Desabilitar gráfico temporariamente para evitar erros
            console.log('📊 Relatório carregado (gráfico desabilitado temporariamente)');
        },

        // Tirar snapshot
        takeSnapshot() {
            const img = document.getElementById('mjpeg');
            if (img) {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                canvas.width = img.naturalWidth;
                canvas.height = img.naturalHeight;
                
                ctx.drawImage(img, 0, 0);
                
                // Download da imagem
                const link = document.createElement('a');
                link.download = `athena-snapshot-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
                link.href = canvas.toDataURL();
                link.click();
                
                console.log('📸 Snapshot salvo');
            }
        },

        // Formatar data/hora
        formatDateTime(timestamp) {
            return new Date(timestamp).toLocaleString('pt-BR');
        },

        // Formatar duração
        formatDuration(seconds) {
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

        // Sistema de notificações toast
        showToast(message, type = 'info') {
            // Remover toast existente se houver
            const existingToast = document.querySelector('.toast-notification');
            if (existingToast) {
                existingToast.remove();
            }

            // Criar toast
            const toast = document.createElement('div');
            toast.className = 'toast-notification fixed top-4 right-4 z-50 max-w-sm w-full';
            
            const bgColor = type === 'success' ? 'bg-green-500' : 
                           type === 'error' ? 'bg-red-500' : 
                           type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500';
            
            toast.innerHTML = `
                <div class="${bgColor} text-white px-6 py-4 rounded-lg shadow-lg flex items-center space-x-3">
                    <div class="flex-shrink-0">
                        <i class="fas ${type === 'success' ? 'fa-check' : 
                                      type === 'error' ? 'fa-times' : 
                                      type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info'}"></i>
                    </div>
                    <div class="flex-1">
                        <p class="text-sm font-medium">${message}</p>
                    </div>
                    <button onclick="this.closest('.toast-notification').remove()" class="flex-shrink-0 text-white hover:text-gray-200">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `;
            
            document.body.appendChild(toast);
            
            // Auto-remover após 5 segundos
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.remove();
                }
            }, 5000);
        },
        
        // Getter para classes do modelo
        get modelClasses() {
            return this.classesInfo.class_names || [
                'person', 'ear', 'ear-mufs', 'face', 'face-guard', 'face-mask-medical', 
                'foot', 'tools', 'glasses', 'gloves', 'helmet', 'hands', 'head', 
                'medical-suit', 'shoes', 'safety-suit', 'safety-vest'
            ];
        },

        // ===== FUNÇÕES DE VÍDEO EM TEMPO REAL =====

        // Carregar vídeo para detecção em tempo real
        loadVideoForRealtimeDetection(event) {
            const file = event.target.files[0];
            if (!file) return;

            // Validar tamanho (500MB máximo)
            const maxSize = 500 * 1024 * 1024;
            if (file.size > maxSize) {
                this.showToast('Arquivo muito grande. Máximo 500MB.', 'error');
                return;
            }

            // Validar tipo
            const supportedTypes = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'];
            const fileExt = '.' + file.name.split('.').pop().toLowerCase();
            if (!supportedTypes.includes(fileExt)) {
                this.showToast('Formato não suportado. Use: MP4, AVI, MOV, MKV, WMV, FLV, WEBM', 'error');
                return;
            }

            // Criar URL do vídeo
            this.currentVideo = file;
            this.currentVideoUrl = URL.createObjectURL(file);
            
            // Resetar estatísticas
            this.realtimeStats = {
                total_pessoas: 0,
                com_capacete: 0,
                com_colete: 0,
                compliance_score: 0
            };
            
            this.currentDetections = [];
            this.detectionActive = false;
            
            this.showToast('Vídeo carregado com sucesso!', 'success');
        },

        // Quando o vídeo é carregado
        onVideoLoaded() {
            this.videoPlayer = document.getElementById('videoPlayer');
            this.videoOverlay = document.getElementById('videoOverlay');
            
            if (this.videoPlayer) {
                this.videoDimensions.width = this.videoPlayer.videoWidth;
                this.videoDimensions.height = this.videoPlayer.videoHeight;
                
                // Ajustar canvas overlay ao tamanho visível do vídeo (evita deslocamento)
                this.resizeOverlayToVideo();
            }
        },

        // Ajusta o canvas overlay ao tamanho atual de exibição do vídeo
        resizeOverlayToVideo() {
            if (!this.videoPlayer || !this.videoOverlay) return;
            const w = this.videoPlayer.clientWidth;
            const h = this.videoPlayer.clientHeight;
            // atributos width/height definem o sistema de coordenadas do canvas
            this.videoOverlay.width = w;
            this.videoOverlay.height = h;
        },

        // Quando o tempo do vídeo atualiza
        onVideoTimeUpdate() {
            // mantemos como fallback; preferir requestVideoFrameCallback se disponível
            if (typeof this.videoPlayer.requestVideoFrameCallback === 'function') return;
            if (this.detectionActive && this.videoPlayer) {
                const now = performance.now();
                const minIntervalMs = 1000 / (this.dynamicTargetFps || CONFIG.DETECTION.TARGET_FPS);
                if (now - this.lastDetectionTime >= minIntervalMs && !this.inFlightDetection) {
                    this.detectInCurrentFrame();
                    this.lastDetectionTime = now;
                }
            }
        },

        // Alternar detecção
        toggleDetection() {
            if (!this.currentVideoUrl) {
                this.showToast('Selecione um vídeo antes de iniciar a detecção', 'warning');
                return;
            }
            this.detectionActive = !this.detectionActive;
            
            if (this.detectionActive) {
                this.showToast('Detecção iniciada', 'success');
                this.lastDetectionTime = performance.now();
                this.openVideoWS();
                // usar requestVideoFrameCallback para melhor cadência quando disponível
                if (typeof this.videoPlayer.requestVideoFrameCallback === 'function') {
                    const loop = () => {
                        if (!this.detectionActive) return;
                        const now = performance.now();
                        const minIntervalMs = 1000 / (this.dynamicTargetFps || CONFIG.DETECTION.TARGET_FPS);
                        if (now - this.lastDetectionTime >= minIntervalMs && !this.inFlightDetection) {
                            this.detectInCurrentFrame();
                            this.lastDetectionTime = now;
                        }
                        this.videoPlayer.requestVideoFrameCallback(loop);
                    };
                    this.videoPlayer.requestVideoFrameCallback(loop);
                }
            } else {
                this.showToast('Detecção pausada', 'info');
                this.clearDetections();
                this.closeVideoWS();
            }
        },

        // Abrir WebSocket para detecção
        openVideoWS() {
            try {
                if (this.videoWS && this.wsConnected) return;
                const base = (CONFIG.API && CONFIG.API.BASE_URL) ? CONFIG.API.BASE_URL : window.location.origin;
                const path = (CONFIG.API && CONFIG.API.ENDPOINTS && CONFIG.API.ENDPOINTS.DETECT_WS) ? CONFIG.API.ENDPOINTS.DETECT_WS : '/ws/detect-video';
                const wsUrl = base.replace('https://', 'wss://').replace('http://', 'ws://') + path;
                this.videoWS = new WebSocket(wsUrl);
                this.videoWS.binaryType = 'arraybuffer';

                this.videoWS.onopen = () => { this.wsConnected = true; console.log('🎥 WS conectado:', wsUrl); this.showToast('Canal de detecção conectado', 'success'); };
                this.videoWS.onclose = () => { this.wsConnected = false; console.log('🎥 WS fechado'); this.showToast('Canal de detecção fechado', 'warning'); };
                this.videoWS.onerror = (e) => { this.wsConnected = false; console.error('🎥 WS erro:', e); this.showToast('Erro no canal de detecção', 'error'); };
                this.videoWS.onmessage = (evt) => {
                    try {
                        const text = typeof evt.data === 'string' ? evt.data : new TextDecoder().decode(evt.data);
                        const data = JSON.parse(text);
                        if (data.type === 'ready') {
                            console.log('🎥 WS pronto para receber frames');
                        } else if (data.type === 'detections') {
                            if (data.frame_width && data.frame_height) {
                                this.lastSentFrameSize = { width: data.frame_width, height: data.frame_height };
                            }
                            this.currentDetections = data.detections || [];
                            this.updateRealtimeStats();
                            this.drawDetections();
                            if (this.wsLastSendAt) { this.wsRTTms = Math.round(performance.now() - this.wsLastSendAt); }
                            this.adaptNetwork();
                            const s = Math.floor(performance.now() / 1000);
                            if (s !== this.lastFpsTick) { this.detectionFPS = this.framesThisSecond; this.framesThisSecond = 0; this.lastFpsTick = s; } else { this.framesThisSecond += 1; }
                            this.inFlightDetection = false;
                        } else if (data.type === 'error') {
                            console.error('🎥 WS erro payload:', data.message);
                            this.showToast(`Erro na detecção: ${data.message}`, 'error');
                            this.inFlightDetection = false;
                        }
                    } catch (_) {}
                };
            } catch (e) { console.error('WS error:', e); }
        },

        // Fechar WebSocket
        closeVideoWS() {
            try { if (this.videoWS) this.videoWS.close(); } catch (_) {}
            this.videoWS = null;
            this.wsConnected = false;
            this.inFlightDetection = false;
        },

        // Detectar no frame atual
        async detectInCurrentFrame() {
            if (!this.videoPlayer || !this.videoOverlay) return;
            if (!this.wsConnected || !this.videoWS) return;

            try {
                if (this.inFlightDetection) return;
                this.inFlightDetection = true;
                // Capturar frame atual do vídeo
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                // Reduzir resolução para acelerar upload/inferência (máx 640px no maior lado)
                const maxSide = CONFIG.DETECTION.MAX_FRAME_WIDTH || 640;
                const vw = this.videoDimensions.width;
                const vh = this.videoDimensions.height;
                const scale = Math.min(1, maxSide / Math.max(vw, vh));
                canvas.width = Math.round(vw * scale);
                canvas.height = Math.round(vh * scale);
                // guardar o tamanho do frame enviado para escalar boxes depois
                this.lastSentFrameSize = { width: canvas.width, height: canvas.height };
                
                ctx.drawImage(this.videoPlayer, 0, 0, canvas.width, canvas.height);
                
                // Converter para blob e enviar via WS
                canvas.toBlob(async (blob) => {
                    try {
                        const buffer = await blob.arrayBuffer();
                        this.wsLastSendAt = performance.now();
                        this.videoWS.send(buffer);
                    } catch (_) {
                        this.inFlightDetection = false;
                    }
                }, 'image/jpeg', (this.dynamicJpegQuality || CONFIG.DETECTION.JPEG_QUALITY || 0.6));
                
            } catch (error) {
                console.error('Erro na detecção:', error);
                this.inFlightDetection = false;
            }
        },

        // Ajuste adaptativo simples
        adaptNetwork() {
            const rtt = this.wsRTTms || 0;
            if (rtt > 500) {
                this.dynamicTargetFps = Math.max(1, (this.dynamicTargetFps || 2) - 1);
                this.dynamicJpegQuality = Math.max(0.4, (this.dynamicJpegQuality || 0.5) - 0.05);
            } else if (rtt < 180) {
                this.dynamicTargetFps = Math.min(6, (this.dynamicTargetFps || 2) + 1);
                this.dynamicJpegQuality = Math.min(0.75, (this.dynamicJpegQuality || 0.5) + 0.05);
            }
        },

        async toggleFullscreen() {
            const container = document.getElementById('videoContainer');
            if (!container) return;
            if (document.fullscreenElement) {
                await document.exitFullscreen();
            } else {
                await container.requestFullscreen();
            }
            this.resizeOverlayToVideo();
        },

        // Atualizar estatísticas em tempo real
        updateRealtimeStats() {
            const stats = {
                total_pessoas: 0,
                com_capacete: 0,
                com_colete: 0,
                compliance_score: 0
            };
            
            this.currentDetections.forEach(detection => {
                if (detection.class_name === 'person') {
                    stats.total_pessoas++;
                } else if (['helmet', 'safety_helmet', 'ear', 'ear-mufs'].includes(detection.class_name)) {
                    stats.com_capacete++;
                } else if (['vest', 'safety_vest', 'safety-suit', 'medical-suit'].includes(detection.class_name)) {
                    stats.com_colete++;
                }
            });
            
            // Calcular compliance score
            if (stats.total_pessoas > 0) {
                const totalEPIs = stats.com_capacete + stats.com_colete;
                const maxPossibleEPIs = stats.total_pessoas * 2; // Capacete + Colete
                stats.compliance_score = Math.round((totalEPIs / maxPossibleEPIs) * 100);
            }
            
            this.realtimeStats = stats;
        },

        // Desenhar detecções no overlay
        drawDetections() {
            if (!this.videoOverlay) return;
            
            // garantir que o canvas está alinhado ao tamanho visível do vídeo
            this.resizeOverlayToVideo();
            const ctx = this.videoOverlay.getContext('2d');
            ctx.clearRect(0, 0, this.videoOverlay.width, this.videoOverlay.height);
            
            // Cores para diferentes classes
            const colors = {
                'person': '#00ff00',
                'helmet': '#0000ff',
                'vest': '#ff0000',
                'safety_helmet': '#0000ff',
                'safety_vest': '#ff0000',
                'ear': '#ffff00',
                'ear-mufs': '#ffa500',
                'face': '#800080',
                'face-guard': '#00ffff',
                'face-mask-medical': '#ffc0cb',
                'foot': '#a52a2a',
                'tools': '#808080',
                'glasses': '#008000',
                'gloves': '#ff1493',
                'hands': '#ff4500',
                'head': '#4b0082',
                'medical-suit': '#006400',
                'shoes': '#8b4513',
                'safety-suit': '#00008b'
            };
            
            // fator de escala considerando letterbox/pillarbox (object-fit)
            const canvasW = this.videoOverlay.width;
            const canvasH = this.videoOverlay.height;
            const vidW = this.videoPlayer ? this.videoPlayer.videoWidth : canvasW;
            const vidH = this.videoPlayer ? this.videoPlayer.videoHeight : canvasH;
            const sentW = this.lastSentFrameSize.width || vidW;
            const sentH = this.lastSentFrameSize.height || vidH;
            const scaleToCanvas = Math.min(canvasW / vidW, canvasH / vidH);
            const displayW = vidW * scaleToCanvas;
            const displayH = vidH * scaleToCanvas;
            const offsetX = (canvasW - displayW) / 2;
            const offsetY = (canvasH - displayH) / 2;
            const scaleX = displayW / sentW;
            const scaleY = displayH / sentH;

            this.currentDetections.forEach(detection => {
                const [bx1, by1, bx2, by2] = detection.bbox;
                // escalar coords do frame processado para o retângulo visível do vídeo
                const x1 = offsetX + bx1 * scaleX;
                const y1 = offsetY + by1 * scaleY;
                const x2 = offsetX + bx2 * scaleX;
                const y2 = offsetY + by2 * scaleY;
                const class_name = detection.class_name;
                const confidence = detection.confidence;
                const color = colors[class_name] || '#ffffff';
                
                // Desenhar bounding box
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                
                // Desenhar label
                const label = `${class_name}: ${(confidence * 100).toFixed(1)}%`;
                ctx.fillStyle = color;
                ctx.font = '14px Arial';
                ctx.fillText(label, x1, y1 - 5);
            });
        },

        // Limpar detecções
        clearDetections() {
            this.currentDetections = [];
            if (this.videoOverlay) {
                const ctx = this.videoOverlay.getContext('2d');
                ctx.clearRect(0, 0, this.videoOverlay.width, this.videoOverlay.height);
            }
        },



    }
}

// Tornar disponível globalmente
window.athenaApp = athenaApp;