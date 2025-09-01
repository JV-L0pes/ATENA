/**
 * Configurações da aplicação Athena
 */

const CONFIG = {
    // Configurações da aplicação
    APP: {
        NAME: 'Athena Dashboard',
        VERSION: '2.0.0',
        DEBUG: true
    },
    
    // Configurações de API
    API: {
        BASE_URL: 'http://localhost:8000',
        ENDPOINTS: {
            HEALTH: '/health',
            STATUS: '/status',
            DETECTIONS: '/events/detections',
            STREAM: '/stream.mjpg',
            CONFIG: '/config',
            STATS: '/api/stats',
            HISTORY: '/history',
            LOGS: '/logs'
        }
    },
    
    // Configurações de UI
    UI: {
        // Intervalos de atualização (em ms)
        STATS_UPDATE_INTERVAL: 2000,
        STATUS_UPDATE_INTERVAL: 5000,
        
        // Configurações de paginação
        HISTORY_PAGE_SIZE: 50,
        
        // Configurações de toast
        TOAST_DURATION: 5000,
        
        // Configurações de gráfico
        CHART_ANIMATION_DURATION: 1000,
        

    },
    
    // Configurações de detecção
    DETECTION: {
        // Limite de FPS para desenho
        MAX_DRAW_FPS: 30,
        
        // Configurações de boxes
        BOX_LINE_WIDTH: 2,
        BOX_FONT_SIZE: 12,
        
        // Cores dos boxes
        COLORS: {
            HELMET_OK: '#10b981',      // Verde para capacete
            HELMET_MISSING: '#ef4444', // Vermelho para sem capacete
            VEST_OK: '#10b981',        // Verde para colete
            VEST_MISSING: '#ef4444',   // Vermelho para sem colete
            PERSON: '#3b82f6',         // Azul para pessoa
            WARNING: '#d5b481'         // Amarelo-bege para aviso
        }
    },
    
    // Configurações de timeout e reconexão
    TIMEOUTS: {
        // Timeouts (em ms)
        REQUEST_TIMEOUT: 10000,
        SSE_TIMEOUT: 30000,
        WEBSOCKET_TIMEOUT: 30000,
        
        // Tentativas de reconexão
        MAX_RECONNECT_ATTEMPTS: 3,
        RECONNECT_INTERVAL: 2000
    },
    

};

// Exportar para uso global
if (typeof window !== 'undefined') {
    window.CONFIG = CONFIG;
}
