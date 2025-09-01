/**
 * API da aplicação Athena - Versão Simplificada
 * Gerencia comunicações com o backend
 */

const API = {
    // Configurações base
    baseURL: 'http://localhost:8000',
    
    // Endpoints
    endpoints: {
        stream: '/stream.mjpg',
        detections: '/events/detections',
        snapshot: '/snapshot',
        stats: '/stats',
        history: '/history',
        status: '/status',
        config: '/config'
    },
    
    // Headers padrão
    defaultHeaders: {
        'Content-Type': 'application/json'
    },
    
    /**
     * Fazer requisição HTTP genérica
     */
    async request(url, options = {}) {
        try {
            const fullUrl = url.startsWith('http') ? url : this.baseURL + url;
            
            const config = {
                headers: { ...this.defaultHeaders, ...options.headers },
                ...options
            };
            
            const response = await fetch(fullUrl, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            // Verificar se a resposta é JSON
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            }
            
            return await response.text();
            
        } catch (error) {
            console.error('Erro na requisição API:', error);
            throw error;
        }
    },
    
    /**
     * GET request
     */
    async get(url, options = {}) {
        return this.request(url, { method: 'GET', ...options });
    },
    
    /**
     * POST request
     */
    async post(url, data, options = {}) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data),
            ...options
        });
    },
    
    /**
     * PUT request
     */
    async put(url, data, options = {}) {
        return this.request(url, {
            method: 'PUT',
            body: JSON.stringify(data),
            ...options
        });
    },
    
    /**
     * DELETE request
     */
    async delete(url, options = {}) {
        return this.request(url, { method: 'DELETE', ...options });
    },
    
    /**
     * Obter estatísticas
     */
    async getStats() {
        return this.get(this.endpoints.stats);
    },
    
    /**
     * Obter histórico
     */
    async getHistory(offset = 0, limit = 50) {
        const params = new URLSearchParams({ offset, limit });
        return this.get(`${this.endpoints.history}?${params}`);
    },
    
    /**
     * Obter status do sistema
     */
    async getStatus() {
        return this.get(this.endpoints.status);
    },
    
    /**
     * Obter configurações
     */
    async getConfig() {
        return this.get(this.endpoints.config);
    },
    
    /**
     * Salvar configurações
     */
    async saveConfig(config) {
        return this.put(this.endpoints.config, config);
    },
    
    /**
     * Capturar snapshot
     */
    async takeSnapshot() {
        return this.post(this.endpoints.snapshot, {});
    },
    
    /**
     * Testar conectividade
     */
    async testConnection() {
        try {
            await this.get(this.endpoints.status);
            return true;
        } catch (error) {
            return false;
        }
    },
    
    /**
     * Obter URL do stream
     */
    getStreamURL() {
        return this.baseURL + this.endpoints.stream;
    },
    
    /**
     * Obter URL das detecções SSE
     */
    getDetectionsURL() {
        return this.baseURL + this.endpoints.detections;
    },
    
    /**
     * Obter URL do WebSocket
     */
    getWebSocketURL() {
        return this.baseURL.replace('http', 'ws') + '/ws/detections';
    }
};

// Exportar para uso global
if (typeof window !== 'undefined') {
    window.API = API;
}
