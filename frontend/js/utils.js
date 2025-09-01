/**
 * Utilitários da aplicação Athena
 */

const Utils = {
    /**
     * Log com timestamp
     */
    log(level, message, data = null) {
        if (!CONFIG || !CONFIG.APP || !CONFIG.APP.DEBUG) return;
        
        const timestamp = new Date().toISOString();
        const prefix = `[${timestamp}] [${level.toUpperCase()}]`;
        
        switch (level.toLowerCase()) {
            case 'error':
                console.error(prefix, message, data || '');
                break;
            case 'warn':
                console.warn(prefix, message, data || '');
                break;
            case 'info':
                console.info(prefix, message, data || '');
                break;
            case 'debug':
                console.debug(prefix, message, data || '');
                break;
            default:
                console.log(prefix, message, data || '');
        }
    },
    
    /**
     * Formatar número com separadores
     */
    formatNumber(num) {
        if (num === null || num === undefined) return '0';
        return num.toLocaleString('pt-BR');
    },
    
    /**
     * Formatar porcentagem
     */
    formatPercentage(value, decimals = 1) {
        if (value === null || value === undefined) return '0%';
        return `${(value * 100).toFixed(decimals)}%`;
    },
    
    /**
     * Formatar data/hora
     */
    formatDateTime(date) {
        if (!date) return 'N/A';
        
        if (typeof date === 'string') {
            date = new Date(date);
        }
        
        if (date instanceof Date && !isNaN(date)) {
            return date.toLocaleString('pt-BR', {
                year: 'numeric',
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        }
        
        return 'Data inválida';
    },
    
    /**
     * Formatar duração em segundos
     */
    formatDuration(seconds) {
        if (!seconds || seconds < 0) return '0s';
        
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
    
    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    /**
     * Throttle function
     */
    throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },
    
    /**
     * Validar objeto
     */
    validateObject(obj, requiredKeys = []) {
        if (!obj || typeof obj !== 'object') return false;
        
        for (const key of requiredKeys) {
            if (!(key in obj)) return false;
        }
        
        return true;
    },
    
    /**
     * Validar array
     */
    validateArray(arr, minLength = 0) {
        return Array.isArray(arr) && arr.length >= minLength;
    },
    
    /**
     * Clonar objeto (shallow)
     */
    clone(obj) {
        if (obj === null || typeof obj !== 'object') return obj;
        if (obj instanceof Date) return new Date(obj.getTime());
        if (obj instanceof Array) return obj.slice();
        if (typeof obj === 'object') return { ...obj };
    },
    
    /**
     * Mesclar objetos
     */
    merge(target, ...sources) {
        if (!target) target = {};
        
        for (const source of sources) {
            if (!source) continue;
            
            for (const key in source) {
                if (source.hasOwnProperty(key)) {
                    if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
                        target[key] = this.merge(target[key], source[key]);
                    } else {
                        target[key] = source[key];
                    }
                }
            }
        }
        
        return target;
    },
    
    /**
     * Gerar ID único
     */
    generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    },
    
    /**
     * Verificar se elemento está visível
     */
    isElementVisible(element) {
        if (!element) return false;
        
        const rect = element.getBoundingClientRect();
        const style = window.getComputedStyle(element);
        
        return rect.width > 0 && 
               rect.height > 0 && 
               style.visibility !== 'hidden' && 
               style.display !== 'none';
    },
    
    /**
     * Scroll suave para elemento
     */
    scrollToElement(element, offset = 0) {
        if (!element) return;
        
        const elementPosition = element.offsetTop - offset;
        window.scrollTo({
            top: elementPosition,
            behavior: 'smooth'
        });
    },
    
    /**
     * Copiar texto para clipboard
     */
    async copyToClipboard(text) {
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
                return true;
            } else {
                // Fallback para navegadores mais antigos
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                const result = document.execCommand('copy');
                textArea.remove();
                return result;
            }
        } catch (error) {
            console.error('Erro ao copiar para clipboard:', error);
            return false;
        }
    },
    
    /**
     * Download de arquivo
     */
    downloadFile(content, filename, contentType = 'text/plain') {
        const blob = new Blob([content], { type: contentType });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
    },
    
    /**
     * Verificar se é mobile
     */
    isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    },
    
    /**
     * Verificar se é desktop
     */
    isDesktop() {
        return !this.isMobile();
    },
    
    /**
     * Verificar se está em modo escuro
     */
    isDarkMode() {
        return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
};

// Exportar para uso global
if (typeof window !== 'undefined') {
    window.Utils = Utils;
    
    // Expor funções específicas globalmente para Alpine.js
    window.formatDuration = Utils.formatDuration.bind(Utils);
    window.formatDateTime = Utils.formatDateTime.bind(Utils);
}
