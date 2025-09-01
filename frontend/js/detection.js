/**
 * Sistema de Detecção de EPIs - Compatível com Dashboard Athena
 * Foco: Funções utilitárias para detecção e desenho de boxes
 */

const DetectionSystem = {
    // Configurações de cores para boxes
        boxColors: {
            helmet: "#10b981",      // Verde para capacete
            vest: "#10b981",        // Verde para colete
        person: "#3b82f6",      // Azul para pessoa
        warning: "#d5b481",     // Amarelo-bege para aviso
        danger: "#ef4444"       // Vermelho para perigo
        },
    
    // Configurações de desenho
    config: {
        lineWidth: 2,
        fontSize: 12,
        fontFamily: "Arial",
        labelHeight: 20,
        labelPadding: 5
    },
    
    // Determinar cor do box baseado nos EPIs
    getBoxColor(personBox, helmetBox, vestBox) {
        if (helmetBox && vestBox) {
            return this.boxColors.helmet; // Verde - com ambos
        } else if (helmetBox || vestBox) {
            return this.boxColors.warning; // Amarelo - com um
        } else {
            return this.boxColors.danger; // Vermelho - sem nenhum
        }
    },
    
    // Desenhar box com rótulo
    drawBox(ctx, box, color, label, confidence) {
        const x = box.x;
        const y = box.y;
        const w = box.w;
        const h = box.h;
        
        // Desenhar retângulo
        ctx.strokeStyle = color;
        ctx.lineWidth = this.config.lineWidth;
        ctx.strokeRect(x, y, w, h);
        
        // Calcular posição do rótulo
        const labelWidth = 120;
        const labelX = Math.max(0, x);
        const labelY = Math.max(this.config.labelHeight, y);
        
        // Desenhar fundo do rótulo
        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY - this.config.labelHeight, labelWidth, this.config.labelHeight);
        
        // Texto do rótulo
        ctx.fillStyle = '#ffffff';
        ctx.font = `${this.config.fontSize}px ${this.config.fontFamily}`;
        ctx.fillText(
            `${label}:${Math.round(confidence * 100)}%`, 
            labelX + this.config.labelPadding, 
            labelY - this.config.labelPadding
        );
    },
    
    // Desenhar todas as detecções de um frame
    drawDetections(ctx, boxes, canvasWidth, canvasHeight) {
        if (!ctx || !boxes || !Array.isArray(boxes)) {
            return;
        }
        
        // Limpar canvas
        ctx.clearRect(0, 0, canvasWidth, canvasHeight);
        
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
                const color = this.getBoxColor(personBox, helmetBox, vestBox);
                
                // Desenhar box da pessoa
                this.drawBox(ctx, personBox, color, 'person', personBox.conf);
                
                // Desenhar boxes dos EPIs
                if (helmetBox) {
                    this.drawBox(ctx, helmetBox, this.boxColors.helmet, 'helmet', helmetBox.conf);
                }
                if (vestBox) {
                    this.drawBox(ctx, vestBox, this.boxColors.vest, 'vest', vestBox.conf);
                }
            }
        });
    },
    
    // Validar dados de detecção
    validateDetectionData(data) {
        if (!data || typeof data !== 'object') {
            return false;
        }
        
        if (!data.frame_id || typeof data.frame_id !== 'number') {
            return false;
        }
        
        if (!Array.isArray(data.boxes)) {
            return false;
        }
        
        // Validar cada box
        for (const box of data.boxes) {
            if (!box.x || !box.y || !box.w || !box.h || !box.label || !box.conf) {
                return false;
            }
        }
        
        return true;
    },
    
    // Processar dados de detecção e retornar estatísticas
    processDetectionStats(boxes) {
        if (!Array.isArray(boxes)) {
            return {
                total_pessoas: 0,
                com_capacete: 0,
                sem_capacete: 0,
                com_colete: 0,
                sem_colete: 0
            };
        }
        
        const stats = {
            total_pessoas: 0,
            com_capacete: 0,
            sem_capacete: 0,
            com_colete: 0,
            sem_colete: 0
        };
        
        // Agrupar por track_id
        const trackGroups = {};
        boxes.forEach(box => {
            if (!trackGroups[box.track_id]) {
                trackGroups[box.track_id] = [];
            }
            trackGroups[box.track_id].push(box);
        });
        
        // Analisar cada track
        Object.values(trackGroups).forEach(trackBoxes => {
            const hasPerson = trackBoxes.some(box => box.label === 'person');
            const hasHelmet = trackBoxes.some(box => box.label === 'helmet');
            const hasVest = trackBoxes.some(box => box.label === 'vest');
            
            if (hasPerson) {
                stats.total_pessoas++;
                
                if (hasHelmet) {
                    stats.com_capacete++;
                } else {
                    stats.sem_capacete++;
                }
                
                if (hasVest) {
                    stats.com_colete++;
                } else {
                    stats.sem_colete++;
                }
            }
        });
        
        return stats;
    },
    
    // Calcular métricas de performance
    calculatePerformanceMetrics(frames, timeWindow) {
        const now = Date.now();
        const recentFrames = frames.filter(frame => 
            (now - frame.timestamp) < timeWindow
        );
        
        if (recentFrames.length === 0) {
            return {
                fps: 0,
                avg_confidence: 0,
                detection_rate: 0
            };
        }
        
        const fps = recentFrames.length / (timeWindow / 1000);
        const avgConfidence = recentFrames.reduce((sum, frame) => 
            sum + frame.avg_confidence, 0
        ) / recentFrames.length;
        
        const detectionRate = recentFrames.filter(frame => 
            frame.boxes && frame.boxes.length > 0
        ).length / recentFrames.length;
        
        return {
            fps: Math.round(fps * 100) / 100,
            avg_confidence: Math.round(avgConfidence * 100) / 100,
            detection_rate: Math.round(detectionRate * 100) / 100
        };
    },
    
    // Exportar dados de detecção
    exportDetectionData(data, format = 'json') {
        switch (format.toLowerCase()) {
            case 'json':
                return JSON.stringify(data, null, 2);
                
            case 'csv':
                return this.convertToCSV(data);
                
            case 'xml':
                return this.convertToXML(data);
                
            default:
                throw new Error(`Formato não suportado: ${format}`);
        }
    },
    
    // Converter para CSV
    convertToCSV(data) {
        if (!data.boxes || !Array.isArray(data.boxes)) {
            return '';
        }
        
        const headers = ['frame_id', 'timestamp', 'track_id', 'label', 'x', 'y', 'w', 'h', 'confidence'];
        const rows = data.boxes.map(box => [
            data.frame_id,
            new Date().toISOString(),
            box.track_id,
            box.label,
            box.x,
            box.y,
            box.w,
            box.h,
            box.conf
        ]);
        
        return [headers, ...rows]
            .map(row => row.join(','))
            .join('\n');
    },
    
    // Converter para XML
    convertToXML(data) {
        if (!data.boxes || !Array.isArray(data.boxes)) {
            return '';
        }
        
        let xml = '<?xml version="1.0" encoding="UTF-8"?>\n';
        xml += '<detections>\n';
        xml += `  <frame_id>${data.frame_id}</frame_id>\n`;
        xml += `  <timestamp>${new Date().toISOString()}</timestamp>\n`;
        xml += '  <boxes>\n';
        
        data.boxes.forEach(box => {
            xml += '    <box>\n';
            xml += `      <track_id>${box.track_id}</track_id>\n`;
            xml += `      <label>${box.label}</label>\n`;
            xml += `      <x>${box.x}</x>\n`;
            xml += `      <y>${box.y}</y>\n`;
            xml += `      <w>${box.w}</w>\n`;
            xml += `      <h>${box.h}</h>\n`;
            xml += `      <confidence>${box.conf}</confidence>\n`;
            xml += '    </box>\n';
        });
        
        xml += '  </boxes>\n';
        xml += '</detections>';
        
        return xml;
    }
};

// Exportar para uso global
if (typeof window !== 'undefined') {
window.DetectionSystem = DetectionSystem;
}
