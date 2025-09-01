"""
Módulo de Histórico
Sistema de gerenciamento de histórico de detecções
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque

from .config import CONFIG
from .utils import format_timestamp, safe_json_dumps

logger = logging.getLogger(__name__)

class DetectionRecord:
    """Registro de uma detecção"""
    
    def __init__(self, frame_id: int, detections: List[Dict], epi_summary: Dict[str, int], timestamp: float = None):
        self.frame_id = frame_id
        self.detections = detections
        self.epi_summary = epi_summary
        self.timestamp = timestamp or time.time()
        self.compliance_status = "unknown"  # Simplificado
    
    def _has_violations(self) -> bool:
        """Verifica se há violações de EPI"""
        try:
            people_count = sum(1 for d in self.detections if d["label"].lower() == "person")
            
            if people_count == 0:
                return False
            
            # Verificar se há pessoas sem EPIs
            without_helmet = self.epi_summary.get("sem_capacete", 0)
            without_vest = self.epi_summary.get("sem_colete", 0)
            
            return without_helmet > 0 or without_vest > 0
                
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return {
            "id": self.frame_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "timestamp_formatted": format_timestamp(self.timestamp),
            "total_pessoas": sum(1 for d in self.detections if d["label"].lower() == "person"),
            "sem_capacete": self.epi_summary.get("sem_capacete", 0),
            "sem_colete": self.epi_summary.get("sem_colete", 0),
            "violations": self._get_violations(),
            "frame_id": self.frame_id
        }
    
    def to_json(self) -> str:
        """Converte para JSON"""
        return safe_json_dumps(self.to_dict())

class HistoryManager:
    """Gerenciador de histórico de detecções"""
    
    def __init__(self):
        self.max_entries = CONFIG.HISTORY_MAX_ENTRIES
        self.cleanup_interval = CONFIG.HISTORY_CLEANUP_INTERVAL
        
        # Histórico em memória (deque para performance)
        self.history = deque(maxlen=self.max_entries)
        
        # Estatísticas
        self.stats = {
            "total_records": 0,
            "compliant_records": 0,
            "partial_records": 0,
            "violation_records": 0,
            "last_update": time.time()
        }
        
        # Última limpeza
        self.last_cleanup = time.time()
        
        logger.info("History Manager inicializado")
    
    def add_detection(self, frame_id: int, detections: List[Dict], epi_summary: Dict[str, int]) -> bool:
        """Adiciona nova detecção ao histórico"""
        try:
            # Criar registro
            record = DetectionRecord(frame_id, detections, epi_summary)
            
            # Adicionar ao histórico
            self.history.append(record)
            
            # Atualizar estatísticas
            self._update_stats(record)
            
            # Verificar se precisa fazer limpeza
            self._check_cleanup()
            
            logger.debug(f"Detecção adicionada ao histórico: frame {frame_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar detecção ao histórico: {e}")
            return False
    
    def _update_stats(self, record: DetectionRecord):
        """Atualiza estatísticas do histórico"""
        self.stats["total_records"] += 1
        
        if record._has_violations():
            self.stats["violation_records"] += 1
        
        self.stats["last_update"] = time.time()
    
    def _check_cleanup(self):
        """Verifica se precisa fazer limpeza"""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.cleanup_interval:
            self.cleanup_old_records()
            self.last_cleanup = current_time
    
    def cleanup_old_records(self, max_age_hours: int = 24) -> int:
        """Remove registros antigos"""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            removed_count = 0
            
            # Filtrar registros antigos
            old_records = [r for r in self.history if r.timestamp < cutoff_time]
            
            for record in old_records:
                if record in self.history:
                    self.history.remove(record)
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Removidos {removed_count} registros antigos do histórico")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Erro ao limpar registros antigos: {e}")
            return 0
    
    def get_history(self, limit: int = 50, offset: int = 0, compliance_filter: str = None) -> List[Dict[str, Any]]:
        """Retorna histórico com filtros"""
        try:
            # Filtrar por compliance se especificado
            if compliance_filter:
                filtered_records = [r for r in self.history if r.compliance_status == compliance_filter]
            else:
                filtered_records = list(self.history)
            
            # Aplicar paginação
            end_index = offset + limit
            paginated_records = filtered_records[offset:end_index]
            
            # Converter para dicionários
            return [record.to_dict() for record in paginated_records]
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico: {e}")
            return []
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do histórico"""
        try:
            total_records = len(self.history)
            
            if total_records == 0:
                            return {
                "total_records": 0,
                "violation_records": 0,
                "violation_percentage": 0,
                "last_update": None
            }
            
            # Calcular porcentagem de violações
            violation_pct = (self.stats["violation_records"] / total_records) * 100
            
            return {
                "total_records": total_records,
                "violation_records": self.stats["violation_records"],
                "violation_percentage": round(violation_pct, 2),
                "last_update": format_timestamp(self.stats["last_update"])
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas do histórico: {e}")
            return {"error": str(e)}
    
    def get_compliance_trend(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retorna tendência de compliance ao longo do tempo"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            # Filtrar registros do período
            recent_records = [r for r in self.history if r.timestamp >= cutoff_time]
            
            if not recent_records:
                return []
            
            # Agrupar por hora
            hourly_stats = {}
            
            for record in recent_records:
                hour_key = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:00")
                
                if hour_key not in hourly_stats:
                    hourly_stats[hour_key] = {
                        "hour": hour_key,
                        "total": 0,
                        "compliant": 0,
                        "partial": 0,
                        "violation": 0
                    }
                
                hourly_stats[hour_key]["total"] += 1
                
                if record.compliance_status == "compliant":
                    hourly_stats[hour_key]["compliant"] += 1
                elif record.compliance_status == "partial":
                    hourly_stats[hour_key]["partial"] += 1
                elif record.compliance_status == "violation":
                    hourly_stats[hour_key]["violation"] += 1
            
            # Converter para lista e ordenar
            trend_data = list(hourly_stats.values())
            trend_data.sort(key=lambda x: x["hour"])
            
            return trend_data
            
        except Exception as e:
            logger.error(f"Erro ao calcular tendência de compliance: {e}")
            return []
    
    def search_history(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Busca no histórico por texto"""
        try:
            query_lower = query.lower()
            results = []
            
            for record in self.history:
                # Buscar por compliance status
                if query_lower in record.compliance_status.lower():
                    results.append(record.to_dict())
                    continue
                
                # Buscar por labels nas detecções
                for detection in record.detections:
                    if query_lower in detection["label"].lower():
                        results.append(record.to_dict())
                        break
                
                # Limitar resultados
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na busca do histórico: {e}")
            return []
    
    def export_history(self, format: str = "json") -> str:
        """Exporta histórico completo"""
        try:
            if format.lower() == "json":
                export_data = {
                    "export_time": datetime.now().isoformat(),
                    "total_records": len(self.history),
                    "records": [record.to_dict() for record in self.history]
                }
                return safe_json_dumps(export_data, indent=2)
            
            else:
                raise ValueError(f"Formato de exportação não suportado: {format}")
                
        except Exception as e:
            logger.error(f"Erro ao exportar histórico: {e}")
            return ""
    
    def clear_history(self) -> bool:
        """Limpa todo o histórico"""
        try:
            self.history.clear()
            self.stats = {
                "total_records": 0,
                "violation_records": 0,
                "last_update": time.time()
            }
            
            logger.info("Histórico limpo com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao limpar histórico: {e}")
            return False

class EPIHistorySystem:
    """Sistema de histórico para EPIs"""
    
    def __init__(self):
        self.history_manager = HistoryManager()
        logger.info("Sistema de histórico EPI inicializado")
    
    def add_detection(self, frame_id: int, detections: List[Dict], epi_summary: Dict[str, int]) -> bool:
        """Adiciona detecção ao histórico"""
        return self.history_manager.add_detection(frame_id, detections, epi_summary)
    
    def get_history(self, limit: int = 50, offset: int = 0, compliance_filter: str = None) -> List[Dict[str, Any]]:
        """Retorna histórico"""
        return self.history_manager.get_history(limit, offset, compliance_filter)
    
    def get_history_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do histórico"""
        return self.history_manager.get_history_stats()
    
    def get_compliance_trend(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retorna tendência de compliance"""
        return self.history_manager.get_compliance_trend(hours)
    
    def get_violations_data(self, period: str = "24h", start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Retorna dados de violações de EPI para relatórios"""
        try:
            if period == "custom" and start_date and end_date:
                # Converter strings de data para timestamps
                start_ts = datetime.fromisoformat(start_date.replace('Z', '+00:00')).timestamp()
                end_ts = datetime.fromisoformat(end_date.replace('Z', '+00:00')).timestamp()
                
                # Filtrar registros no período
                filtered_records = [
                    r for r in self.history_manager.history 
                    if start_ts <= r.timestamp <= end_ts and r._has_violations()
                ]
            else:
                # Períodos predefinidos
                hours_map = {"24h": 24, "7d": 168, "30d": 720}
                hours = hours_map.get(period, 24)
                cutoff_time = time.time() - (hours * 3600)
                
                filtered_records = [
                    r for r in self.history_manager.history 
                    if r.timestamp >= cutoff_time and r._has_violations()
                ]
            
            # Converter para formato de relatório
            report_data = []
            for record in filtered_records:
                report_item = {
                    "id": record.frame_id,
                    "timestamp": record.timestamp,
                    "total_pessoas": sum(1 for d in record.detections if d["label"].lower() == "person"),
                    "sem_capacete": record.epi_summary.get("sem_capacete", 0),
                    "sem_colete": record.epi_summary.get("sem_colete", 0),
                    "violations": self._get_violations(record),
                    "frame_id": record.frame_id
                }
                report_data.append(report_item)
            
            return report_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de violações: {e}")
            return []
    
    def get_violations_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Retorna dados de tendências de violações para gráficos"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            # Agrupar por intervalos de hora
            intervals = {}
            
            for record in self.history_manager.history:
                if record.timestamp >= cutoff_time and record._has_violations():
                    # Calcular intervalo
                    hour = int((record.timestamp - cutoff_time) / 3600)
                    interval_key = f"h{hour}"
                    
                    if interval_key not in intervals:
                        intervals[interval_key] = {
                            "violations": 0,
                            "people_without_helmet": 0,
                            "people_without_vest": 0,
                            "total_people": 0
                        }
                    
                    intervals[interval_key]["violations"] += 1
                    intervals[interval_key]["people_without_helmet"] += record.epi_summary.get("sem_capacete", 0)
                    intervals[interval_key]["people_without_vest"] += record.epi_summary.get("sem_colete", 0)
                    intervals[interval_key]["total_people"] += sum(1 for d in record.detections if d["label"].lower() == "person")
            
            return intervals
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de tendências de violações: {e}")
            return {}
    
    def _calculate_compliance_rate(self, record: DetectionRecord) -> float:
        """Calcula taxa de compliance para um registro"""
        try:
            total_people = record.epi_summary.get("com_capacete", 0) + record.epi_summary.get("sem_capacete", 0)
            if total_people == 0:
                return 0.0
            
            compliant_people = record.epi_summary.get("com_capacete", 0) + record.epi_summary.get("com_colete", 0)
            return round(compliant_people / (total_people * 2), 2)
        except:
            return 0.0
    
    def _get_violations(self, record: DetectionRecord) -> List[str]:
        """Retorna lista de violações para um registro"""
        violations = []
        
        if record.compliance_status == "violation":
            violations.append("Sem capacete e sem colete")
        elif record.compliance_status == "partial":
            if record.epi_summary.get("sem_capacete", 0) > 0:
                violations.append("Sem capacete")
            if record.epi_summary.get("sem_colete", 0) > 0:
                violations.append("Sem colete")
        
        return violations
    
    def search_history(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Busca no histórico"""
        return self.history_manager.search_history(query, limit)
    
    def export_history(self, format: str = "json") -> str:
        """Exporta histórico"""
        return self.history_manager.export_history(format)
    
    def clear_history(self) -> bool:
        """Limpa histórico"""
        return self.history_manager.clear_history()
    
    def cleanup(self):
        """Cleanup do sistema"""
        try:
            self.history_manager.cleanup_old_records()
            logger.info("Sistema de histórico finalizado")
        except Exception as e:
            logger.error(f"Erro no cleanup do histórico: {e}")
