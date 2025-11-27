"""
Sistema de Relatório de Detecções - Athena Dashboard
Gerencia relatórios de detecções positivas e negativas com exportação CSV
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class VideoReportSystem:
    """Sistema de relatório para vídeos processados"""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.reports = {}  # video_id -> report_data
        
    def create_report(self, video_id: str, video_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cria relatório completo de detecções positivas e negativas
        
        Args:
            video_id: ID do vídeo
            video_results: Resultados do processamento do vídeo
            
        Returns:
            Dict com relatório completo
        """
        try:
            detections = video_results.get('detections', [])
            summary = video_results.get('summary', {})
            video_info = video_results.get('video_info', {})
            
            # Processar detecções para separar positivas e negativas
            positive_detections = []
            negative_detections = []
            frame_detections = defaultdict(list)
            
            for detection in detections:
                frame_num = detection.get('frame_number', 0)
                class_name = detection.get('class_name', '')
                timestamp = detection.get('timestamp', 0)
                
                # Detecções positivas (EPIs presentes)
                if class_name not in ['person'] and not class_name.startswith('missing-'):
                    positive_detections.append({
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'class_name': class_name,
                        'confidence': detection.get('confidence', 0.0),
                        'bbox': detection.get('bbox', []),
                        'type': 'positive'
                    })
                
                # Detecções negativas (EPIs ausentes - missing-*)
                elif class_name.startswith('missing-'):
                    epi_name = class_name.replace('missing-', '')
                    negative_detections.append({
                        'frame_number': frame_num,
                        'timestamp': timestamp,
                        'class_name': class_name,
                        'missing_epi': epi_name,
                        'confidence': detection.get('confidence', 0.0),
                        'bbox': detection.get('bbox', []),
                        'type': 'negative'
                    })
                
                # Agrupar por frame
                frame_detections[frame_num].append(detection)
            
            # Analisar pessoas para detectar EPIs faltando
            people_detections = [d for d in detections if d.get('class_name') == 'person']
            for person in people_detections:
                missing_epis = person.get('missing_epis', [])
                if missing_epis:
                    frame_num = person.get('frame_number', 0)
                    timestamp = person.get('timestamp', 0)
                    for missing_epi in missing_epis:
                        negative_detections.append({
                            'frame_number': frame_num,
                            'timestamp': timestamp,
                            'class_name': f'missing-{missing_epi}',
                            'missing_epi': missing_epi,
                            'confidence': 1.0,  # Confiança alta para detecções virtuais
                            'bbox': person.get('bbox', []),
                            'type': 'negative',
                            'person_bbox': person.get('bbox', [])
                        })
            
            # Calcular estatísticas dinâmicas baseadas em todas as classes
            positive_by_class = self._count_by_class(positive_detections)
            negative_by_class = self._count_by_class(negative_detections)
            
            # Estatísticas dinâmicas - todas as classes
            stats = {
                'total_frames': video_results.get('total_frames', 0),
                'processed_frames': video_results.get('processed_frames', 0),
                'total_positive_detections': len(positive_detections),
                'total_negative_detections': len(negative_detections),
                'positive_by_class': positive_by_class,
                'negative_by_class': negative_by_class,
                'all_classes_detected': sorted(set(list(positive_by_class.keys()) + list(negative_by_class.keys()))),
                'compliance_score': summary.get('compliance_score', 0.0),
                'total_pessoas': summary.get('total_pessoas', 0),
                # Manter campos antigos para compatibilidade, mas calcular dinamicamente
                'com_capacete': positive_by_class.get('helmet', 0),
                'sem_capacete': negative_by_class.get('missing-helmet', 0),
                'com_colete': positive_by_class.get('safety-vest', 0),
                'sem_colete': negative_by_class.get('missing-safety-vest', 0),
                # Estatísticas por tipo de EPI
                'epi_statistics': self._calculate_epi_statistics(positive_detections, negative_detections)
            }
            
            # Criar relatório completo
            report = {
                'video_id': video_id,
                'created_at': datetime.now().isoformat(),
                'video_info': video_info,
                'summary': summary,
                'statistics': stats,
                'positive_detections': positive_detections,
                'negative_detections': negative_detections,
                'frame_detections': dict(frame_detections),
                'total_detections': len(positive_detections) + len(negative_detections)
            }
            
            # Salvar relatório
            self.reports[video_id] = report
            self._save_report_json(video_id, report)
            
            logger.info(f"✅ Relatório criado para vídeo {video_id}: {len(positive_detections)} positivas, {len(negative_detections)} negativas")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Erro ao criar relatório: {e}")
            raise
    
    def _count_by_class(self, detections: List[Dict]) -> Dict[str, int]:
        """Conta detecções por classe"""
        counts = defaultdict(int)
        for det in detections:
            class_name = det.get('class_name', 'unknown')
            # Para detecções negativas (missing-*), contar o EPI faltando
            if class_name.startswith('missing-'):
                epi_name = class_name.replace('missing-', '')
                counts[f'missing-{epi_name}'] += 1
            else:
                counts[class_name] += 1
        return dict(counts)
    
    def _calculate_epi_statistics(self, positive_detections: List[Dict], negative_detections: List[Dict]) -> Dict[str, Dict]:
        """Calcula estatísticas detalhadas por tipo - DINÂMICO (sem hardcode)"""
        epi_stats = {}
        
        # Contar positivas por classe (qualquer classe que o modelo detectar)
        for det in positive_detections:
            class_name = det.get('class_name', '')
            # Ignorar apenas 'person' e classes que começam com 'missing-'
            if class_name != 'person' and not class_name.startswith('missing-'):
                if class_name not in epi_stats:
                    epi_stats[class_name] = {'present': 0, 'missing': 0}
                epi_stats[class_name]['present'] += 1
        
        # Contar negativas por classe
        for det in negative_detections:
            missing_epi = det.get('missing_epi') or det.get('class_name', '').replace('missing-', '')
            if missing_epi:
                if missing_epi not in epi_stats:
                    epi_stats[missing_epi] = {'present': 0, 'missing': 0}
                epi_stats[missing_epi]['missing'] += 1
        
        return epi_stats
    
    def _save_report_json(self, video_id: str, report: Dict[str, Any]):
        """Salva relatório em JSON"""
        report_path = self.reports_dir / f"{video_id}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def get_report(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Obtém relatório de um vídeo"""
        return self.reports.get(video_id)
    
    def export_csv(self, video_id: str, output_path: Optional[str] = None) -> str:
        """
        Exporta relatório para CSV
        
        Args:
            video_id: ID do vídeo
            output_path: Caminho de saída (opcional)
            
        Returns:
            Caminho do arquivo CSV criado
        """
        report = self.get_report(video_id)
        if not report:
            raise ValueError(f"Relatório não encontrado para vídeo {video_id}")
        
        if output_path is None:
            output_path = str(self.reports_dir / f"{video_id}_report.csv")
        
        # Preparar dados para CSV
        csv_rows = []
        
        # Adicionar detecções positivas
        for det in report.get('positive_detections', []):
            csv_rows.append({
                'Tipo': 'Positiva',
                'Frame': det['frame_number'],
                'Timestamp (s)': f"{det['timestamp']:.2f}",
                'Classe': det['class_name'],
                'EPI Ausente': '',
                'Confiança': f"{det['confidence']:.3f}",
                'BBox X1': det['bbox'][0] if len(det['bbox']) > 0 else '',
                'BBox Y1': det['bbox'][1] if len(det['bbox']) > 1 else '',
                'BBox X2': det['bbox'][2] if len(det['bbox']) > 2 else '',
                'BBox Y2': det['bbox'][3] if len(det['bbox']) > 3 else ''
            })
        
        # Adicionar detecções negativas
        for det in report.get('negative_detections', []):
            csv_rows.append({
                'Tipo': 'Negativa',
                'Frame': det['frame_number'],
                'Timestamp (s)': f"{det['timestamp']:.2f}",
                'Classe': det['class_name'],
                'EPI Ausente': det.get('missing_epi', ''),
                'Confiança': f"{det['confidence']:.3f}",
                'BBox X1': det['bbox'][0] if len(det['bbox']) > 0 else '',
                'BBox Y1': det['bbox'][1] if len(det['bbox']) > 1 else '',
                'BBox X2': det['bbox'][2] if len(det['bbox']) > 2 else '',
                'BBox Y2': det['bbox'][3] if len(det['bbox']) > 3 else ''
            })
        
        # Ordenar por frame e timestamp
        csv_rows.sort(key=lambda x: (x['Frame'], float(x['Timestamp (s)'])))
        
        # Escrever CSV
        fieldnames = ['Tipo', 'Frame', 'Timestamp (s)', 'Classe', 'EPI Ausente', 'Confiança', 
                     'BBox X1', 'BBox Y1', 'BBox X2', 'BBox Y2']
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        logger.info(f"✅ CSV exportado: {output_path} ({len(csv_rows)} linhas)")
        
        return output_path
    
    def list_reports(self) -> List[Dict[str, Any]]:
        """Lista todos os relatórios disponíveis"""
        reports_list = []
        for video_id, report in self.reports.items():
            reports_list.append({
                'video_id': video_id,
                'created_at': report.get('created_at'),
                'total_detections': report.get('total_detections', 0),
                'positive_count': len(report.get('positive_detections', [])),
                'negative_count': len(report.get('negative_detections', [])),
                'compliance_score': report.get('statistics', {}).get('compliance_score', 0.0)
            })
        return reports_list

# Instância global
video_report_system = VideoReportSystem()

