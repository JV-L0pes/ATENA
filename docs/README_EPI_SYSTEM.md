# 🚧 Sistema de Detecção Automática de EPIs em Canteiros de Obra

Sistema completo de visão computacional para detectar e validar Equipamentos de Proteção Individual (EPIs) em canteiros de obra, com foco na **validação física** dos equipamentos na pessoa.

## 🎯 OBJETIVO PRINCIPAL

**DETECTAR E VALIDAR SE OS EPIs ESTÃO FISICAMENTE NA PESSOA:**

- ✅ **CAPACETE**: Deve estar na cabeça (não na mão, no chão, pendurado)
- ✅ **COLETE**: Deve estar no torso (não pendurado, dobrado, no chão)
- ❌ **INFRINGIMENTO**: EPI presente na imagem mas não na pessoa

## 🏗️ ARQUITETURA DO SISTEMA

### Componentes Principais

1. **`epi_detector_v2.py`** - Detector principal de EPIs com validação física
2. **`construction_camera_system.py`** - Sistema de câmera otimizado para canteiros
3. **`real_time_epi_detection.py`** - Sistema principal de detecção em tempo real
4. **`epi_config_v2.yaml`** - Configurações do sistema

### Funcionalidades

- **Detecção em Tempo Real**: Monitoramento contínuo via webcam
- **Validação Física**: Verifica se EPIs estão realmente na pessoa
- **Configurações de Câmera**: Otimizadas para diferentes ambientes
- **Sistema de Alertas**: Notifica violações de EPIs
- **Estatísticas**: Acompanha compliance e violações
- **Screenshots**: Captura imagens de detecção
- **Logs**: Registra todas as atividades

## 🚀 COMO USAR

### 1. Teste do Sistema

```bash
# Testa se todos os componentes estão funcionando
py test_epi_system.py
```

### 2. Execução Principal

```bash
# Inicia o sistema de detecção em tempo real
py src/real_time_epi_detection.py
```

### 3. Controles Durante Execução

- **Q**: Sair do sistema
- **S**: Salvar screenshot
- **C**: Ajustar configurações de câmera
- **A**: Alternar ambiente (indoor/outdoor/sunny/etc.)
- **O**: Otimizar câmera para detecção de EPIs
- **I**: Mostrar informações do sistema

## ⚙️ CONFIGURAÇÕES

### Ambientes Suportados

- **`sunny_outdoor`**: Ambiente externo ensolarado
- **`cloudy_outdoor`**: Ambiente externo nublado (padrão)
- **`indoor`**: Ambiente interno
- **`low_light`**: Baixa luminosidade

### Parâmetros de Câmera

- **Resolução**: 1280x720 (HD otimizado)
- **FPS**: 30
- **Brilho**: 60 (aumentado para ambientes externos)
- **Contraste**: 45 (reduzido para evitar sombras)
- **Exposição**: 1 (ligeiramente positiva)
- **Saturação**: 55 (aumentada para cores dos EPIs)

## 🔧 VALIDAÇÃO FÍSICA DOS EPIs

### Regras de Validação

#### Capacete
- **Zona da Cabeça**: Topo 25% da altura da pessoa
- **Sobreposição Mínima**: 30% (IoU)
- **Posicionamento**: Deve estar fisicamente na cabeça

#### Colete
- **Zona do Torso**: 30-70% da altura da pessoa
- **Sobreposição Mínima**: 25% (IoU)
- **Cobertura Mínima**: 40% do torso
- **Posicionamento**: Deve estar fisicamente no torso

### Status de Compliance

- **🟢 COMPLIANT**: Capacete e colete corretos
- **🟡 PARCIAL**: Apenas um EPI correto
- **🔴 VIOLATION**: Nenhum EPI correto

## 📊 ESTATÍSTICAS E RELATÓRIOS

### Métricas Coletadas

- Total de pessoas detectadas
- Taxa de compliance
- Violações por tipo (capacete/colete)
- FPS e performance
- Histórico de detecções

### Arquivos Gerados

- **`screenshots/`**: Imagens de detecção
- **`alerts/`**: Alertas de violação (JSON)
- **`statistics/`**: Estatísticas da sessão
- **`*.log`**: Logs de execução

## 🛠️ INSTALAÇÃO E DEPENDÊNCIAS

### Requisitos

- Python 3.8+
- OpenCV 4.5+
- PyTorch
- Ultralytics
- PyYAML

### Instalação

```bash
# Ativa ambiente virtual (se existir)
venv_39\Scripts\activate

# Instala dependências
pip install -r requirements.txt
```

## 🔍 DETECÇÃO YOLO

### Modelo Utilizado

- **Arquivo**: `yolov5/runs/train/epi_detection_v24/weights/best.pt`
- **Classes**: 5 (helmet, no-helmet, no-vest, person, vest)
- **Dataset**: Treinado com imagens de canteiros de obra

### Configurações do Modelo

- **Confiança Mínima**: 0.5
- **IoU Threshold**: 0.45
- **Dispositivo**: CUDA (GPU) por padrão

## 🚨 SISTEMA DE ALERTAS

### Configurações

- **Threshold de Violação**: 3 violações consecutivas
- **Cooldown**: 30 segundos entre alertas
- **Formato**: JSON com timestamp e detalhes

### Exemplo de Alerta

```json
{
  "timestamp": "2024-01-15T14:30:25",
  "frame": 1250,
  "people_count": 3,
  "violations": 2,
  "details": [
    {
      "person_id": 0,
      "helmet_status": "AUSENTE",
      "vest_status": "CORRETO"
    }
  ]
}
```

## 🌍 AMBIENTES E CONDIÇÕES

### Otimizações por Ambiente

#### Externo Ensolarado
- Brilho reduzido (40)
- Contraste aumentado (55)
- Exposição negativa (-1)

#### Externo Nublado
- Brilho médio (55)
- Contraste neutro (50)
- Exposição neutra (0)

#### Interno
- Brilho aumentado (70)
- Contraste reduzido (40)
- Exposição positiva (2)

#### Baixa Luminosidade
- Brilho máximo (80)
- Contraste mínimo (35)
- Exposição alta (3)
- Ganho aumentado (3)

## 📁 ESTRUTURA DE ARQUIVOS

```
src/
├── epi_detector_v2.py          # Detector principal
├── construction_camera_system.py # Sistema de câmera
├── real_time_epi_detection.py   # Sistema principal
└── main.py                      # Sistema antigo (legado)

config/
├── epi_config_v2.yaml          # Configurações do sistema
└── config.yaml                 # Configurações antigas

screenshots/                     # Screenshots de detecção
alerts/                         # Alertas de violação
statistics/                     # Estatísticas da sessão
```

## 🧪 TESTES E VALIDAÇÃO

### Script de Teste

```bash
py test_epi_system.py
```

### Testes Executados

1. **OpenCV**: Funcionamento da biblioteca
2. **Módulos Python**: Dependências necessárias
3. **Arquivos de Configuração**: Presença dos YAMLs
4. **Modelo YOLO**: Disponibilidade do modelo treinado
5. **Componentes do Sistema**: Importação dos módulos

## 🚀 EXECUÇÃO RÁPIDA

```bash
# 1. Testa o sistema
py test_epi_system.py

# 2. Se todos os testes passarem, executa
py src/real_time_epi_detection.py

# 3. Use os controles para ajustar
# C: Configurações de câmera
# A: Alternar ambiente
# O: Otimizar para EPIs
```

## 🔧 SOLUÇÃO DE PROBLEMAS

### Câmera Escura

- Use **C** para ajustar configurações
- Alternar para ambiente **indoor** ou **low_light**
- Pressione **O** para otimização automática

### Baixa Performance

- Verifique se GPU está sendo usada
- Reduza resolução da câmera
- Ajuste thresholds de confiança

### Falsos Positivos

- Ajuste thresholds de sobreposição no config
- Configure zona útil do canteiro
- Use regras de validação mais estritas

## 📈 MELHORIAS FUTURAS

- [ ] Tracking entre frames
- [ ] Interface web para monitoramento
- [ ] Integração com sistemas de segurança
- [ ] Análise de padrões de violação
- [ ] Relatórios automáticos
- [ ] Notificações em tempo real

## 📞 SUPORTE

Para dúvidas ou problemas:

1. Execute `py test_epi_system.py` para diagnóstico
2. Verifique os logs em `*.log`
3. Consulte as configurações em `config/`
4. Teste com diferentes ambientes

---

**🎯 LEMBRE-SE: O objetivo é validar se os EPIs estão FISICAMENTE na pessoa, não apenas se estão presentes na imagem!**
