# Configuração CPU/GPU - Sistema Athena

## Visão Geral

O sistema Athena agora suporta configuração flexível de dispositivos de processamento, permitindo escolher entre CPU, GPU ou detecção automática baseada na disponibilidade de hardware.

## Opções de Configuração

### 1. Modo Automático (Padrão)
```yaml
model:
  device: "auto"
  force_cpu_only: false
```
- Detecta automaticamente se CUDA está disponível
- Usa GPU se disponível, senão usa CPU
- Recomendado para a maioria dos casos

### 2. Modo CPU Forçado
```yaml
model:
  device: "cpu"
  force_cpu_only: true
```
- Força uso de CPU mesmo com GPU disponível
- Útil para sistemas com GPU limitada ou problemas de compatibilidade
- Mais lento mas mais estável

### 3. Modo GPU Forçado
```yaml
model:
  device: "cuda"
  force_cpu_only: false
```
- Força uso de GPU CUDA
- Falha se CUDA não estiver disponível
- Máxima performance se GPU estiver disponível

## Configuração via Variáveis de Ambiente

### Backend
```bash
# Forçar CPU apenas
export FORCE_CPU_ONLY=true

# Preferência de dispositivo
export DEVICE_PREFERENCE=cpu  # ou cuda, auto
```

### Frontend
As configurações podem ser alteradas através da interface web:
1. Acesse a aba "Config"
2. Na seção "Configurações de Performance"
3. Selecione o dispositivo desejado
4. Marque "Forçar CPU" se necessário

## Exemplos de Uso

### Para Sistemas com GPU Limitada
```bash
# Via variável de ambiente
export FORCE_CPU_ONLY=true
python start_api.py
```

### Para Sistemas Sem GPU
```bash
# Via variável de ambiente
export DEVICE_PREFERENCE=cpu
python start_api.py
```

### Para Máxima Performance
```bash
# Via variável de ambiente (padrão)
export DEVICE_PREFERENCE=auto
python start_api.py
```

## Verificação de Status

O sistema mostra o dispositivo atual sendo usado nos logs:
```
INFO - Usando GPU CUDA (detecção automática)
INFO - Usando CPU (CUDA não disponível)
INFO - Forçando uso de CPU (FORCE_CPU_ONLY=True)
```

## Performance Esperada

| Dispositivo | FPS Esperado | Uso de Memória | Estabilidade |
|-------------|--------------|----------------|--------------|
| GPU (CUDA)  | 30-60 FPS    | Alta           | Boa          |
| CPU         | 5-15 FPS     | Baixa          | Excelente    |

## Troubleshooting

### Problema: GPU não detectada
**Solução**: Verifique se CUDA está instalado corretamente
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Problema: Sistema muito lento
**Solução**: Use CPU apenas se GPU estiver causando problemas
```bash
export FORCE_CPU_ONLY=true
```

### Problema: Erro de memória GPU
**Solução**: Force uso de CPU
```bash
export DEVICE_PREFERENCE=cpu
```

## Arquivos de Configuração

- `config/config.yaml` - Configuração principal
- `config/cpu_only_example.yaml` - Exemplo para CPU apenas
- `backend/config.py` - Configurações do backend
- `src/epi_detector.py` - Lógica de seleção de dispositivo
