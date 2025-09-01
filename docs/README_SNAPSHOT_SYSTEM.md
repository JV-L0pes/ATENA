# 📸 Sistema de Snapshot Automático para Violações de EPI

Sistema inteligente que monitora automaticamente violações de EPIs (Equipamentos de Proteção Individual) em canteiros de obra e tira snapshots após um período de paciência configurável.

## 🎯 Funcionalidades Principais

### ⏰ Sistema de Paciência
- **Monitoramento Inteligente**: Detecta quando um funcionário remove um EPI
- **Período de Paciência**: Aguarda 3 segundos (configurável) para o funcionário recolocar o EPI
- **Snapshot Automático**: Se o EPI não for recolocado, tira uma foto automaticamente
- **Cancelamento de Timer**: Se o EPI for recolocado, cancela o timer

### 📸 Captura de Imagens
- **Anotações Automáticas**: Adiciona informações de violação na imagem
- **Timestamp**: Inclui data e hora exata da violação
- **Bordas de Alerta**: Imagens com bordas vermelhas para identificação rápida
- **Organização**: Salva em diretório estruturado com nomes descritivos

### 📧 Notificações Automáticas
- **Email**: Envia alertas com imagem anexada
- **WhatsApp**: Notificações via API de mensageria
- **Detalhes Completos**: Inclui tipo de EPI, pessoa, timestamp e duração

## 🚀 Como Usar

### 1. Execução Principal
```bash
# Executa o sistema completo com snapshot integrado
python src/real_time_epi_detection.py
```

### 2. Controles da Interface
Durante a execução, use as seguintes teclas:

- **N**: Mostra status do sistema de snapshot
- **V**: Visualiza histórico de snapshots
- **T**: Configura tempo de paciência
- **Q**: Sair do sistema
- **S**: Salvar screenshot manual
- **C**: Ajustar configurações de câmera
- **A**: Alternar ambiente
- **O**: Otimizar para EPIs
- **I**: Mostrar informações do sistema

### 3. Teste do Sistema
```bash
# Testa apenas o sistema de snapshot
python test_snapshot_system.py
```

## ⚙️ Configuração

### Arquivo de Configuração
Edite `config/epi_snapshot_config.yaml` para personalizar:

```yaml
# Configurações gerais
patience_period: 3.0  # Segundos de paciência
snapshot_dir: "snapshots"  # Diretório para salvar

# Email
email:
  enabled: true
  smtp_server: "smtp.gmail.com"
  username: "seu_email@gmail.com"
  password: "sua_senha_app"
  recipients:
    - "supervisor@empresa.com"

# WhatsApp
whatsapp:
  enabled: false
  api_key: "sua_api_key"
  phone_numbers:
    - "+5511999999999"
```

### Configuração de Email (Gmail)
1. Ative autenticação de 2 fatores na sua conta Google
2. Gere uma senha de app específica
3. Use essa senha no arquivo de configuração (NÃO sua senha normal)

## 📁 Estrutura de Arquivos

```
snapshots/
├── epi_violation_helmet_12345_20241215_143022.jpg
├── epi_violation_vest_67890_20241215_143045.jpg
└── ...

logs/
├── epi_snapshot.log
└── real_time_epi.log

config/
├── epi_snapshot_config.yaml
└── epi_config_v2.yaml
```

## 🔍 Como Funciona

### 1. Detecção de Violação
```
Funcionário remove EPI → Sistema detecta → Inicia timer de 3s
```

### 2. Período de Paciência
```
Timer ativo → Sistema aguarda → Funcionário pode recolocar EPI
```

### 3. Ação Automática
```
Se EPI não recolocado → Snapshot tirado → Notificações enviadas
Se EPI recolocado → Timer cancelado → Monitoramento continua
```

### 4. Processamento
```
Snapshot salvo → Imagem anotada → Histórico atualizado → Notificações enviadas
```

## 📊 Monitoramento em Tempo Real

### Status do Sistema
- **Monitoramento**: Ativo/Inativo
- **Violações Ativas**: Lista de pessoas com timers ativos
- **Tempo Restante**: Contagem regressiva para cada violação
- **Total de Snapshots**: Histórico completo

### Informações na Tela
- Contador de snapshots em tempo real
- Status de violações ativas
- Período de paciência configurado
- Diretório de armazenamento

## 🧪 Testes e Validação

### Script de Teste
O arquivo `test_snapshot_system.py` testa:

1. **Violação de Capacete**: Simula pessoa sem capacete
2. **Violação de Colete**: Simula pessoa sem colete
3. **Cancelamento de Timer**: Simula EPI sendo recolocado
4. **Múltiplas Pessoas**: Testa cenários complexos

### Executar Testes
```bash
python test_snapshot_system.py
```

### Resultados Esperados
- Snapshots tirados após período de paciência
- Timers cancelados quando EPIs recolocados
- Imagens salvas com anotações corretas
- Histórico atualizado adequadamente

## 🔧 Personalização

### Tempo de Paciência
- **Padrão**: 3 segundos
- **Mínimo**: 1 segundo
- **Máximo**: 60 segundos
- **Alteração**: Durante execução com tecla **T**

### Anotações nas Imagens
- **Texto de Violação**: Tipo de EPI violado
- **Timestamp**: Data e hora exata
- **Borda Vermelha**: Identificação visual
- **Qualidade**: Configurável (1-100)

### Diretórios
- **Snapshots**: Imagens de violações
- **Backup**: Cópias de segurança
- **Logs**: Registros de atividade

## 📈 Estatísticas e Relatórios

### Métricas Coletadas
- Total de snapshots por sessão
- Violações por tipo de EPI
- Tempo médio de violação
- Taxa de cancelamento (EPIs recolocados)

### Histórico
- Últimos 50 snapshots por padrão
- Informações completas de cada violação
- Arquivos organizados por data/hora
- Metadados estruturados

## 🚨 Alertas e Notificações

### Email
- **Assunto**: "ALERTA: Violação de EPI - [TIPO]"
- **Conteúdo**: Detalhes da violação
- **Anexo**: Imagem da violação
- **Destinatários**: Lista configurável

### WhatsApp
- **Mensagem**: Formato estruturado com emojis
- **Imagem**: Snapshot da violação
- **API**: Integração com serviços externos
- **Números**: Lista configurável

## 🔒 Segurança e Privacidade

### Proteção de Dados
- **Logs de Acesso**: Registro de visualizações
- **Backup Automático**: Cópias de segurança
- **Limpeza Automática**: Remoção de arquivos antigos
- **Criptografia**: Opcional para imagens

### Controle de Acesso
- **Diretórios Protegidos**: Snapshots em local seguro
- **Logs de Auditoria**: Rastreamento de atividades
- **Configurações Seguras**: Arquivos de configuração protegidos

## 🛠️ Solução de Problemas

### Problemas Comuns

#### Snapshots não sendo tirados
- Verifique se o sistema está monitorando
- Confirme o período de paciência
- Verifique logs para erros

#### Notificações não enviadas
- Confirme configurações de email/WhatsApp
- Verifique credenciais
- Teste conectividade de rede

#### Imagens corrompidas
- Verifique espaço em disco
- Confirme permissões de diretório
- Teste com período de paciência maior

### Logs de Diagnóstico
```bash
# Ver logs em tempo real
tail -f epi_snapshot.log

# Ver logs de erro
grep "ERROR" epi_snapshot.log

# Ver logs de snapshot
grep "Snapshot salvo" epi_snapshot.log
```

## 📋 Requisitos do Sistema

### Software
- Python 3.8+
- OpenCV 4.5+
- PyYAML
- Bibliotecas de email padrão

### Hardware
- Câmera compatível com OpenCV
- Espaço em disco para snapshots
- Memória RAM: 2GB+ recomendado

### Sistema Operacional
- Windows 10/11
- Linux (Ubuntu 18.04+)
- macOS 10.14+

## 🔄 Atualizações e Manutenção

### Limpeza Automática
- **Snapshots Antigos**: Removidos após 30 dias (configurável)
- **Logs Rotativos**: Backup automático de logs
- **Cache Limpo**: Limpeza periódica de arquivos temporários

### Backup
- **Frequência**: Diária (configurável)
- **Retenção**: 7 dias (configurável)
- **Compressão**: Automática para economizar espaço

## 📞 Suporte

### Documentação
- Este README
- Comentários no código
- Logs detalhados
- Configurações exemplos

### Contato
Para dúvidas ou problemas:
1. Verifique os logs
2. Execute os testes
3. Consulte a documentação
4. Verifique configurações

---

**🎯 LEMBRE-SE: O sistema é projetado para dar uma chance ao funcionário de recolocar o EPI antes de documentar a violação!**
