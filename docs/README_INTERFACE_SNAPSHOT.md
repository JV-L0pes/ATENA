# 📸 Sistema de Snapshot Integrado na Interface Gráfica

O sistema de snapshot automático para violações de EPI foi **totalmente integrado** na interface gráfica principal! 🎯

## 🚀 **Como Usar**

### 1. **Execute a Interface Principal**
```bash
python src/epi_detection_interface.py
```

### 2. **Controles do Sistema de Snapshot**
A interface agora possui uma **seção dedicada** ao sistema de snapshot com os seguintes botões:

- **🔄 INICIAR SNAPSHOT**: Inicia/para o sistema de snapshot
- **📊 STATUS**: Mostra status detalhado do sistema
- **📚 HISTÓRICO**: Visualiza histórico de snapshots tirados
- **⏰ CONFIGURAR**: Ajusta tempo de paciência

### 3. **Configurações**
- **Tempo de Paciência**: Slider de 1 a 10 segundos (padrão: 3s)
- **Qualidade da Imagem**: Configurável via código
- **Diretório**: Snapshots salvos em `snapshots/`

## 🎯 **Funcionamento Automático**

### **Detecção de Violações**
1. **Carregue uma imagem** com pessoas sem EPIs
2. **Clique em "🔄 INICIAR SNAPSHOT"**
3. **Processe a imagem** com "📸 DETECTAR IMAGEM"
4. **O sistema detecta automaticamente** violações de EPIs

### **Sistema de Paciência**
- **Timer inicia** quando violação é detectada
- **Aguarda 3 segundos** (configurável) para o funcionário recolocar o EPI
- **Snapshot automático** se o EPI não for recolocado
- **Timer cancela** se o EPI for recolocado

### **Snapshots Automáticos**
- **Imagens anotadas** com informações da violação
- **Bordas vermelhas** para identificação
- **Timestamp** automático
- **Salvos em** `snapshots/epi_violation_[tipo]_[pessoa]_[timestamp].jpg`

## 📊 **Informações na Tela**

### **Labels de Status**
- **📸 Snapshots**: Contador de snapshots tirados
- **🚨 Violações Ativas**: Violações com timers ativos
- **⏰ Tempo de Paciência**: Configuração atual

### **Informações na Imagem**
- **Status dos EPIs** com cores e símbolos
- **✅ Verde**: EPIs corretos
- **❌ Vermelho**: EPIs ausentes
- **⏳ Ciano**: EPIs parciais

### **Overlay do Sistema**
- **Status do snapshot** (ATIVO/INATIVO)
- **Contadores** de snapshots e violações
- **Tempo de paciência** configurado
- **Controles** disponíveis

## 🔧 **Configurações Avançadas**

### **Alterar Tempo de Paciência**
1. **Clique em "⏰ CONFIGURAR"**
2. **Digite novo valor** (1-60 segundos)
3. **Clique em "✅ APLICAR"**

### **Via Slider**
- **Arraste o slider** "⏰ Paciência (s)"
- **Mudanças aplicadas** automaticamente

## 📁 **Estrutura de Arquivos**

```
📁 snapshots/
├── epi_violation_helmet_12345_20241215_143022.jpg
├── epi_violation_vest_67890_20241215_143045.jpg
└── ...

📁 output/
├── epi_detection_image_20241215_143022.jpg
└── ...
```

## 🧪 **Teste da Integração**

Execute o script de teste para verificar se tudo está funcionando:

```bash
python test_interface_snapshot.py
```

## 🎬 **Sistema Completo**

### **Passo a Passo**
1. **Execute a interface**: `python src/epi_detection_interface.py`
2. **Clique em "🔄 INICIAR SNAPSHOT"**
3. **Carregue uma imagem** com pessoas sem EPIs
4. **Clique em "📸 DETECTAR IMAGEM"**
5. **Aguarde 3 segundos** - o snapshot será tirado automaticamente!
6. **Verifique o diretório** `snapshots/`

### **Resultado Esperado**
- **Imagem processada** com detecções
- **Status dos EPIs** visível na tela
- **Snapshot automático** após período de paciência
- **Arquivo salvo** com anotações

## 🚨 **Importante**

- **Não precisa fazer nada manual** - o sistema funciona sozinho
- **Snapshots são tirados automaticamente** após o período de paciência
- **Imagens são anotadas** com informações da violação
- **Histórico é mantido** para auditoria
- **Interface atualizada** em tempo real

## 🔍 **Solução de Problemas**

### **Sistema não inicia**
- Verifique se clicou em "🔄 INICIAR SNAPSHOT"
- Confirme se o botão mostra "⏹️ PARAR SNAPSHOT"

### **Snapshots não sendo tirados**
- Verifique se o sistema está "ATIVO"
- Confirme o período de paciência configurado
- Verifique se há violações sendo detectadas

### **Diretório não criado**
- O diretório `snapshots/` é criado automaticamente
- Verifique permissões de escrita no diretório atual

## 🎉 **Pronto para Uso!**

O sistema de snapshot está **100% integrado** na interface gráfica e funciona automaticamente! 

**Basta executar a interface e clicar em "🔄 INICIAR SNAPSHOT"** para começar a monitorar violações de EPIs em tempo real! 🚀
