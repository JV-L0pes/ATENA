# Atena - Sistema de Detecção de EPIs

Sistema de visão computacional para detecção e validação de Equipamentos de Proteção Individual (EPIs) usando YOLOv5n.

## Funcionalidades

- **Detecção de pessoas** como âncora principal
- **Detecção de EPIs**: Capacete e colete
- **Validação de posicionamento**: EPIs devem estar na área adequada
- **Sistema de alertas**: Boxes verdes (EPI correto) e vermelhos (infração)

## Regras de Validação

- **Capacete**: Deve estar na cabeça da pessoa
- **Colete**: Deve estar vestido no corpo da pessoa
- **Infrações**: EPI na mão, chão, cintura ou qualquer posição inadequada

## Estrutura do Projeto

```
Atena/
├── datasets/           # Datasets para treinamento
├── models/            # Modelos YOLO treinados
├── src/               # Código fonte
├── config/            # Configurações
├── utils/             # Utilitários
└── tests/             # Testes
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

```bash
py src/main.py --source images/ --weights models/best.pt
```

## Configuração GPU

O projeto está configurado para usar GPU por padrão. Certifique-se de ter CUDA instalado.
