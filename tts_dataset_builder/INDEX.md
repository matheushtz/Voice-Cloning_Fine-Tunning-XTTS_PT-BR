# Índice de Arquivos - XTTS v2 Fine-Tuning

## 📖 Documentação (Comece por aqui!)

| Arquivo | Propósito |
|---------|-----------|
| [USAGE.md](USAGE.md) | ⭐ **Guia passo-a-passo completo** - Comece aqui |
| [README.md](README.md) | Documentação técnica completa |
| [FAQ.md](FAQ.md) | Perguntas frequentes e troubleshooting |
| [EXAMPLES.md](EXAMPLES.md) | Exemplos de código e customizações |
| [INDEX.md](INDEX.md) | Este arquivo |

---

## 🚀 Scripts Principais

Execute nesta ordem:

| # | Script | Descrição | Comando |
|---|--------|-----------|---------|
| 1 | [install.py](install.py) | Instala dependências, detecta CUDA | `python install.py` |
| 2 | [prepare_dataset.py](prepare_dataset.py) | Valida e converte dataset para 24kHz | `python prepare_dataset.py` |
| 3 | [train_xtts.py](train_xtts.py) | Fine-tuna modelo XTTS v2 | `python train_xtts.py` |
| 4 | [inference.py](inference.py) | Gera áudio sintetizado | `python inference.py` |

**Atalho (todos os passos):**
```powershell
python quickstart.py
```

---

## 📁 Estrutura de Pastas

```
tts_dataset_builder/
│
├── 📚 DOCUMENTAÇÃO
│   ├── README.md              # Documentação técnica
│   ├── USAGE.md               # Guia passo-a-passo ⭐
│   ├── FAQ.md                 # Perguntas frequentes
│   ├── EXAMPLES.md            # Exemplos de código
│   └── INDEX.md               # Este arquivo
│
├── 🔧 SCRIPTS PRINCIPAIS
│   ├── install.py             # 1º: Setup de dependências
│   ├── prepare_dataset.py     # 2º: Preparar dataset
│   ├── train_xtts.py          # 3º: Fine-tuning
│   ├── inference.py           # 4º: Gerar áudio
│   └── quickstart.py          # Atalho: executa todos
│
├── 📦 CÓDIGO INTERNO (não modificar)
│   ├── config.py              # Configurações
│   ├── audio_utils.py         # Utilitários de áudio
│   ├── segmenter.py           # Segmentação (legado)
│   ├── transcriber.py         # Transcrição (legado)
│   └── metadata_builder.py    # Builder de metadados (legado)
│
├── 📊 DATA
│   ├── dataset/               # Seu dataset local
│   │   ├── wavs/              # WAVs: 0001.wav, 0002.wav, ...
│   │   └── metadata.csv       # Lista: 0001|texto
│   ├── output/                # Saída gerada
│   │   └── output.wav         # ← Seu audio final!
│   ├── checkpoints/           # Modelos treinados
│   │   ├── best_model.pt
│   │   ├── epoch_5.pt
│   │   └── ...
│   └── models/                # Cache XTTS v2
│
├── 📝 CONFIGURAÇÃO
│   ├── requirements.txt       # Dependências Python
│   └── input/                 # Pasta vazia (legado)
│
└── 🐍 __pycache__/            # Cache Python (ignorar)
```

---

## 🎯 Fluxos de Uso

### Fluxo 1: Quick Start (Recomendado)

```powershell
python quickstart.py
```

Isto:
1. Instala dependências
2. Prepara dataset
3. Treina modelo
4. Gera output.wav

Tempo total: 20-90 min (depende de GPU)

### Fluxo 2: Passo-a-Passo

```powershell
# 1. Instalar (só na primeira vez)
python install.py

# 2. Preparar dataset
python prepare_dataset.py

# 3. Treinar
python train_xtts.py

# 4. Inferência
python inference.py
```

### Fluxo 3: Apenas Inferência (com modelo já treinado)

```powershell
# Edite inference.py se quiser outro texto, depois:
python inference.py
```

### Fluxo 4: Customizar e Retreinar

```powershell
# 1. Edite train_xtts.py para ajustar epochs, batch_size, etc.

# 2. Adicione novos WAVs em dataset/wavs/

# 3. Atualize dataset/metadata.csv

# 4. Rode novamente:
python prepare_dataset.py
python train_xtts.py
python inference.py
```

---

## 🔍 Quando Usar Cada Script

| Preciso fazer... | Use... | Link |
|---|---|---|
| Instalar tudo | `python install.py` | [install.py](install.py) |
| Validar dataset | `python prepare_dataset.py` | [prepare_dataset.py](prepare_dataset.py) |
| Treinar modelo | `python train_xtts.py` | [train_xtts.py](train_xtts.py) |
| Gerar áudio | `python inference.py` | [inference.py](inference.py) |
| Tudo de uma vez | `python quickstart.py` | [quickstart.py](quickstart.py) |

---

## 📚 Documentação por Tópico

### Iniciantes
1. Leia [USAGE.md](USAGE.md) - Passo-a-passo completo
2. Execute `python quickstart.py`
3. Verifique `output/output.wav`

### Troubleshooting
1. Consulte [FAQ.md](FAQ.md)
2. Procure o erro no FAQ
3. Siga a solução

### Customizações
1. Leia [EXAMPLES.md](EXAMPLES.md)
2. Copie código exemplo
3. Adapte para seu caso

### Desenvolvimento
1. Leia [README.md](README.md)
2. Estude os scripts em Python
3. Modifique conforme necessário

---

## ⚡ Requisitos

### Hardware Mínimo
- Windows 10/11
- CPU: i5 ou equivalente
- RAM: 8GB
- SSD: 20GB livre

### Hardware Recomendado (GPU)
- GPU NVIDIA: RTX 3060 ou melhor
- VRAM: 6GB+
- Driver NVIDIA atualizado

### Software
- Python 3.9+
- Conexão de internet (primeira vez)

---

## 📊 Seu Dataset

Estrutura esperada:

```
dataset/
├── wavs/
│   ├── 0001.wav
│   ├── 0002.wav
│   ├── 0003.wav
│   └── ... até 0088.wav
└── metadata.csv
    0001|Coração é o músculo mais forte.
    0002|Por que não ir beber leite quente então?
    0003|Primeiro lutamos, depois comemos.
    ...
```

**Total:** 88 clips do Braum em português 🎤

---

## 🎁 Saída Final

Após completar o pipeline:

```
output/
└── output.wav           # ← Seu áudio gerado!
                         # Contém: "Boa tarde, Torneirinha"
                         # Voz: Braum (fine-tunada)
                         # Formato: WAV 24kHz PCM16
```

Você pode:
- Escutar em qualquer player
- Usar em apps
- Compartilhar
- Editar com Audacity
- Converter para MP3, etc.

---

## 🆘 Precisa de Ajuda?

1. **Erros durante instalação?**
   → Veja [FAQ.md - Instalação](FAQ.md#instalação)

2. **Dataset não valida?**
   → Veja [FAQ.md - Dataset](FAQ.md#dataset)

3. **Treino muito lento ou falha?**
   → Veja [FAQ.md - Treinamento](FAQ.md#treinamento)

4. **Áudio gerado tem qualidade ruim?**
   → Veja [FAQ.md - Qualidade](FAQ.md#qualidade-e-troubleshooting)

5. **Quer customizar código?**
   → Veja [EXAMPLES.md](EXAMPLES.md)

---

## 📝 Próximos Passos

Após gerar `output.wav`:

1. **Testar com outros textos:**
   - Edite [inference.py](inference.py)
   - Mude `text = "Nova frase"`
   - Rode `python inference.py` novamente

2. **Retreinar com mais dados:**
   - Adicione novos WAVs em `dataset/wavs/`
   - Atualize `dataset/metadata.csv`
   - Execute `python prepare_dataset.py` e `train_xtts.py`

3. **Usar em sua aplicação:**
   - Integre [inference.py](inference.py) em seu código
   - Ou copie lógica de síntese
   - Veja [EXAMPLES.md - Integração](EXAMPLES.md#integração-em-aplicações)

4. **Compartilhar modelo:**
   - Os checkpoints estão em `checkpoints/`
   - Você pode compartilhá-los com `best_model.pt`

---

## 📞 Resumo Rápido

| O que fazer | Comando |
|---|---|
| Setup inicial | `python install.py` |
| Executar tudo | `python quickstart.py` |
| Treinar de novo | `python train_xtts.py` |
| Gerar novo áudio | `python inference.py` |
| Validar dados | `python prepare_dataset.py` |

---

**Versão:** XTTS v2 + Coqui TTS
**Compatibilidade:** Windows 10/11 + GPU NVIDIA/CPU
**Linguagem:** Python 3.9+
**Dataset:** 88 clips do Braum em PT-BR 🎤
