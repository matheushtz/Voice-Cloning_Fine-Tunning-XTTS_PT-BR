# XTTS v2 Fine-Tuning - Início Rápido (a partir da raiz)

## Estrutura
```
TTS-VOICE-TRAINING/
├── dataset/
│   ├── wavs/ (0001.wav até 0088.wav)
│   └── metadata.csv
├── tts_dataset_builder/ (projeto interno)
├── quickstart.py ← Execute a partir daqui
├── install.py
├── prepare_dataset.py
├── train_xtts.py
└── inference.py
```

## Executar (4 passos)

### Passo 1: Instalar dependências
```powershell
python install.py
```

### Passo 2: Validar dataset
```powershell
python prepare_dataset.py
```

### Passo 3: Treinar modelo
```powershell
python train_xtts.py
```

### Passo 4: Gerar áudio
```powershell
python inference.py --text "Seu texto aqui"
```

#### Controlar velocidade da fala
A síntese de fala agora suporta controle nativo de velocidade:

```powershell
# Fala mais rápida (padrão)
python inference.py --text "Seu texto" --speed 1.15

# Fala bem rápida
python inference.py --text "Seu texto" --speed 1.4

# Fala normal
python inference.py --text "Seu texto" --speed 1.0

# Fala mais lenta
python inference.py --text "Seu texto" --speed 0.9
```

**Escala de velocidade:**
- `1.0` = Velocidade normal
- `> 1.0` = Mais rápido (ex: 1.2, 1.4, 1.6)
- `< 1.0` = Mais lento (ex: 0.8, 0.9)

A velocidade padrão é **1.15x** para síntese mais rápida por padrão.

## Quick Start (tudo de uma vez)
```powershell
python quickstart.py
```

## Output
Seu áudio gerado estará em:
```
tts_dataset_builder/output/output.wav
```

## Documentação completa
Veja: [tts_dataset_builder/GETTING_STARTED.txt](tts_dataset_builder/GETTING_STARTED.txt)
