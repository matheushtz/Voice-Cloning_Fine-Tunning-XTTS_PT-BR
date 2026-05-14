# GUIA DE USO - XTTS v2 Fine-Tuning

## Seu dataset está pronto!

Você tem:
- `dataset/metadata.csv`: 88 linhas de frases do Braum
- `dataset/wavs/0001.wav` até `0088.wav`: arquivos de áudio

Agora vamos treinar uma voz TTS personalizada.

## Passo 1: Setup (1ª vez apenas)

Abra PowerShell na pasta do projeto e execute:

```powershell
python install.py
```

Isso vai:
- Instalar PyTorch com CUDA (se tiver GPU NVIDIA)
- Instalar Coqui TTS
- Baixar o modelo XTTS v2 base
- Verificar tudo

**Tempo:** 5-15 minutos (depende da internet)

Se vir mensagens de sucesso: ✓ Pronto!

## Passo 2: Preparar dataset

```powershell
python prepare_dataset.py
```

Isso vai:
- Validar cada arquivo WAV
- Converter para 24kHz (requisito XTTS v2)
- Detectar problemas
- Gerar relatório

**Output esperado:**
```
RELATORIO DE VALIDACAO DO DATASET
====================================================
Total de clips: 88
Clips validos: 88
Clips convertidos: XX
Duracao total: XXs

✓ Dataset valido com 88 clips
```

## Passo 3: Treinar modelo (15-60 minutos)

```powershell
python train_xtts.py
```

Isso vai:
- Carregar modelo XTTS v2 base
- Fine-tunar com seus 88 clips do Braum
- Salvar checkpoints em `checkpoints/`
- Mostrar loss do treino

**Com GPU NVIDIA:** ~10-20 minutos
**Só CPU:** ~45-90 minutos

**Output esperado:**
```
Epoca 1/20
Train Loss: 2.5432
Val Loss: 2.1234
✓ Melhor modelo salvo (loss: 2.1234)
...
Epoca 20/20
====================================================
FINE-TUNING CONCLUIDO
Melhor validacao loss: 1.2345
Checkpoints salvos em: checkpoints/
====================================================
```

## Passo 4: Gerar áudio (30 segundos a 2 minutos)

```powershell
python inference.py
```

Isso vai:
- Carregar modelo treinado
- Sintetizar: "Boa tarde, Torneirinha"
- Salvar em `output/output.wav`

**Output esperado:**
```
XTTS v2 - INFERENCIA COM VOZ FINE-TUNADA
====================================================
Texto: Boa tarde, Torneirinha
Checkpoints: checkpoints/
Carregando modelo XTTS v2 base...
Pesos fine-tunados carregados com sucesso
Modelo pronto para inferencia

Sintetizando: Boa tarde, Torneirinha
Tempo de sintese: 8.32s
Audio salvo: output/output.wav
Sample rate: 24000Hz
Duracao: 2.41s

====================================================
SUCESSO!
Output: output/output.wav
====================================================
```

## Quick Start (todos os passos)

Ou execute tudo de uma vez:

```powershell
python quickstart.py
```

Isso executa install → prepare → train → inference automaticamente.

## Troubleshooting

### "Erro: module not found"
Rode novamente:
```powershell
python install.py
```

### "CUDA out of memory"
Edite `train_xtts.py`, mude batch_size:
```python
trainer.train(
    epochs=20,
    learning_rate=1e-4,
    batch_size=2,  # Reduza de 4 para 2
)
```

### "Dataset validation falha"
Verifique:
- Todos os WAVs existem em `dataset/wavs/`
- `metadata.csv` tem linhas no formato: `0001|texto`
- Encoding UTF-8 (abra em Notepad, File → Save As, UTF-8)

### "Inferencia muito lenta"
- Normal na CPU (~30s por frase)
- Se tem GPU: verifique que foi detectada
  - Edite `inference.py`, rode direto e veja se mostra GPU
  - Pode ser driver NVIDIA desatualizado

## Customizar inferência

Para gerar outro texto, edite `inference.py`:

```python
# Procure por esta linha (no final do arquivo main()):
text = "Boa tarde, Torneirinha"

# Mude para seu texto:
text = "Olá, eu sou o Braum"
```

Depois rode:
```powershell
python inference.py
```

## Próximos passos

1. **Melhorar qualidade:**
   - Adicione mais WAVs ao dataset (50+ é bom)
   - Re-rode treino

2. **Usar em seu app:**
   - `output/output.wav` é o audio gerado
   - Integre com seu sistema

3. **Diferentes idiomas:**
   - Mude `language="pt"` para outro em `inference.py`
   - Compatível com: pt, en, es, fr, de, it, pl, tr, ru, nl, ja, zh, ar, hi, ko

## Estrutura final

Após treino:
```
tts_dataset_builder/
├── dataset/                 # Seu dataset original
│   ├── wavs/0001.wav ... 0088.wav
│   └── metadata.csv
│
├── output/
│   └── output.wav          # ← AUDIO GERADO!
│
├── checkpoints/
│   ├── best_model.pt       # ← Modelo treinado
│   ├── epoch_5.pt
│   ├── epoch_10.pt
│   ├── epoch_15.pt
│   └── epoch_20.pt
│
└── [scripts Python]
```

---

**Pronto!** Você tem um modelo TTS treinado com a voz personalizada. 🎉
