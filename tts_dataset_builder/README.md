# XTTS v2 Fine-Tuning - Voz Braum

Projeto completo para fine-tuning de voz com XTTS v2 (Coqui TTS).
Treina um modelo TTS customizado com sua propria voz e gera sintese de fala em português.

## 1) Estrutura

```text
tts_dataset_builder/
├── install.py          # Setup automatico de dependencias
├── prepare_dataset.py  # Validacao e normalizacao do dataset
├── train_xtts.py       # Fine-tuning do modelo XTTS v2
├── inference.py        # Gerar audio sintetizado
├── requirements.txt
│
├── dataset/            # Seu dataset local
│   ├── wavs/           # WAVs mono 24kHz
│   │   ├── 0001.wav
│   │   ├── 0002.wav
│   │   └── ...
│   └── metadata.csv    # 0001|Texto da fala
│
├── output/             # Saidas geradas
├── checkpoints/        # Modelos salvos durante treino
└── models/             # Cache do XTTS v2
```

## 2) Setup rapido (Windows)

No PowerShell, dentro da pasta do projeto:

```powershell
python install.py
```

Esse script:
- Atualiza pip, setuptools, wheel
- Detecta CUDA automaticamente
- Instala PyTorch com CUDA (ou CPU se indisponivel)
- Instala todas as dependencias
- Valida instalacao

## 3) Preparar dataset

Seu dataset ja deve estar em:
- `dataset/wavs/0001.wav`, `0002.wav`, etc.
- `dataset/metadata.csv` com formato: `0001|Texto da fala`

Para validar e normalizar:

```powershell
python prepare_dataset.py
```

Isso:
- Valida todos os WAVs
- Converte para 24kHz mono PCM16 (requisito XTTS)
- Detecta e relata problemas
- Gera relatorio final

## 4) Fine-tunar modelo

Depois que o dataset estiver validado:

```powershell
python train_xtts.py
```

Isso:
- Carrega modelo XTTS v2 base
- Fine-tuna com seu dataset
- Salva checkpoints em `checkpoints/`
- Mostra progresso e loss
- Detecta automaticamente CUDA/CPU

Parametros principais (edite em train_xtts.py):
- `epochs=20`: numero de epocas (ajuste conforme overfitting)
- `learning_rate=1e-4`: taxa de aprendizado
- `batch_size=4`: tamanho do batch (reduza se falta memoria)

## 5) Gerar audio sintetizado

Com modelo treinado, gere sintese de fala:

```powershell
python inference.py
```

Isso:
- Carrega modelo XTTS v2 fine-tunado
- Sintetiza: "Boa tarde, Torneirinha"
- Salva em `output/output.wav`
- Mostra tempo de processamento

## 6) Customizar inferencia

Para mudar o texto gerado, edite `inference.py`:

```python
text = "Boa tarde, Torneirinha"  # Mude aqui
```

## 7) Requisitos de hardware

**Minimo:**
- CPU: Intel i5 ou equivalente
- RAM: 8GB

**Recomendado (GPU NVIDIA):**
- GPU: NVIDIA RTX 3060 ou melhor
- VRAM: 6GB minimo
- Driver NVIDIA: versao recente
- CUDA Toolkit 11.8+

Com GPU: treino ~5-10 min (20 epocas)
Sem GPU: treino ~30-60 min (muito mais lento)

## 8) Troubleshooting

**Erro: "CUDA nao detectado"**
- Instale driver NVIDIA e CUDA Toolkit
- Ou rode novamente `python install.py` (vai reinstalar PyTorch)

**Erro: "CUDA out of memory"**
- Reduza `batch_size` em `train_xtts.py`
- Ou reduza `epochs`

**Dataset validation falha**
- Confirme que todos os WAVs existem
- Confirme que `metadata.csv` esta no formato correto: `0001|texto`
- Confirme encoding UTF-8

**Inferencia lenta**
- Normal na CPU (~30s)
- Com GPU: <5s por sentenca
- Se muito lenta, verifique se GPU foi detectada

## 9) Adicionar novos WAVs

1. Coloque novos WAVs em `dataset/wavs/`
   - Nome: `0089.wav`, `0090.wav`, etc. (sequencial)
2. Adicione linhas em `dataset/metadata.csv`
   - Exemplo: `0089|Novo texto da fala`
3. Rode `python prepare_dataset.py` novamente
4. Rode `python train_xtts.py` para retreinar
5. Rode `python inference.py` para testar

## 10) Dicas de qualidade

- WAVs devem estar limpos (sem ruido de fundo muito alto)
- Duracao minima: ~1s por clip
- Melhor qualidade com 50+ clips
- Texto deve estar correto (sem typos)
- Linguagem consistente (PT-BR recomendado)
- Volume consistente entre clips
- Evitar clipping (distorcao) nos WAVs

## 11) Output final

Apos treino e inferencia:
- `output/output.wav`: Audio sintetizado (44100Hz)
- `checkpoints/best_model.pt`: Melhor modelo salvo
- `checkpoints/epoch_*.pt`: Checkpoints periodicos

## 12) Proximos passos

Depois de gerar `output.wav`, voce pode:
- Integrar em aplicacoes de sintese de fala
- Fazer inference com textos diferentes (editar `inference.py`)
- Treinar novamente com mais dados para melhorar qualidade
- Exportar modelo para usar em outras ferramentas

---

**Criado para: Fine-tuning de voz personalizacao com XTTS v2 (Coqui TTS)**
**Compativel com: Windows 10/11, GPU NVIDIA CUDA, CPU Fallback**
