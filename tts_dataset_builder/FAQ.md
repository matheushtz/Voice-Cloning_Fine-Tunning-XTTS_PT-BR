# Perguntas Frequentes - XTTS v2 Fine-Tuning

## Instalação

**P: Quanto espaço em disco preciso?**
R: ~15GB no total:
- Modelo XTTS v2: ~5GB
- PyTorch + dependências: ~5GB
- Dataset + checkpoints: ~5GB (varia com quantidade de clips)

**P: Funciona em Windows 10?**
R: Sim, totalmente compatível. Windows 11 também.

**P: Preciso de GPU?**
R: Não é obrigatório, mas recomendado. Treino na GPU é ~5x mais rápido.

**P: Qual versão de Python devo usar?**
R: Python 3.9, 3.10, 3.11 ou 3.12. O script install.py usa a versão que você já tem.

**P: Posso usar GPU AMD ou Intel?**
R: Por enquanto não. XTTS v2 suporta NVIDIA CUDA ou CPU.

---

## Dataset

**P: Qual o sample rate dos WAVs?**
R: XTTS v2 requer 24kHz. O script prepare_dataset.py converte automaticamente.

**P: Posso usar MP3 ou outros formatos?**
R: Sim, librosa suporta vários formatos. Mas salva sempre como WAV 24kHz.

**P: Qual a duração ideal de cada clip?**
R: 1-10 segundos é bom. Mais curto que 1s é descartado automaticamente.

**P: Quantos clips preciso mínimo?**
R: Com 20 clips já dá para treinar. Mas 50+ dá melhor qualidade.

**P: Como adiciono novos clips?**
R: 
1. Coloque WAV em `dataset/wavs/00XX.wav` (com número sequencial)
2. Adicione linha em `metadata.csv`: `00XX|Novo texto`
3. Rode `prepare_dataset.py` novamente
4. Rode `train_xtts.py` para retreinar

**P: Precisa ser a mesma pessoa falando em todos os clips?**
R: Sim, para melhor resultado. Se forem pessoas diferentes, o modelo vai ficar confuso.

**P: Posso usar voz sintetizada/robô?**
R: Tecnicamente sim, mas vai ficar com qualidade pior.

---

## Treinamento

**P: Quanto tempo leva o treino?**
R: 
- Com GPU NVIDIA: 10-30 minutos (20 epocas)
- Só CPU: 1-3 horas

**P: Como saber se está overfitting?**
R: Compare Loss do treino vs validação:
- Normal: ambas diminuem gradualmente
- Overfitting: treino baixa muito, validação aumenta

Se isto acontecer, reduza `epochs` ou aumente `batch_size` em `train_xtts.py`.

**P: Preciso rodar treino múltiplas vezes?**
R: Não. Uma vez está bom. Mas pode retreinar com mais dados depois.

**P: Posso pausar e continuar treino depois?**
R: Não ainda, mas checkpoints são salvos. Deixe rodando.

**P: Qual learning rate é bom?**
R: 1e-4 é conservador e seguro. Se convergir muito rápido, reduza para 1e-5.

**P: Batch size: qual é bom?**
R: 4 é padrão. Se faltar memória, reduza para 2.

---

## Inferência

**P: Qual a qualidade do áudio gerado?**
R: Depende do dataset e treino:
- 20-50 clips: aceitável
- 50+ clips: bom
- 100+ clips: muito bom

**P: Posso gerar áudio de múltiplas frases?**
R: Sim, edite `inference.py` e mude o `text`.

**P: Qual a duração máxima de uma frase?**
R: Sem limite prático, mas frases longas são mais lentas (~5s por 20 palavras).

**P: O áudio sai em qual sample rate?**
R: 24kHz (padrão XTTS v2). Pode converter com librosa/ffmpeg depois.

**P: Posso salvar em MP3?**
R: `inference.py` salva em WAV. Converta depois com ffmpeg:
```bash
ffmpeg -i output.wav -b:a 128k output.mp3
```

---

## Qualidade e Troubleshooting

**P: Frase gerada soa robótica/estranha.**
R: Possíveis causas:
1. Dataset muito pequeno (<20 clips): adicione mais
2. Áudio de má qualidade: re-grave melhor
3. Treino incompleto: aumente `epochs`
4. Texto muito diferente do dataset: use frases similares

**P: Erro: "Nenhum WAV de referência encontrado"**
R: Significa que `dataset/wavs/` está vazia ou sem arquivos.
- Verifique se os WAVs estão em `dataset/wavs/`
- Rode `prepare_dataset.py` primeiro

**P: Erro: "CUDA out of memory"**
R: Reduza em `train_xtts.py`:
```python
trainer.train(
    epochs=20,
    learning_rate=1e-4,
    batch_size=2,  # De 4 para 2
)
```
Ou reduza `epochs` para 10.

**P: Treino abortou no meio, como recupero?**
R: Checkpoints são salvos durante treino. Rode `train_xtts.py` novamente com mais `epochs` e ele vai continuar do melhor checkpoint.

**P: Inferência está muito lenta.**
R: Esperado na CPU (~30s). Com GPU deve ser <5s.
- Verifique se CUDA foi detectado: rode `install.py` novamente
- Atualize driver NVIDIA
- Se ainda lento, considere usar CPU só para CPU e parar GPU (deixe em CPU mesmo)

---

## Avançado

**P: Posso usar modelo XTTS v2 pré-treinado sem fine-tuning?**
R: Sim, edite `inference.py` e comente a linha de carregamento do checkpoint.

**P: Posso exportar o modelo fine-tunado?**
R: Os checkpoints estão em `checkpoints/*.pt`. Você pode compartilhá-los.

**P: Posso treinar com múltiplos speakers/vozes?**
R: Não nesta versão. Projeto é single-speaker.

**P: Suporta TTS multilíngue?**
R: Sim, XTTS v2 suporta. Edite `language="pt"` em `inference.py` para outro idioma.

**P: Posso usar gradio para interface visual?**
R: Sim, TTS tem suporte nativo. Veja documentação oficial do Coqui TTS.

---

## Performance

**P: Quantos modelos posso treinar ao mesmo tempo?**
R: Um por GPU. Rodando dois ao mesmo tempo vai dar CUDA out of memory.

**P: Vale a pena usar GPU RTX 3060 vs RTX 4070?**
R: RTX 3060 (6GB) é o mínimo. RTX 4070 é mais rápido mas overkill para isso.

**P: CPU i5-10400 consegue treinar?**
R: Sim, mas vai demorar. Esperado: 2-3 horas (vs 20 min na GPU).

---

## Suporte

Verifique também:
- [README.md](README.md) - Instruções principais
- [USAGE.md](USAGE.md) - Guia passo-a-passo
- Documentação oficial: https://docs.coqui.ai/en/latest/models/xtts.html
