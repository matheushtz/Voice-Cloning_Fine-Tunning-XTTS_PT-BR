# Exemplos de Uso - XTTS v2 Fine-Tuning

## Exemplos de Inferência

### Exemplo 1: Texto Simples

No arquivo `inference.py`, procure por:

```python
text = "Boa tarde, Torneirinha"
```

E mude para:

```python
text = "Olá, como você está?"
```

Depois rode:
```powershell
python inference.py
```

### Exemplo 2: Múltiplas Frases

Crie um arquivo `batch_inference.py`:

```python
from pathlib import Path
import logging
from inference import XttsInference

logging.basicConfig(level=logging.INFO)

texts = [
    "Boa tarde, Torneirinha",
    "Olá mundo",
    "Teste de síntese de fala",
    "XTTS é incrível",
]

project_root = Path(__file__).resolve().parent
checkpoint_dir = project_root / "checkpoints"

inferencer = XttsInference(checkpoint_dir=checkpoint_dir)

for i, text in enumerate(texts, 1):
    output_file = project_root / "output" / f"output_{i:02d}.wav"
    print(f"\nGerando {i}/{len(texts)}: {text}")
    inferencer.synthesize(text=text, output_path=output_file)
    print(f"Salvo: {output_file}")
```

Rode:
```powershell
python batch_inference.py
```

### Exemplo 3: Diferentes Idiomas

Para usar outro idioma, edite `inference.py`:

```python
# Procure por:
output = self.model.synthesize(
    text=text,
    speaker_wav=speaker_wav,
    language="pt",  # ← Mude aqui
    split_sentences=True,
)
```

Idiomas suportados:
- `"pt"` - Português
- `"en"` - Inglês
- `"es"` - Espanhol
- `"fr"` - Francês
- `"de"` - Alemão
- `"it"` - Italiano
- `"pl"` - Polonês
- `"tr"` - Turco
- `"ru"` - Russo
- `"nl"` - Holandês
- `"ja"` - Japonês
- `"zh"` - Chinês
- `"ar"` - Árabe
- `"hi"` - Hindi
- `"ko"` - Coreano

Exemplo Inglês:
```python
text = "Hello, how are you today?"
output = self.model.synthesize(
    text=text,
    speaker_wav=speaker_wav,
    language="en",
    split_sentences=True,
)
```

### Exemplo 4: Customizar Taxa de Voz

```python
# Em inference.py, procure por synthesize() e adicione:
output = self.model.synthesize(
    text=text,
    speaker_wav=speaker_wav,
    language="pt",
    split_sentences=True,
    # Customize aqui:
    # temperature=0.85,  # Variação (0.5-1.0)
    # top_p=0.85,        # Diversidade
)
```

---

## Exemplos de Dataset

### Exemplo 1: Adicionar Novo Clip

1. Grave um áudio seu em `dataset/wavs/0089.wav`
2. Abra `dataset/metadata.csv` e adicione ao final:
   ```
   0089|Meu novo texto aqui
   ```
3. Rode:
   ```powershell
   python prepare_dataset.py
   python train_xtts.py
   ```

### Exemplo 2: Validar Seu Dataset

```powershell
python prepare_dataset.py
```

Output esperado:
```
RELATORIO DE VALIDACAO DO DATASET
====================================================
Total de clips: 88
Clips validos: 88
Clips convertidos: 0
Duracao total: 234.45s (3.91min)

✓ Dataset valido com 88 clips
```

### Exemplo 3: Convertir Formato de Áudio

Se os WAVs estão em outro formato/sample rate:

```powershell
python prepare_dataset.py
```

Ele converte automaticamente para 24kHz mono PCM16.

---

## Exemplos Avançados

### Exemplo 1: Configurar Treino Customizado

Edite `train_xtts.py`:

```python
# No final do arquivo:
if __name__ == "__main__":
    # ... código anterior ...
    trainer.train(
        epochs=50,              # Mais epocas
        learning_rate=5e-5,     # Learning rate menor
        batch_size=2,           # Batch menor para precisão
    )
```

### Exemplo 2: Usar Múltiplos Speaker WAVs

```python
# Em inference.py:
def get_speaker_wavs(self) -> List[str]:
    """Retorna múltiplos WAVs de referência."""
    wavs_dir = self.dataset_dir / "wavs"
    wav_files = sorted(wavs_dir.glob("*.wav"))[:3]  # Primeiros 3
    return [str(f) for f in wav_files]

# No synthesize:
speaker_wavs = self.get_speaker_wavs()
output = self.model.synthesize(
    text=text,
    speaker_wav=speaker_wavs,
    language="pt",
)
```

### Exemplo 3: Monitorar Treino em Tempo Real

```python
# Adicione em train_xtts.py após criar trainer:
import tensorboard
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/braum_training")

# Em _train_epoch:
for batch_idx, (loss) in enumerate(...):
    writer.add_scalar("loss/train", loss.item(), epoch * len(dataloader) + batch_idx)

# Depois rode:
# tensorboard --logdir=runs/
```

### Exemplo 4: Salvar Áudio com Metadados

```python
# Crie save_audio_with_metadata.py:
import soundfile as sf
import json
from pathlib import Path

def save_with_metadata(wav_data, sr, output_path, text, model_name):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Salva WAV
    sf.write(str(output_path), wav_data, sr, subtype="PCM_16")
    
    # Salva metadados JSON
    metadata = {
        "text": text,
        "model": model_name,
        "sample_rate": sr,
        "duration_sec": len(wav_data) / sr,
    }
    
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Salvo: {output_path} + {json_path}")
```

---

## Integração em Aplicações

### Exemplo 1: API Flask

```python
# app.py
from flask import Flask, request, jsonify, send_file
from inference import XttsInference
from pathlib import Path
import io

app = Flask(__name__)

checkpoint_dir = Path("checkpoints")
inferencer = XttsInference(checkpoint_dir)

@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.json
    text = data.get("text", "Olá mundo")
    
    output_file = Path("temp_output.wav")
    inferencer.synthesize(text=text, output_path=output_file)
    
    return send_file(output_file, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

Teste:
```bash
curl -X POST http://localhost:5000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Boa tarde, mundo"}'
```

### Exemplo 2: Integração com Gradio

```python
# gradio_app.py
import gradio as gr
from inference import XttsInference
from pathlib import Path

checkpoint_dir = Path("checkpoints")
inferencer = XttsInference(checkpoint_dir)

def tts(text):
    output_file = Path("temp.wav")
    inferencer.synthesize(text=text, output_path=output_file)
    return str(output_file)

demo = gr.Interface(
    fn=tts,
    inputs="text",
    outputs="audio",
    title="XTTS v2 - Braum TTS",
)

if __name__ == "__main__":
    demo.launch()
```

Rode:
```powershell
pip install gradio
python gradio_app.py
```

---

## Scripts Úteis

### Script 1: Converter WAV para MP3

```python
# wav_to_mp3.py
import subprocess
from pathlib import Path

input_wav = Path("output/output.wav")
output_mp3 = Path("output/output.mp3")

subprocess.run([
    "ffmpeg", "-i", str(input_wav),
    "-b:a", "192k",
    str(output_mp3), "-y"
])

print(f"Convertido: {output_mp3}")
```

### Script 2: Medir Qualidade do Áudio

```python
# measure_quality.py
import librosa
import numpy as np
from pathlib import Path

wav_file = Path("output/output.wav")
y, sr = librosa.load(str(wav_file))

# Métricas
duration = librosa.get_duration(y=y, sr=sr)
rms = np.sqrt(np.mean(y**2))
peak = np.max(np.abs(y))
silence = np.sum(np.abs(y) < 0.01) / len(y) * 100

print(f"Duração: {duration:.2f}s")
print(f"RMS: {rms:.4f}")
print(f"Peak: {peak:.4f}")
print(f"Silêncio: {silence:.1f}%")
```

---

Mais exemplos e documentação: https://docs.coqui.ai/
