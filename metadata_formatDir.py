from pathlib import Path

dataset = Path("dataset")
metadata_in = dataset / "metadata.csv"
metadata_out = dataset / "metadata_f.csv"

lines = metadata_in.read_text(encoding="utf-8").splitlines()

rows = []

for line in lines:
    parts = line.split("|", 1)
    if len(parts) != 2:
        continue

    wav_id = parts[0].strip()
    text = parts[1].strip()

    audio_file = f"wav/{wav_id}.wav"
    rows.append((audio_file, text))

# escreve com HEADER (isso é o que faltava)
with open(metadata_out, "w", encoding="utf-8") as f:
    f.write("audio_file|text\n")
    for audio_file, text in rows:
        f.write(f"{audio_file}|{text}\n")

print("metadata_f.csv criado no formato correto")