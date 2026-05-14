"""
Fine-tuning REAL do XTTS v2
Compatível com Coqui TTS 0.22.0
"""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent


def resolve_dataset_dir() -> Path:
    if (PROJECT_ROOT / "dataset").exists():
        return PROJECT_ROOT / "dataset"
    if (ROOT / "dataset").exists():
        return ROOT / "dataset"
    raise FileNotFoundError("Dataset nao encontrado")


def validate_dataset(dataset_dir: Path) -> tuple[Path, Path]:
    metadata = dataset_dir / "metadata.csv"

    if not metadata.exists():
        raise FileNotFoundError("metadata.csv nao encontrado")

    wavs_dir = dataset_dir / "wav"
    if not wavs_dir.exists():
        raise FileNotFoundError("Pasta wav/wavs nao encontrada")

    lines = metadata.read_text(encoding="utf-8").splitlines()
    valid = 0

    for line in lines:
        parts = line.split("|")
        if len(parts) < 2:
            continue

        wav_id = parts[0].strip()
        wav_file = wavs_dir / f"{wav_id}.wav"

        if wav_file.exists():
            valid += 1

    print(f"Arquivos validos: {valid}")

    if valid < 10:
        raise ValueError("Dataset muito pequeno")

    return metadata, wavs_dir


def run_training(dataset_dir: Path) -> None:
    output_dir = ROOT / "xtts_training"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("XTTS V2 FINE-TUNING")
    print("=" * 60)

    print("\n[1/4] Verificando modelo XTTS v2...")
    from TTS.api import TTS

    try:
        TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("Modelo XTTS carregado com sucesso")
    except Exception as exc:
        raise RuntimeError(f"Erro ao carregar XTTS: {exc}") from exc

    print("\n[2/4] Validando dataset...")
    _, wavs_dir = validate_dataset(dataset_dir)

    print("\n[3/4] Iniciando fine-tuning...")
    from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt

    train_gpt(
        language="pt",
        num_epochs=10,
        batch_size=2,
        grad_acumm=8,
        train_csv=str(dataset_dir / "metadata_f.csv"),
        eval_csv=str(dataset_dir / "metadata_f.csv"),
        output_path=str(output_dir),
    )

    print("\n[4/4] Gerando audio de teste...")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    tts.tts_to_file(
        text="Boa tarde, Torneirinha",
        file_path="braum_test.wav",
        speaker_wav=[str(wavs_dir / "0001.wav")],
        language="pt",
    )

    print("Arquivo gerado: braum_test.wav")
    print("\nTREINAMENTO CONCLUIDO")


def main() -> int:
    try:
        dataset_dir = resolve_dataset_dir()
        print(f"Dataset encontrado: {dataset_dir}")
        run_training(dataset_dir)
        return 0
    except Exception as exc:
        print(exc)
        return 1


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    sys.exit(main())