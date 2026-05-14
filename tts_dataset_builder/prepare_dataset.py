"""
Script para preparacao e validacao do dataset.
Valida sample rate, bitrate e duracoes.
Converte automaticamente para formato compativel com XTTS v2.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm


def setup_logging() -> logging.Logger:
    """Configura logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()

# XTTS v2 requirements
TARGET_SR = 24000
TARGET_CHANNELS = 1  # mono


class DatasetValidator:
    """Valida e prepara dataset para XTTS v2."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        # Procura por 'wavs' ou 'wav'
        if (dataset_dir / "wavs").exists():
            self.wavs_dir = dataset_dir / "wavs"
        elif (dataset_dir / "wav").exists():
            self.wavs_dir = dataset_dir / "wav"
        else:
            raise FileNotFoundError(f"Nenhuma pasta 'wavs' ou 'wav' encontrada em {dataset_dir}")
        self.metadata_file = dataset_dir / "metadata.csv"
        self.stats = {
            "total_clips": 0,
            "valid_clips": 0,
            "converted": 0,
            "skipped": 0,
            "total_duration": 0.0,
            "errors": [],
        }

    def validate_structure(self) -> bool:
        """Valida estrutura basica do dataset."""
        logger.info("Validando estrutura do dataset...")

        if not self.dataset_dir.exists():
            logger.error(f"Dataset dir nao existe: {self.dataset_dir}")
            return False

        if not self.wavs_dir.exists():
            logger.error(f"Pasta wavs nao existe: {self.wavs_dir}")
            return False

        if not self.metadata_file.exists():
            logger.error(f"metadata.csv nao existe: {self.metadata_file}")
            return False

        logger.info(f"Dataset dir: {self.dataset_dir}")
        logger.info(f"Wavs dir: {self.wavs_dir}")
        logger.info(f"Metadata: {self.metadata_file}")
        return True

    def load_metadata(self) -> pd.DataFrame:
        """Carrega metadata.csv."""
        logger.info("Carregando metadata.csv...")
        try:
            df = pd.read_csv(
                self.metadata_file,
                sep="|",
                header=None,
                names=["id", "text"],
                encoding="utf-8",
                dtype={"id": str},
            )
            logger.info(f"Total de linhas no metadata: {len(df)}")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar metadata.csv: {e}")
            raise

    def validate_and_convert_audio(self, file_id: str, text: str) -> Tuple[bool, str]:
        """
        Valida e converte audio para formato XTTS v2.
        Retorna (sucesso, mensagem).
        """
        # Garante padding de 4 dígitos para o ID
        try:
            id_padded = f"{int(file_id):04d}"
        except (ValueError, TypeError):
            id_padded = file_id.zfill(4) if isinstance(file_id, str) else f"{int(file_id):04d}"
        wav_path = self.wavs_dir / f"{id_padded}.wav"

        if not wav_path.exists():
            msg = f"WAV nao existe: {wav_path}"
            logger.warning(msg)
            self.stats["errors"].append(msg)
            return False, msg

        try:
            # Carrega audio
            samples, sr = librosa.load(str(wav_path), sr=None, mono=False)

            # Verifica se esta vazio
            if samples.size == 0:
                msg = f"WAV vazio: {file_id}"
                logger.warning(msg)
                self.stats["errors"].append(msg)
                return False, msg

            # Converte para mono se necessario
            if len(samples.shape) > 1:
                samples = librosa.to_mono(samples)

            duration = librosa.get_duration(y=samples, sr=sr)

            # Valida duracao minima (1 segundo)
            if duration < 1.0:
                msg = f"WAV muito curto ({duration:.2f}s): {file_id}"
                logger.warning(msg)
                self.stats["errors"].append(msg)
                return False, msg

            # Resampling se necessario
            if sr != TARGET_SR:
                logger.info(f"Convertendo {file_id}: {sr}Hz -> {TARGET_SR}Hz")
                samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)

            # Normaliza peak
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = samples / max_val * 0.95

            # Salva como PCM16 mono
            sf.write(str(wav_path), samples, TARGET_SR, subtype="PCM_16")

            duration = len(samples) / TARGET_SR
            self.stats["total_duration"] += duration
            self.stats["valid_clips"] += 1
            self.stats["converted"] += 1

            return True, f"OK ({duration:.2f}s)"

        except Exception as e:
            msg = f"Erro ao processar {file_id}: {e}"
            logger.error(msg)
            self.stats["errors"].append(msg)
            return False, msg

    def run_validation(self) -> bool:
        """Executa validacao completa do dataset."""
        if not self.validate_structure():
            return False

        df = self.load_metadata()
        self.stats["total_clips"] = len(df)

        logger.info("\nProcessando WAVs...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Validando"):
            file_id = str(row["id"]).strip()
            text = str(row["text"]).strip()
            # Garante que file_id está em formato correto para validação
            try:
                file_id = f"{int(file_id):04d}"
            except (ValueError, TypeError):
                pass

            success, msg = self.validate_and_convert_audio(file_id, text)
            if not success:
                self.stats["skipped"] += 1

        return self._print_report()

    def _print_report(self) -> bool:
        """Imprime relatorio final."""
        print("\n" + "=" * 60)
        print("RELATORIO DE VALIDACAO DO DATASET")
        print("=" * 60)

        print(f"Total de clips: {self.stats['total_clips']}")
        print(f"Clips validos: {self.stats['valid_clips']}")
        print(f"Clips convertidos: {self.stats['converted']}")
        print(f"Clips pulados: {self.stats['skipped']}")
        print(f"Duracao total: {self.stats['total_duration']:.2f}s ({self.stats['total_duration']/60:.2f}min)")

        if self.stats["errors"]:
            print(f"\nErros encontrados: {len(self.stats['errors'])}")
            for error in self.stats["errors"][:10]:
                print(f"  - {error}")
            if len(self.stats["errors"]) > 10:
                print(f"  ... e mais {len(self.stats['errors']) - 10} erros")

        print("\n" + "=" * 60)

        success = self.stats["valid_clips"] > 0
        if success:
            print(f"✓ Dataset valido com {self.stats['valid_clips']} clips")
        else:
            print("✗ Dataset nao valido ou sem clips validos")

        return success


def main() -> None:
    # Se passou como argumento, usa; senão detecta automaticamente
    if len(sys.argv) > 1:
        dataset_dir = Path(sys.argv[1])
    else:
        # Tenta raiz primeiro, depois tts_dataset_builder
        project_root = Path(__file__).resolve().parent.parent
        dataset_dir = project_root / "dataset"
        if not dataset_dir.exists():
            dataset_dir = Path(__file__).resolve().parent / "dataset"

    logger.info(f"Dataset: {dataset_dir}")

    validator = DatasetValidator(dataset_dir)
    success = validator.run_validation()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
