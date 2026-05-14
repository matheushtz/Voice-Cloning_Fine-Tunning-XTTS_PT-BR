"""
Script de inferencia com modelo XTTS v2 fine-tunado.
Gera audio sintetizado com a voz customizada do Braum.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torchaudio
from TTS.api import TTS


def setup_logging(log_dir: Path) -> logging.Logger:
    """Configura logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


logger = setup_logging(Path(__file__).resolve().parent / "output")

TARGET_SR = 24000


def patch_torchaudio_loader() -> None:
    """Faz fallback para soundfile quando torchaudio exigir TorchCodec."""
    original_load = torchaudio.load

    def _safe_load(filepath, *args, **kwargs):
        try:
            return original_load(filepath, *args, **kwargs)
        except Exception as exc:
            message = str(exc)
            if (
                "TorchCodec is required" not in message
                and "load_with_torchcodec" not in message
                and "libtorchcodec" not in message
            ):
                raise

            logger.warning("Falha no torchaudio.load com TorchCodec; usando fallback via soundfile.")
            audio_np, sample_rate = sf.read(filepath, dtype="float32", always_2d=True)
            audio = torch.from_numpy(audio_np.T)
            return audio, sample_rate

    torchaudio.load = _safe_load


patch_torchaudio_loader()


class XttsInference:
    """Inferencia com XTTS v2 fine-tunado."""

    def __init__(
        self,
        checkpoint_dir: Path,
        dataset_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = Path(__file__).resolve().parent / "output"
        self.training_dir = Path(__file__).resolve().parent / "xtts_training"
        self.dataset_dir = dataset_dir or Path(__file__).resolve().parent.parent / "dataset"

        # Detecta device com fallback para CPU.
        requested_device = (device or "auto").lower()
        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                logger.warning("CUDA solicitada, mas indisponivel. Usando CPU.")
                self.device = "cpu"
        else:
            self.device = "cpu"

        logger.info(f"Device: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Carrega checkpoint fine-tunado se disponivel, senao usa o modelo base.
        self.model = self._load_model()

        logger.info("Modelo pronto para inferencia")

    def _instantiate_tts(self, model_name: Optional[str] = None, model_path: Optional[Path] = None, config_path: Optional[Path] = None) -> TTS:
        """Inicializa TTS no device atual, com fallback para CPU se necessario."""
        kwargs = {"gpu": self.device == "cuda"}
        if model_name is not None:
            kwargs["model_name"] = model_name
        else:
            kwargs["model_path"] = str(model_path)
            kwargs["config_path"] = str(config_path)

        try:
            return TTS(**kwargs)
        except Exception as exc:
            if self.device != "cuda":
                raise

            logger.warning(f"Falha ao inicializar modelo na GPU: {exc}")
            logger.warning("Aplicando fallback para CPU...")
            self.device = "cpu"

            kwargs["gpu"] = False
            return TTS(**kwargs)

    def _load_model(self) -> TTS:
        """Carrega o modelo XTTS base ou o checkpoint fine-tunado local."""
        model_dir, config_path = self._prepare_checkpoint_dir()

        if model_dir is not None and config_path is not None:
            logger.info(f"Carregando checkpoint fine-tunado: {model_dir}")
            try:
                return self._instantiate_tts(
                    model_path=model_dir,
                    config_path=config_path,
                )
            except Exception as exc:
                logger.warning(f"Nao foi possivel carregar checkpoint local: {exc}")
                logger.warning("Usando modelo base XTTS v2")

        logger.info("Carregando modelo XTTS v2 base...")
        return self._instantiate_tts(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

    def _resolve_checkpoint_files(self) -> tuple[Optional[Path], Optional[Path]]:
        """Localiza o checkpoint mais recente gerado pelo treino."""

        direct_best = self.checkpoint_dir / "best_model.pth"
        direct_config = self.checkpoint_dir / "config.json"
        if direct_best.exists() and direct_config.exists():
            return self.checkpoint_dir, direct_config

        training_root = self.training_dir / "run" / "training"
        if not training_root.exists():
            return None, None

        candidates: list[Path] = []
        for run_dir in sorted(training_root.iterdir(), reverse=True):
            if not run_dir.is_dir() or not run_dir.name.startswith("GPT_XTTS_FT-"):
                continue

            best_model = run_dir / "best_model.pth"
            config_path = run_dir / "config.json"
            if best_model.exists() and config_path.exists():
                candidates.append(run_dir)

        if not candidates:
            return None, None

        chosen_run = candidates[0]
        return chosen_run, chosen_run / "config.json"

    def _prepare_checkpoint_dir(self) -> tuple[Optional[Path], Optional[Path]]:
        """Cria um diretorio de inferencia no formato esperado pelo carregador XTTS."""
        run_dir, config_path = self._resolve_checkpoint_files()
        if run_dir is None or config_path is None:
            return None, None

        prepared_dir = self.output_dir / "xtts_inference_checkpoint"
        prepared_dir.mkdir(parents=True, exist_ok=True)

        source_model = run_dir / "best_model.pth"
        target_model = prepared_dir / "model.pth"
        shutil.copy2(source_model, target_model)
        shutil.copy2(config_path, prepared_dir / "config.json")

        vocab_source = self.training_dir / "run" / "training" / "XTTS_v2.0_original_model_files" / "vocab.json"
        if vocab_source.exists():
            shutil.copy2(vocab_source, prepared_dir / "vocab.json")

        return prepared_dir, prepared_dir / "config.json"

    def get_speaker_wav(self) -> Optional[str]:
        """Seleciona um WAV de referencia para a voz do Braum."""
        # Prioriza braum.wav em tts_dataset_builder/input/wav/
        braum_wav = Path(__file__).resolve().parent / "input" / "wav" / "braum.wav"
        if braum_wav.exists():
            logger.info(f"Usando voz de referencia: {braum_wav.name}")
            return str(braum_wav)
        
        logger.warning(f"braum.wav nao encontrado em {braum_wav}")
        
        # Fallback: procura por 'wavs' ou 'wav' no dataset
        if (self.dataset_dir / "wavs").exists():
            wavs_dir = self.dataset_dir / "wavs"
        elif (self.dataset_dir / "wav").exists():
            wavs_dir = self.dataset_dir / "wav"
        else:
            logger.warning(f"Nenhuma pasta 'wavs' ou 'wav' encontrada em {self.dataset_dir}")
            return None

        if not wavs_dir.exists():
            logger.warning(f"Pasta wavs nao encontrada: {wavs_dir}")
            return None

        # Seleciona o primeiro WAV como referencia
        wav_files = sorted(wavs_dir.glob("*.wav"))
        if wav_files:
            speaker_wav = str(wav_files[0])
            logger.info(f"Usando voz de referencia (fallback): {wav_files[0].name}")
            return speaker_wav

        logger.warning("Nenhum arquivo WAV encontrado em fallback")
        return None

    def _detect_model_speaker(self) -> Optional[str]:
        """Escolhe um speaker interno do modelo, quando disponivel."""
        speakers = getattr(self.model, "speakers", None)
        if not speakers:
            return None

        if isinstance(speakers, dict):
            candidates = [str(key) for key in speakers.keys()]
        elif isinstance(speakers, (list, tuple, set)):
            candidates = [str(item) for item in speakers]
        else:
            return None

        if not candidates:
            return None

        preferred_names = ["braum", "default", "speaker_0", "0"]
        lowered = {name.lower(): name for name in candidates}
        for pref in preferred_names:
            if pref in lowered:
                chosen = lowered[pref]
                logger.info(f"Usando speaker do modelo: {chosen}")
                return chosen

        chosen = candidates[0]
        logger.info(f"Usando speaker do modelo: {chosen}")
        return chosen
    
    def synthesize(self, text: str, output_path: Path, language: str = "pt", speed: float = 1.0) -> None:
        """Sintetiza audio a partir de texto."""
        logger.info(f"Sintetizando: {text}")
        # Sempre usamos WAV de referencia (XTTS fine-tuned normalmente precisa deste arquivo)
        speaker_wav = self.get_speaker_wav()
        if not speaker_wav:
            logger.error("Nenhum WAV de referencia encontrado")
            raise FileNotFoundError("Nenhum arquivo de voz de referencia disponivel")

        try:
            start_time = time.time()

            # Coqui XTTS usa speed > 1.0 para acelerar e < 1.0 para desacelerar.
            tts_kwargs = {
                "text": text,
                "speaker_wav": speaker_wav,
                "language": language,
                "file_path": str(output_path),
                "speed": speed,
            }

            try:
                # Gera audio em arquivo usando a API publica do Coqui TTS.
                self.model.tts_to_file(**tts_kwargs)
            except TypeError as exc:
                # Compatibilidade com versoes antigas da lib sem argumento `speed`.
                if "speed" not in str(exc):
                    raise

                logger.warning("Versao atual da biblioteca nao suporta `speed`; gerando com velocidade padrao.")
                tts_kwargs.pop("speed", None)
                self.model.tts_to_file(**tts_kwargs)

            elapsed = time.time() - start_time
            logger.info(f"Tempo de sintese: {elapsed:.2f}s")

            logger.info(f"Audio salvo: {output_path}")
            logger.info("Audio gerado via TTS API")

        except Exception as e:
            logger.error(f"Erro durante sintese: {e}")
            raise


def sanitize_filename(text: str, max_length: int = 13) -> str:
    """Sanitiza os primeiros caracteres do texto para usar como nome de arquivo."""
    # Remove caracteres inválidos para nomes de arquivo
    sanitized = re.sub(r'[<>:"/\\|?*]', '', text[:max_length])
    sanitized = sanitized.strip()
    if not sanitized:
        sanitized = "output"
    return sanitized


def parse_args() -> argparse.Namespace:
    """Le argumentos de linha de comando para a inferencia."""
    parser = argparse.ArgumentParser(description="Executa inferencia XTTS v2 com voz fine-tunada.")
    parser.add_argument(
        "--text",
        "-t",
        required=True,
        help="Texto a ser sintetizado (max 203 caracteres para lingua 'pt').",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Arquivo de saida WAV (relativo a tts_dataset_builder ou caminho absoluto). Se nao especificado, usa os 13 primeiros caracteres do texto.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Dispositivo de execucao. Em 'auto', usa GPU se disponivel e fallback para CPU.",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="pt",
        help="Codigo de idioma para sintese (ex: pt, en, es).",
    )
    parser.add_argument(
        "--speed",
        "-s",
        type=float,
        default=1.15,
        help="Velocidade da fala. 1.0 = normal, >1.0 mais rapido, <1.0 mais lento.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Validar tamanho do texto
    MAX_TEXT_LENGTH = 203
    if len(args.text) > MAX_TEXT_LENGTH:
        logger.error(f"Texto muito longo: {len(args.text)} caracteres (maximo: {MAX_TEXT_LENGTH})")
        logger.error(f"Truncue o texto para no maximo {MAX_TEXT_LENGTH} caracteres.")
        sys.exit(1)
    
    project_root = Path(__file__).resolve().parent.parent  # Volta para raiz
    
    # Detecta se dataset está em raiz ou em tts_dataset_builder
    dataset_dir = project_root / "dataset"
    if not dataset_dir.exists():
        dataset_dir = Path(__file__).resolve().parent / "dataset"
    
    output_dir = Path(__file__).resolve().parent / "output"
    training_dir = Path(__file__).resolve().parent / "xtts_training"
    
    # Define nome do arquivo de saida
    if args.output:
        output_file = Path(args.output)
    else:
        # Usa os primeiros 13 caracteres do texto como nome
        filename = sanitize_filename(args.text) + ".wav"
        output_file = output_dir / filename
    
    if not output_file.is_absolute():
        output_file = Path(__file__).resolve().parent / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("XTTS v2 - INFERENCIA COM VOZ FINE-TUNADA")
    logger.info("=" * 60)
    logger.info(f"Texto: {args.text}")
    logger.info(f"Tamanho do texto: {len(args.text)} caracteres")
    logger.info(f"Idioma: {args.language}")
    logger.info(f"Velocidade: {args.speed}")
    logger.info(f"Device solicitado: {args.device}")

    try:
        inferencer = XttsInference(checkpoint_dir=training_dir, dataset_dir=dataset_dir, device=args.device)

        inferencer.synthesize(
            text=args.text,
            output_path=output_file,
            language=args.language,
            speed=args.speed,
        )

        logger.info("\n" + "=" * 60)
        logger.info("SUCESSO!")
        logger.info(f"Output: {output_file}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
