from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from audio_utils import check_ffmpeg_available, format_duration
from config import RuntimeConfig
from metadata_builder import MetadataBuilder
from segmenter import AudioSegment, AudioSegmenter
from transcriber import WhisperTranscriber


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de preparacao de dataset de voz para Piper/Coqui/VITS."
    )
    parser.add_argument("--input", type=Path, help="Arquivo unico de audio")
    parser.add_argument("--input-dir", type=Path, help="Pasta com varios audios")
    parser.add_argument(
        "--glob",
        type=str,
        default="*.wav",
        help="Padrao de busca para --input-dir (ex: *.wav, *.mp3)",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Pasta de output")
    parser.add_argument("--models-dir", type=Path, default=None, help="Pasta de modelos")

    parser.add_argument("--language", type=str, default="pt", help="Idioma do Whisper")
    parser.add_argument("--whisper-model", type=str, default="small", help="Modelo Whisper")
    parser.add_argument(
        "--compute-type",
        type=str,
        default="auto",
        choices=["auto", "int8", "int8_float16", "float16", "float32"],
        help="Tipo de computacao do Faster-Whisper",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device para transcricao",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")

    parser.add_argument("--min-segment-sec", type=float, default=2.0)
    parser.add_argument("--max-segment-sec", type=float, default=10.0)
    parser.add_argument("--pad-sec", type=float, default=0.08)
    parser.add_argument(
        "--trim-segment-silence",
        action="store_true",
        help="Remove silencio do inicio/fim de cada segmento",
    )
    parser.add_argument(
        "--convert-numbers",
        action="store_true",
        help="Converte numeros simples para texto (0-20)",
    )

    return parser


def configure_logging(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    return log_path


def discover_input_files(single_input: Path | None, input_dir: Path | None, glob_pattern: str) -> List[Path]:
    files: List[Path] = []

    if single_input:
        files.append(single_input)

    if input_dir:
        files.extend(sorted(input_dir.glob(glob_pattern)))

    unique_files = []
    seen = set()
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(f)

    return unique_files


def run_pipeline(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parent
    config = RuntimeConfig.from_paths(
        project_root=project_root,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
    )
    config.ensure_dirs()

    config.language = args.language
    config.whisper_model = args.whisper_model
    config.whisper_compute_type = args.compute_type
    config.whisper_device = args.device
    config.beam_size = args.beam_size
    config.min_segment_sec = args.min_segment_sec
    config.max_segment_sec = args.max_segment_sec
    config.pad_sec = args.pad_sec
    config.trim_segment_silence = args.trim_segment_silence
    config.convert_numbers = args.convert_numbers

    log_path = configure_logging(config.logs_dir)
    logger = logging.getLogger("main")
    logger.info("Log em: %s", log_path)

    if not check_ffmpeg_available():
        logger.warning("ffmpeg nao encontrado no PATH. Alguns formatos podem falhar.")

    input_files = discover_input_files(args.input, args.input_dir, args.glob)
    if not input_files:
        logger.error("Nenhum arquivo de entrada encontrado. Use --input ou --input-dir.")
        return 1

    for f in input_files:
        if not f.exists():
            logger.error("Arquivo nao existe: %s", f)
            return 1

    logger.info("Arquivos para processar: %d", len(input_files))

    segmenter = AudioSegmenter(config)
    transcriber = WhisperTranscriber(config)
    metadata_builder = MetadataBuilder(convert_numbers=config.convert_numbers)

    all_segments: List[AudioSegment] = []
    next_idx = 1

    for audio_file in input_files:
        logger.info("Segmentando: %s", audio_file)
        segments = segmenter.segment_audio(audio_file, start_index=next_idx)
        all_segments.extend(segments)
        next_idx += len(segments)
        logger.info("Segmentos criados para %s: %d", audio_file.name, len(segments))

    if not all_segments:
        logger.error("Nenhum segmento valido foi gerado.")
        return 2

    total_duration = sum(seg.duration_sec for seg in all_segments)
    logger.info(
        "Total de segmentos validos: %d | Duracao total: %s",
        len(all_segments),
        format_duration(total_duration),
    )

    transcriptions, transcription_errors = transcriber.transcribe_segments(all_segments)
    stats = metadata_builder.build_and_export(transcriptions, config.output_dir)

    logger.info("Erros de transcricao: %d", transcription_errors)
    logger.info("Linhas gravadas no metadata.csv: %d", stats.rows_written)
    logger.info("Linhas ignoradas (vazias): %d", stats.skipped_empty)
    logger.info("Dataset final pronto em: %s", config.output_dir)

    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = run_pipeline(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
