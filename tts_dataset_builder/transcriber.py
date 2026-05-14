from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm

from config import RuntimeConfig
from segmenter import AudioSegment


@dataclass(slots=True)
class TranscriptionResult:
    segment_id: str
    wav_path: Path
    text: str
    duration_sec: float
    error: Optional[str] = None


class WhisperTranscriber:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        device, compute_type = self._resolve_device_and_compute_type(
            requested_device=config.whisper_device,
            requested_compute=config.whisper_compute_type,
        )

        self.logger.info(
            "Inicializando Faster-Whisper: model=%s device=%s compute_type=%s",
            config.whisper_model,
            device,
            compute_type,
        )

        self.model = WhisperModel(
            config.whisper_model,
            device=device,
            compute_type=compute_type,
            download_root=str(config.models_dir),
        )

    def transcribe_segments(
        self,
        segments: Iterable[AudioSegment],
    ) -> Tuple[List[TranscriptionResult], int]:
        results: List[TranscriptionResult] = []
        errors = 0

        segment_list = list(segments)
        for segment in tqdm(segment_list, desc="Transcrevendo", unit="segmento"):
            try:
                text = self._transcribe_file(segment.wav_path)
                if not text:
                    continue
                results.append(
                    TranscriptionResult(
                        segment_id=segment.segment_id,
                        wav_path=segment.wav_path,
                        text=text,
                        duration_sec=segment.duration_sec,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                errors += 1
                self.logger.exception(
                    "Falha ao transcrever %s: %s",
                    segment.wav_path.name,
                    exc,
                )
                results.append(
                    TranscriptionResult(
                        segment_id=segment.segment_id,
                        wav_path=segment.wav_path,
                        text="",
                        duration_sec=segment.duration_sec,
                        error=str(exc),
                    )
                )

        return results, errors

    def _transcribe_file(self, wav_path: Path) -> str:
        segments, _info = self.model.transcribe(
            str(wav_path),
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.72,
            compression_ratio_threshold=2.5,
        )

        text_parts: List[str] = []
        for seg in segments:
            if seg.text:
                text_parts.append(seg.text.strip())

        return " ".join(text_parts).strip()

    def _resolve_device_and_compute_type(
        self,
        requested_device: str,
        requested_compute: str,
    ) -> Tuple[str, str]:
        device = requested_device.lower().strip()
        compute = requested_compute.lower().strip()

        cuda_available = torch.cuda.is_available()

        if device == "auto":
            device = "cuda" if cuda_available else "cpu"

        if device == "cuda" and not cuda_available:
            self.logger.warning("CUDA solicitado, mas indisponivel. Fazendo fallback para CPU.")
            device = "cpu"

        if compute == "auto":
            if device == "cuda":
                compute = "float16"
            else:
                compute = "int8"

        return device, compute
