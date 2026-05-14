from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from silero_vad import get_speech_timestamps, load_silero_vad

from audio_utils import (
    apply_fade,
    compute_rms,
    load_audio_mono,
    normalize_peak,
    safe_stem,
    samples_to_sec,
    slice_with_padding,
    trim_silence,
    write_wav_pcm16,
)
from config import RuntimeConfig


@dataclass(slots=True)
class AudioSegment:
    segment_id: str
    wav_path: Path
    start_sec: float
    end_sec: float
    duration_sec: float
    source_file: Path


class AudioSegmenter:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vad_model = load_silero_vad(onnx=True)

    def segment_audio(self, audio_path: Path, start_index: int = 1) -> List[AudioSegment]:
        if not audio_path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {audio_path}")

        self.logger.info("Carregando audio para VAD: %s", audio_path.name)
        vad_samples = load_audio_mono(audio_path, target_sr=self.config.vad_sample_rate)

        speech_timestamps = get_speech_timestamps(
            vad_samples,
            self.vad_model,
            sampling_rate=self.config.vad_sample_rate,
            min_speech_duration_ms=220,
            min_silence_duration_ms=120,
            speech_pad_ms=35,
            return_seconds=False,
        )

        if not speech_timestamps:
            self.logger.warning("Nenhum segmento de fala detectado em %s", audio_path.name)
            return []

        raw_intervals = [
            (
                ts["start"] / self.config.vad_sample_rate,
                ts["end"] / self.config.vad_sample_rate,
            )
            for ts in speech_timestamps
        ]

        merged = self._merge_intervals(raw_intervals, max_gap_sec=0.22)
        split_intervals = self._split_intervals(merged)

        out_sr = self.config.output_sample_rate
        source_samples = load_audio_mono(audio_path, target_sr=out_sr)

        segments: List[AudioSegment] = []
        idx = start_index

        for start_sec, end_sec in split_intervals:
            chunk, padded_start, padded_end = slice_with_padding(
                source_samples,
                out_sr,
                start_sec,
                end_sec,
                self.config.pad_sec,
            )

            if self.config.trim_segment_silence:
                chunk = trim_silence(chunk, top_db=self.config.top_db_trim)

            duration = samples_to_sec(len(chunk), out_sr)
            if duration < self.config.min_segment_sec:
                continue

            if compute_rms(chunk) < self.config.min_rms:
                continue

            chunk = normalize_peak(chunk, target_peak_dbfs=-1.0)
            chunk = apply_fade(chunk, sr=out_sr, fade_ms=12)

            segment_id = f"{idx:04d}"
            wav_path = self.config.wavs_dir / f"{segment_id}.wav"
            write_wav_pcm16(wav_path, chunk, out_sr)

            segments.append(
                AudioSegment(
                    segment_id=segment_id,
                    wav_path=wav_path,
                    start_sec=padded_start,
                    end_sec=padded_end,
                    duration_sec=duration,
                    source_file=audio_path,
                )
            )
            idx += 1

        return segments

    def _merge_intervals(
        self,
        intervals: Sequence[Tuple[float, float]],
        max_gap_sec: float,
    ) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged: List[Tuple[float, float]] = [sorted_intervals[0]]

        for current_start, current_end in sorted_intervals[1:]:
            prev_start, prev_end = merged[-1]
            if current_start - prev_end <= max_gap_sec:
                merged[-1] = (prev_start, max(prev_end, current_end))
            else:
                merged.append((current_start, current_end))

        return merged

    def _split_intervals(
        self,
        intervals: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        min_sec = self.config.min_segment_sec
        max_sec = self.config.max_segment_sec

        chunks: List[Tuple[float, float]] = []

        for start, end in intervals:
            length = end - start
            if length < min_sec:
                continue

            if length <= max_sec:
                chunks.append((start, end))
                continue

            n_chunks = max(1, math.ceil(length / max_sec))
            target_chunk = length / n_chunks

            cursor = start
            for i in range(n_chunks):
                chunk_start = cursor
                if i == n_chunks - 1:
                    chunk_end = end
                else:
                    chunk_end = min(end, chunk_start + target_chunk)
                if chunk_end - chunk_start >= min_sec:
                    chunks.append((chunk_start, chunk_end))
                cursor = chunk_end

        return chunks
