from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf


def check_ffmpeg_available() -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return True
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def load_audio_mono(audio_path: Path, target_sr: int) -> np.ndarray:
    samples, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    return samples.astype(np.float32)


def trim_silence(samples: np.ndarray, top_db: int = 38) -> np.ndarray:
    if samples.size == 0:
        return samples
    trimmed, _ = librosa.effects.trim(samples, top_db=top_db)
    return trimmed.astype(np.float32)


def normalize_peak(samples: np.ndarray, target_peak_dbfs: float = -1.0) -> np.ndarray:
    if samples.size == 0:
        return samples
    peak = np.max(np.abs(samples))
    if peak <= 1e-8:
        return samples
    target_peak = 10 ** (target_peak_dbfs / 20.0)
    gain = target_peak / peak
    normalized = samples * gain
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def compute_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float32))))


def apply_fade(samples: np.ndarray, sr: int, fade_ms: int = 12) -> np.ndarray:
    if samples.size == 0:
        return samples
    fade_len = int(sr * fade_ms / 1000)
    if fade_len <= 0 or fade_len * 2 >= samples.size:
        return samples

    faded = samples.copy()
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    faded[:fade_len] *= fade_in
    faded[-fade_len:] *= fade_out
    return faded


def write_wav_pcm16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="PCM_16")


def sec_to_samples(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def samples_to_sec(sample_count: int, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return sample_count / sample_rate


def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def slice_with_padding(
    samples: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    pad_sec: float,
) -> Tuple[np.ndarray, float, float]:
    total_dur = samples_to_sec(len(samples), sr)
    padded_start = max(0.0, start_sec - pad_sec)
    padded_end = min(total_dur, end_sec + pad_sec)

    start_idx = sec_to_samples(padded_start, sr)
    end_idx = sec_to_samples(padded_end, sr)
    chunk = samples[start_idx:end_idx]
    return chunk, padded_start, padded_end


def safe_stem(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    return cleaned or "audio"
