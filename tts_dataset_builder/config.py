from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class RuntimeConfig:
    project_root: Path
    output_dir: Path
    wavs_dir: Path
    logs_dir: Path
    models_dir: Path
    language: str = "pt"
    vad_sample_rate: int = 16000
    output_sample_rate: int = 22050
    min_segment_sec: float = 2.0
    max_segment_sec: float = 10.0
    pad_sec: float = 0.08
    min_rms: float = 0.003
    trim_segment_silence: bool = False
    top_db_trim: int = 38
    whisper_model: str = "small"
    whisper_compute_type: str = "auto"
    whisper_device: str = "auto"
    beam_size: int = 5
    convert_numbers: bool = False

    @classmethod
    def from_paths(
        cls,
        project_root: Path,
        output_dir: Optional[Path] = None,
        models_dir: Optional[Path] = None,
    ) -> "RuntimeConfig":
        output = output_dir or (project_root / "output")
        models = models_dir or (project_root / "models")
        return cls(
            project_root=project_root,
            output_dir=output,
            wavs_dir=output / "wavs",
            logs_dir=output / "logs",
            models_dir=models,
        )

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.wavs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
