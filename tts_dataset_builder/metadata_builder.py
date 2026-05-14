from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from transcriber import TranscriptionResult


@dataclass(slots=True)
class MetadataStats:
    rows_written: int
    skipped_empty: int


class MetadataBuilder:
    SMALL_NUMBERS_PT = {
        0: "zero",
        1: "um",
        2: "dois",
        3: "tres",
        4: "quatro",
        5: "cinco",
        6: "seis",
        7: "sete",
        8: "oito",
        9: "nove",
        10: "dez",
        11: "onze",
        12: "doze",
        13: "treze",
        14: "quatorze",
        15: "quinze",
        16: "dezesseis",
        17: "dezessete",
        18: "dezoito",
        19: "dezenove",
        20: "vinte",
    }

    def __init__(self, convert_numbers: bool = False) -> None:
        self.convert_numbers = convert_numbers

    def build_and_export(
        self,
        transcriptions: Iterable[TranscriptionResult],
        output_dir: Path,
        speaker_id: str = "speaker_0",
    ) -> MetadataStats:
        rows_piper: List[Tuple[str, str]] = []
        rows_coqui: List[Tuple[str, str, str]] = []
        rows_vits: List[Tuple[str, str]] = []

        skipped_empty = 0

        for item in transcriptions:
            if item.error:
                continue

            cleaned = self.clean_text(item.text)
            if not cleaned:
                skipped_empty += 1
                continue

            clip_id = item.wav_path.stem
            rel_path = f"wavs/{item.wav_path.name}"

            rows_piper.append((clip_id, cleaned))
            rows_coqui.append((rel_path, cleaned, speaker_id))
            rows_vits.append((rel_path, cleaned))

        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_pipe_csv(
            output_dir / "metadata.csv",
            rows_piper,
            columns=["id", "text"],
        )

        self._write_pipe_csv(
            output_dir / "metadata_coqui.csv",
            rows_coqui,
            columns=["path", "text", "speaker"],
        )

        self._write_pipe_csv(
            output_dir / "filelist_vits.txt",
            rows_vits,
            columns=["path", "text"],
        )

        return MetadataStats(rows_written=len(rows_piper), skipped_empty=skipped_empty)

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = self._remove_emojis(text)
        text = text.replace("\n", " ").replace("\t", " ")

        # Mantem letras (incluindo acentos), numeros e pontuacao comum.
        text = re.sub(r"[^\w\s\.,!\?;:\-\'\"()À-ÖØ-öø-ÿ]", " ", text, flags=re.UNICODE)
        text = re.sub(r"\s+", " ", text, flags=re.UNICODE).strip()

        if self.convert_numbers:
            text = self._convert_simple_numbers(text)

        return text

    def _remove_emojis(self, text: str) -> str:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)

    def _convert_simple_numbers(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            value = int(match.group(0))
            if value in self.SMALL_NUMBERS_PT:
                return self.SMALL_NUMBERS_PT[value]
            return match.group(0)

        return re.sub(r"\b\d{1,2}\b", repl, text)

    def _write_pipe_csv(
        self,
        output_path: Path,
        rows: List[Tuple[str, ...]],
        columns: List[str],
    ) -> None:
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(
            output_path,
            sep="|",
            index=False,
            header=False,
            encoding="utf-8",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
